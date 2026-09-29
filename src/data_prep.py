"""Day 1 data foundation for the R.O.A.D. Barbados HTR challenge.

What it does
  1. Loads Train.csv, Test.csv, SampleSubmission.csv
  2. Detects id / image-path / label / group columns (override with flags)
  3. Verifies every referenced image exists
  4. Image width / height / aspect-ratio stats (header read only, fast)
  5. Label length stats (chars, words, optional tokenizer tokens) incl. p95/p99
  6. Character-set audit (non-ASCII, long s, brackets, markers, whitespace, NFC)
  7. Finds page/document group candidates and checks train/test group overlap
  8. Writes a seeded validation split to data/splits/ (git-ignored)
  9. Writes an aggregate-only report to reports/day1_data_audit.md

The report contains statistics only, no transcriptions, so it is safe to commit.

Usage
  python src/data_prep.py
  python src/data_prep.py --image-root data/raw/images --group-col page_id
  python src/data_prep.py --tokenizer microsoft/trocr-large-handwritten
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
PCTS = (50, 90, 95, 99)


# ----------------------------------------------------------------- loading
def load_csvs(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    def find(name: str) -> Path:
        hits = [p for p in raw_dir.rglob("*.csv") if p.name.lower() == name.lower()]
        if not hits:
            sys.exit(f"[error] {name} not found under {raw_dir}")
        return hits[0]

    train = pd.read_csv(find("Train.csv"), encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(find("Test.csv"), encoding="utf-8-sig", keep_default_na=False)
    try:
        sample = pd.read_csv(find("SampleSubmission.csv"), encoding="utf-8-sig", keep_default_na=False)
    except SystemExit:
        sample = None
    return train, test, sample


# ------------------------------------------------------- column detection
def _looks_like_path(s: pd.Series) -> float:
    v = s.astype(str).str.lower()
    return float(v.str.endswith(IMG_EXTS).mean() + v.str.contains(r"[/\\]").mean() * 0.5)


def detect_columns(train: pd.DataFrame, test: pd.DataFrame, args) -> dict:
    cols = {}
    shared = [c for c in train.columns if c in test.columns]

    # id column
    id_cands = [c for c in shared if re.search(r"(^|_)id$|^id", c, re.I)]
    cols["id"] = args.id_col or (id_cands[0] if id_cands else shared[0])

    # image column: highest path-likeness among shared columns
    if args.image_col:
        cols["image"] = args.image_col
    else:
        scored = sorted(((_looks_like_path(train[c]), c) for c in shared), reverse=True)
        named = [c for c in shared if re.search(r"image|img|path|file", c, re.I)]
        cols["image"] = scored[0][1] if scored and scored[0][0] > 0.5 else (named[0] if named else cols["id"])

    # label column: train-only string column with the most spaces / longest text
    if args.label_col:
        cols["label"] = args.label_col
    else:
        train_only = [c for c in train.columns if c not in test.columns]
        cands = train_only or [c for c in train.columns if c not in (cols["id"], cols["image"])]
        textish = [c for c in cands if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
        if not textish:
            sys.exit(f"[error] could not detect label column; candidates: {cands}. Use --label-col")
        cols["label"] = max(textish, key=lambda c: train[c].astype(str).str.len().mean())
    return cols


# --------------------------------------------------------- image resolving
def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                p = Path(dirpath) / f
                index.setdefault(f, p)
                index.setdefault(Path(f).stem, p)
    return index


def resolve_paths(values: pd.Series, image_root: Path, index: dict[str, Path]) -> pd.Series:
    out = []
    for v in values.astype(str):
        p = Path(v)
        cand = [p, image_root / p]
        found = next((c for c in cand if c.is_file()), None)
        if found is None:
            found = index.get(p.name) or index.get(p.stem)
        out.append(str(found) if found else "")
    return pd.Series(out, index=values.index)


def image_stats(paths: pd.Series) -> pd.DataFrame:
    from PIL import Image

    rows = []
    for p in paths:
        if not p:
            rows.append((np.nan, np.nan, "")); continue
        try:
            with Image.open(p) as im:  # reads header only
                rows.append((im.width, im.height, im.mode))
        except Exception:
            rows.append((np.nan, np.nan, "unreadable"))
    df = pd.DataFrame(rows, columns=["width", "height", "mode"], index=paths.index)
    df["aspect"] = df["width"] / df["height"]
    return df


# ------------------------------------------------------------ label stats
def pct_table(s: pd.Series) -> dict:
    s = s.dropna()
    d = {"min": float(s.min()), "mean": round(float(s.mean()), 2)}
    d.update({f"p{p}": round(float(np.percentile(s, p)), 1) for p in PCTS})
    d["max"] = float(s.max())
    return d


def token_lengths(labels: pd.Series, tokenizer_name: str) -> pd.Series | None:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:  # offline laptop, missing package, etc.
        print(f"[warn] tokenizer '{tokenizer_name}' unavailable: {e}")
        return None
    ids = tok(labels.tolist(), add_special_tokens=True)["input_ids"]
    # round-trip check: does the tokenizer preserve every label?
    dec = tok.batch_decode(ids, skip_special_tokens=True)
    bad = sum(a != b for a, b in zip(labels.tolist(), dec))
    print(f"[info] tokenizer round-trip mismatches: {bad}/{len(labels)}")
    s = pd.Series([len(x) for x in ids], index=labels.index)
    s.attrs["roundtrip_mismatch"] = bad
    return s


def charset_audit(labels: pd.Series) -> dict:
    counts = Counter("".join(labels))
    def info(ch: str) -> str:
        try:
            return unicodedata.name(ch)
        except ValueError:
            return f"U+{ord(ch):04X}"
    non_ascii = {ch: n for ch, n in counts.items() if ord(ch) > 127}
    return {
        "n_unique_chars": len(counts),
        "charset": counts,
        "non_ascii": {f"{ch!r} {info(ch)}": n for ch, n in sorted(non_ascii.items(), key=lambda x: -x[1])},
        "rare_chars_le5": {f"{ch!r} {info(ch)}": n for ch, n in counts.items() if n <= 5},
        "lines_with_long_s": int(labels.str.contains("ſ").sum()),
        "lines_with_square_brackets": int(labels.str.contains(r"[\[\]]").sum()),
        "lines_with_question_mark_in_brackets": int(labels.str.contains(r"\[\s*\?\s*\]").sum()),
        "lines_with_angle_or_curly": int(labels.str.contains(r"[<>{}]").sum()),
        "lines_with_superscript_chars": int(labels.apply(lambda t: any(unicodedata.category(c) == "No" or "SUPERSCRIPT" in info(c) for c in t)).sum()),
        "lines_with_combining_marks": int(labels.apply(lambda t: any(unicodedata.combining(c) for c in t)).sum()),
        "lines_not_nfc": int((labels != labels.apply(lambda t: unicodedata.normalize("NFC", t))).sum()),
        "lines_leading_trailing_ws": int((labels != labels.str.strip()).sum()),
        "lines_double_space": int(labels.str.contains("  ").sum()),
        "lines_tab_or_newline": int(labels.str.contains(r"[\t\n\r]").sum()),
        "empty_labels": int((labels.str.strip() == "").sum()),
        "all_upper_lines": int(labels.apply(lambda t: t.isupper()).sum()),
    }


# ------------------------------------------------------- group detection
def derive_filename_groups(values: pd.Series) -> dict[str, pd.Series]:
    """Guess page/document keys from image filenames, e.g. 'vol3_p012_line05.jpg'."""
    stems = values.astype(str).map(lambda v: Path(v).stem)
    out = {}
    for sep in ("_", "-"):
        parts = stems.str.split(sep)
        max_parts = int(parts.map(len).max())
        for k in range(1, max_parts):
            out[f"filename[{sep}] first {k} parts"] = parts.map(lambda p, k=k: sep.join(p[:k]))
    return out


def group_candidates(train: pd.DataFrame, test: pd.DataFrame, cols: dict) -> list[dict]:
    n = len(train)
    cands: dict[str, tuple[pd.Series, pd.Series | None]] = {}
    for c in train.columns:
        if c in (cols["id"], cols["image"], cols["label"]):
            continue
        cands[f"column:{c}"] = (train[c].astype(str), test[c].astype(str) if c in test.columns else None)
    tr_fn = derive_filename_groups(train[cols["image"]])
    te_fn = derive_filename_groups(test[cols["image"]])
    for k, s in tr_fn.items():
        cands[k] = (s, te_fn.get(k))

    rows = []
    for name, (s, t) in cands.items():
        nu = s.nunique()
        if not (2 <= nu <= n // 2):
            continue  # constant or ~unique per line: useless as a group
        overlap = None
        if t is not None:
            overlap = round(float(t.isin(set(s)).mean()), 3)
        rows.append({"name": name, "n_groups": int(nu), "median_lines_per_group": float(s.value_counts().median()),
                     "test_rows_in_train_groups": overlap})
    rows.sort(key=lambda r: (r["name"].startswith("filename"), -r["n_groups"]))
    return rows


def get_group_series(train: pd.DataFrame, cols: dict, name: str) -> pd.Series:
    if name == "column:label":
        return train[cols["label"]].astype(str).str.strip()
    if name.startswith("column:"):
        return train[name.split(":", 1)[1]].astype(str)
    return derive_filename_groups(train[cols["image"]])[name]


# ---------------------------------------------------------------- split
def make_split(train: pd.DataFrame, cols: dict, groups: pd.Series | None, val_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if groups is None:
        is_val = rng.random(len(train)) < val_frac
        strategy = "random line split"
    else:
        uniq = np.array(sorted(groups.unique()), dtype=object)
        rng.shuffle(uniq)
        sizes = groups.value_counts()
        target, acc, val_groups = val_frac * len(train), 0, set()
        for g in uniq:
            if acc >= target:
                break
            val_groups.add(g); acc += sizes[g]
        is_val = groups.isin(val_groups).to_numpy()
        strategy = "group split"
    split = pd.DataFrame({cols["id"]: train[cols["id"]], "fold": np.where(is_val, "val", "train")})
    split.attrs["strategy"] = strategy
    return split


# --------------------------------------------------------------- report
def md_table(d: dict) -> str:
    return "| " + " | ".join(d) + " |\n| " + " | ".join("---" for _ in d) + " |\n| " + \
        " | ".join(str(v) for v in d.values()) + " |\n"


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else "missing"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--image-root", default=None, help="folder holding line images (default: --raw-dir)")
    ap.add_argument("--id-col"); ap.add_argument("--image-col"); ap.add_argument("--label-col")
    ap.add_argument("--group-col", help="column name, or exact candidate name from the report")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-version", default="v1")
    ap.add_argument("--tokenizer", default=None, help="e.g. microsoft/trocr-large-handwritten")
    ap.add_argument("--skip-images", action="store_true", help="skip existence + size checks")
    ap.add_argument("--report", default="reports/day1_data_audit.md")
    ap.add_argument("--split-dir", default="data/splits")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    image_root = Path(args.image_root) if args.image_root else raw
    train, test, sample = load_csvs(raw)
    cols = detect_columns(train, test, args)
    labels = train[cols["label"]].astype(str)
    print(f"[info] train={len(train)} test={len(test)} columns={cols}")

    R: list[str] = ["# Day 1 data audit", "",
                    "_Generated by `src/data_prep.py`. Aggregate statistics only; no transcriptions._", ""]

    # --- files & columns
    R += ["## Files and columns", "",
          f"- Train: {len(train)} rows, columns `{list(train.columns)}` (sha {file_sha(next(raw.rglob('Train.csv')))})",
          f"- Test: {len(test)} rows, columns `{list(test.columns)}`",
          f"- Detected: id=`{cols['id']}`, image=`{cols['image']}`, label=`{cols['label']}`",
          f"- Duplicate train ids: {int(train[cols['id']].duplicated().sum())}; "
          f"train/test id overlap: {int(train[cols['id']].isin(set(test[cols['id']])).sum())}", ""]

    # --- sample submission
    R += ["## SampleSubmission check", ""]
    if sample is None:
        R += ["- SampleSubmission.csv not found", ""]
    else:
        sid = sample.columns[0]
        same_ids = set(sample[sid].astype(str)) == set(test[cols["id"]].astype(str))
        R += [f"- Columns: `{list(sample.columns)}`, rows: {len(sample)} (test rows: {len(test)})",
              f"- First column `{sid}` matches Test ids exactly: **{same_ids}**", ""]

    # --- images
    if not args.skip_images:
        index = build_image_index(image_root)
        R += ["## Images", "", f"- Image files indexed under `{image_root}`: {len({p for p in index.values()})}"]
        for name, df in (("train", train), ("test", test)):
            paths = resolve_paths(df[cols["image"]], image_root, index)
            missing = int((paths == "").sum())
            st = image_stats(paths)
            if name == "train":
                train_img = st
            R += [f"- {name}: missing images **{missing}** / {len(df)}, unreadable {int((st['mode'] == 'unreadable').sum())}, "
                  f"modes {dict(Counter(st['mode']))}"]
            if st["width"].notna().any():
                R += ["", f"**{name} width (px)**", "", md_table(pct_table(st["width"])),
                      f"**{name} height (px)**", "", md_table(pct_table(st["height"])),
                      f"**{name} aspect ratio (w/h)**", "", md_table({k: round(v, 2) for k, v in pct_table(st["aspect"]).items()})]
        if "train_img" in locals() and train_img["aspect"].notna().any():
            wide = float((train_img["aspect"] > 8).mean())
            R += [f"- Share of train lines with aspect > 8 (badly squashed at 384x384): **{wide:.1%}**", ""]
            if train_img["width"].notna().any():
                corr = np.corrcoef(train_img["width"].fillna(0), labels.str.len())[0, 1]
                R += [f"- Correlation width vs label chars: {corr:.2f}", ""]

    # --- labels
    chars, words = labels.str.len(), labels.str.split().str.len()
    R += ["## Label lengths", "", "**Characters**", "", md_table(pct_table(chars)),
          "**Words**", "", md_table(pct_table(words))]
    if args.tokenizer:
        toks = token_lengths(labels, args.tokenizer)
        if toks is not None:
            p99 = float(np.percentile(toks, 99))
            R += [f"**Tokens ({args.tokenizer})**", "", md_table(pct_table(toks)),
                  f"- Suggested `max_new_tokens` = ceil(p99 x 1.2) = **{int(np.ceil(p99 * 1.2))}**; "
                  f"covers max? {int(np.ceil(p99 * 1.2)) >= toks.max()}",
                  f"- Tokenizer round-trip mismatches: **{toks.attrs['roundtrip_mismatch']}** (must be 0 or explained)", ""]

    # --- duplicate labels (formulaic text inflates random-split scores)
    vc = labels.value_counts()
    dup_share = float(labels.isin(vc[vc > 1].index).mean())
    R += ["## Duplicate labels", "",
          f"- Unique label strings: {labels.nunique()} / {len(labels)}",
          f"- Share of lines whose exact text appears more than once: **{dup_share:.1%}**",
          f"- Largest duplicate group size: {int(vc.max())}",
          "- If this is high, also score on a split that keeps identical labels together (`--group-col label`).", ""]

    # --- charset
    ca = charset_audit(labels)
    flags = {k: v for k, v in ca.items() if k.startswith(("lines_", "empty_", "all_upper", "n_unique"))}
    R += ["## Character set audit", "", "| check | count |", "| --- | --- |"]
    R += [f"| {k} | {v} |" for k, v in flags.items()]
    R += ["", "**Non-ASCII characters (by frequency)**", ""]
    R += [f"- {k}: {v}" for k, v in ca["non_ascii"].items()] or ["- none"]
    R += ["", "**Rare characters (<= 5 occurrences)**", ""]
    R += [f"- {k}: {v}" for k, v in ca["rare_chars_le5"].items()] or ["- none"]
    R += ["", "**Full character inventory**", "", "```",
          " ".join(f"{ch!r}:{n}" for ch, n in sorted(ca["charset"].items(), key=lambda x: -x[1])), "```", ""]

    # --- groups & split
    gc = group_candidates(train, test, cols)
    R += ["## Validation grouping candidates", "",
          "`test_rows_in_train_groups` near 1.0 means test lines come from the same pages as train lines. "
          "In that case a random line split mirrors the test set better; near 0.0 means use a group split.", ""]
    if gc:
        R += ["| candidate | n_groups | median lines/group | test rows in train groups |", "| --- | --- | --- | --- |"]
        R += [f"| {r['name']} | {r['n_groups']} | {r['median_lines_per_group']} | {r['test_rows_in_train_groups']} |" for r in gc[:15]]
    else:
        R += ["- No usable group candidates found."]

    chosen = None
    if args.group_col:
        chosen = args.group_col if args.group_col.startswith(("column:", "filename")) else f"column:{args.group_col}"
    elif gc and gc[0]["name"].startswith("column:"):
        top = gc[0]
        if top["test_rows_in_train_groups"] is None or top["test_rows_in_train_groups"] < 0.5:
            chosen = top["name"]  # only auto-pick real columns; filename guesses need a human check
    groups = get_group_series(train, cols, chosen) if chosen else None
    split = make_split(train, cols, groups, args.val_frac, args.seed)

    split_dir = Path(args.split_dir); split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"split_{args.split_version}.csv"
    split.to_csv(split_path, index=False)
    meta = {"version": args.split_version, "seed": args.seed, "val_frac": args.val_frac,
            "strategy": split.attrs["strategy"], "group": chosen, "columns": cols,
            "n_train": int((split.fold == "train").sum()), "n_val": int((split.fold == "val").sum())}
    (split_dir / f"split_{args.split_version}.json").write_text(json.dumps(meta, indent=2))

    val_chars = chars[split.fold.values == "val"]
    R += ["", "## Split written", "",
          f"- File: `{split_path}` (git-ignored) + `.json` metadata",
          f"- Strategy: **{meta['strategy']}**, group: `{chosen}`, seed {args.seed}, val_frac {args.val_frac}",
          f"- Train {meta['n_train']} / Val {meta['n_val']}",
          f"- Val label chars mean {val_chars.mean():.1f} vs all {chars.mean():.1f} (should be close)", ""]
    if chosen is None and gc:
        R += ["> Group was not auto-selected. Review the candidates above and rerun with "
              "`--group-col \"<candidate name>\"` if one represents page/document.", ""]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(R))
    print(f"[done] report -> {args.report}; split -> {split_path} ({meta['strategy']})")


if __name__ == "__main__":
    main()
