# Data audit

- Train rows: 4098, Test rows: 1374, ID overlap: 0
- Missing train images: 0
- Missing test images: 0
- Empty train labels: 0, duplicate labels: 12
- Labels with leading/trailing whitespace: 98, double spaces: 55

## Label length
- Characters: p0=37, p1=40, p50=62, p95=85, p99=93, p100=120
- Words: p0=4, p1=6, p50=11, p95=16, p99=18, p100=22

## Character set
81 distinct characters. Rare (<=5 occurrences): `|`=1 `\`=1 `#`=2 `?`=2 `*`=3 `Q`=4

Non-alphanumeric characters (convention markers such as `^`, `~`, `\`, `|`):
`^`=872 `&`=757 `,`=628 `:`=470 `.`=211 `-`=184 `~`=134 `_`=48 `(`=40 `'`=38 `)`=31 `;`=25 `"`=18 `+`=6 `*`=3 `#`=2 `?`=2 `|`=1 `\`=1

## Image sizes (width x height)
- train width: p0=267, p1=637, p50=1118, p95=4941, p99=5599, p100=6030
- train height: p0=31, p1=39, p50=65, p95=403, p99=471, p100=1131
- train aspect (w/h): p0=3, p1=8, p50=17, p95=25, p99=28, p100=34
- test width: p0=504, p1=591, p50=1120, p95=4850, p99=5547, p100=6051
- test height: p0=28, p1=40, p50=65, p95=409, p99=471, p100=617
- test aspect (w/h): p0=4, p1=8, p50=17, p95=25, p99=28, p100=41
