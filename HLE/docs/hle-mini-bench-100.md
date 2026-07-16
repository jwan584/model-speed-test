# HLE mini-benchmark: balanced 100-question subset

This manifest defines a fixed, text-only HLE mini-benchmark for repeatable model comparisons.
It records both the durable HLE `question_id` and the current zero-based `sorted_index` used by `bench.py`.

## Selection contract

- Source: `cais/hle`, `test` split, local dataset revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`.
- Candidate pool: 2,158 text-only questions, sorted lexicographically by HLE `id`, exactly matching `bench.py`.
- Balance: 100 questions across all eight broad categories. Since 100 is not divisible by 8, four categories have 13 questions and four have 12.
- The four largest text-only categories receive the extra question: Biology/Medicine, Computer Science/AI, Math, and Physics.
- Deterministic selection: within each category, rank candidates by `SHA-256("hle-mini-bench-v1-2026-07-02:" + question_id)` and take the category quota.
- Final run order: ascending `sorted_index`.
- Dataset changes: prefer `question_id`; `sorted_index` is specific to the dataset revision above.
- Ordered-ID fingerprint: `sha256:599a49520013d7b4893dfaf19f0727e414ef991b628b8723803fda50391f5ad6` (newline-delimited IDs with a trailing newline).

## Category quotas

| Category | Text-only available | Selected |
|---|---:|---:|
| Biology/Medicine | 222 | 13 |
| Chemistry | 101 | 12 |
| Computer Science/AI | 224 | 13 |
| Engineering | 64 | 12 |
| Humanities/Social Science | 193 | 12 |
| Math | 976 | 13 |
| Other | 176 | 12 |
| Physics | 202 | 13 |
| **Total** | **2,158** | **100** |

## Running the subset

The following extracts the IDs from this manifest and passes all 100 to `bench.py`. Set the usual model/API environment variables first.

```bash
awk -F'|' '/^\| [0-9]+ / { gsub(/^ +| +$/, "", $4); print $4 }' docs/hle-mini-bench-100.md | \
  xargs .venv/bin/python bench.py \
    --endpoint-name hle-mini-100-v1 \
    --runs 1 \
    --no-print-question \
    --max-tokens 100000 \
    --omit-temperature \
    --question-ids
```

Batch judging is enabled by default and runs after all selected model responses finish. Use the normal judge arguments to override its model or endpoint, or pass `--no-judge-after-run` to skip correctness scoring.

## Question manifest

| Mini # | sorted_index | question_id | Category | Raw subject | Answer type |
|---:|---:|---|---|---|---|
| 1 | 12 | 66e8784d70625d8c7700315a | Computer Science/AI | Artificial Intelligence | multipleChoice |
| 2 | 19 | 66e8a1833aa94517d4573b0d | Humanities/Social Science | Political Science | exactMatch |
| 3 | 62 | 66e95faf8451a9b41f307932 | Humanities/Social Science | English Literature | exactMatch |
| 4 | 83 | 66ea2cc3c602e2b991ae8aba | Computer Science/AI | Computer Science | exactMatch |
| 5 | 88 | 66ea3d3fa715c6c835b25764 | Other | Musicology | exactMatch |
| 6 | 135 | 66ec02c52ec65d6153428744 | Engineering | Chemical Engineering | multipleChoice |
| 7 | 156 | 66ed5f1e85adbeda9f978022 | Other | Geography | multipleChoice |
| 8 | 249 | 66f2cda3b508188b6e7328a8 | Engineering | Process Optimization | exactMatch |
| 9 | 266 | 66f402add1c77d20ca3338ef | Chemistry | Chemoinformatics | exactMatch |
| 10 | 355 | 66fc23cfa7be4edbe85cf177 | Humanities/Social Science | Economics | exactMatch |
| 11 | 362 | 66fc45034293a9638d7e0f47 | Physics | Physics | exactMatch |
| 12 | 392 | 66fcde117e54294eb4a8fbba | Biology/Medicine | Biology | multipleChoice |
| 13 | 397 | 66fcf81e8a146dd80cfb2296 | Biology/Medicine | Medicine | multipleChoice |
| 14 | 414 | 66fdaf20d1e0bb15b8fc1eb6 | Biology/Medicine | Biology | multipleChoice |
| 15 | 451 | 66ffb3e3ab9ced47e903bbec | Chemistry | Chemistry | exactMatch |
| 16 | 462 | 67008a05ad0fee7d7b4efb3c | Chemistry | Chemistry | exactMatch |
| 17 | 467 | 67009ad56c339d61ecccb85c | Computer Science/AI | Computer Science | exactMatch |
| 18 | 481 | 67015d62777a275ca50eb18b | Humanities/Social Science | Classics | multipleChoice |
| 19 | 492 | 6701d869aee3881b852d40a0 | Biology/Medicine | Neuroscience | exactMatch |
| 20 | 496 | 670285bc39fbddbbfdaffdfe | Humanities/Social Science | Linguistics | multipleChoice |
| 21 | 515 | 6704465caf0a436d92c65160 | Physics | Physics | exactMatch |
| 22 | 572 | 6709df4c3c2174379ffee04b | Other | Art History | exactMatch |
| 23 | 596 | 670c83ba4aece479236947cb | Biology/Medicine | Physical Medicine And Rehabilitation | multipleChoice |
| 24 | 605 | 670d86ec56f489221087dc67 | Other | Movies | exactMatch |
| 25 | 677 | 67126745df2820fcc29acc5f | Computer Science/AI | Computer Science | exactMatch |
| 26 | 690 | 6713afa11e0e03ffe2253dd4 | Chemistry | Combined Chemistry And Trivia | exactMatch |
| 27 | 702 | 67152ee0953411f24cd994f0 | Computer Science/AI | Computer Science | exactMatch |
| 28 | 724 | 67164d0b4c922006e9e93a8d | Humanities/Social Science | Epistemology | multipleChoice |
| 29 | 801 | 67192d7e0fa7bca6462f63a9 | Computer Science/AI | Artificial Intelligence | exactMatch |
| 30 | 802 | 671963d90f87e9920aff9d11 | Computer Science/AI | Computer Science | multipleChoice |
| 31 | 816 | 671a567961c380782c9eea17 | Engineering | Mechanical Engineering | exactMatch |
| 32 | 896 | 671d97e729e7fde7166e4743 | Engineering | Remote Sensing | multipleChoice |
| 33 | 902 | 671db266fe1146e348ef1267 | Other | Chess | exactMatch |
| 34 | 903 | 671dba3e5102c27a58a6c501 | Chemistry | Chemistry | multipleChoice |
| 35 | 916 | 671e7fd05cd705ffbd3faab7 | Chemistry | Chemistry | exactMatch |
| 36 | 921 | 671eb856c357c6b4f73592dd | Engineering | Electrical Engineering | exactMatch |
| 37 | 923 | 671ebbf35cc535d3e94216ac | Engineering | Electrical Engineering | exactMatch |
| 38 | 927 | 671ee933019b32e00d827382 | Engineering | Electrical Engineering | multipleChoice |
| 39 | 928 | 671eeb53c1a668a6c81e5993 | Engineering | Electrical Engineering | multipleChoice |
| 40 | 934 | 671f083dc8da11076ce9960e | Biology/Medicine | Ecology | multipleChoice |
| 41 | 936 | 671f0b0c7301fac39660e7a3 | Engineering | Electrical Engineering | multipleChoice |
| 42 | 945 | 671f1f88e6600c2d52d9fbe6 | Humanities/Social Science | Linguistics | exactMatch |
| 43 | 950 | 671f33cb75523fe63c0a8b60 | Chemistry | Chemistry | multipleChoice |
| 44 | 975 | 671f99152e60076c5693554f | Math | Mathematics | exactMatch |
| 45 | 993 | 671fd9236c5d3903234cd3aa | Other | Musicology | exactMatch |
| 46 | 1019 | 672065f65681ce2b6f5a08a0 | Computer Science/AI | Computer Science | multipleChoice |
| 47 | 1042 | 6720cd0acf47ec0733864dd8 | Chemistry | Chemistry | multipleChoice |
| 48 | 1047 | 6720e184a9e1d1cc990cc8e9 | Physics | Physics | exactMatch |
| 49 | 1081 | 67217f97262eafa82562cc2b | Other | Earth Science | multipleChoice |
| 50 | 1162 | 67230d6e736f03c0e4c1adee | Physics | Physics | exactMatch |
| 51 | 1181 | 6723a613f747d32c6b0b65dc | Chemistry | Chemistry | multipleChoice |
| 52 | 1192 | 6723cfdeddb9a8e96a06901a | Math | Mathematics | exactMatch |
| 53 | 1200 | 6723ec50479384d8942cca75 | Humanities/Social Science | Law | multipleChoice |
| 54 | 1215 | 672403fa5461772b24b2e651 | Biology/Medicine | Genetics | multipleChoice |
| 55 | 1236 | 672433577fb5d24be68f010d | Other | Go | exactMatch |
| 56 | 1247 | 67248cadd04b3798125682f3 | Math | Mathematics | exactMatch |
| 57 | 1254 | 67249fe6d917564737255342 | Biology/Medicine | Medicine | exactMatch |
| 58 | 1309 | 67252aad70e5e32e5897fa56 | Math | Mathematics | exactMatch |
| 59 | 1321 | 67253beb5a5d8bd389020394 | Chemistry | Chemistry | exactMatch |
| 60 | 1341 | 67256b14ac4f9591b137e180 | Biology/Medicine | Genetics | multipleChoice |
| 61 | 1349 | 67257466e173b172c061372a | Math | Applied Mathematics | exactMatch |
| 62 | 1386 | 6725e42052e181595c8bf328 | Physics | Physics | multipleChoice |
| 63 | 1459 | 67298280a5f43bd5a3870e14 | Physics | Physics | exactMatch |
| 64 | 1464 | 672a27f5d30d6f5584cde73d | Physics | Physics | exactMatch |
| 65 | 1475 | 672a5ecf541155da3e036094 | Physics | Physics | multipleChoice |
| 66 | 1499 | 672c033ff576aed47449d75f | Chemistry | Computational Chemistry | multipleChoice |
| 67 | 1536 | 672da2566d1f60da4a748aca | Math | Mathematics | exactMatch |
| 68 | 1553 | 672e09b50a85795d0ed2d36e | Physics | Physics | multipleChoice |
| 69 | 1598 | 6730a9be58ef965949f1faa4 | Math | Mathematical Physics: Functional Equations | exactMatch |
| 70 | 1613 | 6731c87c6c74786218717a81 | Math | Mathematics | exactMatch |
| 71 | 1636 | 67330386bda81b2106f720dc | Computer Science/AI | Cryptography | exactMatch |
| 72 | 1637 | 67330f175d26efacb4819f35 | Other | Art History | exactMatch |
| 73 | 1660 | 6734956467d2904eebed3a09 | Physics | Physics | multipleChoice |
| 74 | 1678 | 673531cae26d0576765bd963 | Math | Mathematics | exactMatch |
| 75 | 1686 | 6735bafad86155d1e57160e7 | Physics | Physics | exactMatch |
| 76 | 1713 | 673681def5487e4de6e78e1e | Chemistry | Chemistry | exactMatch |
| 77 | 1742 | 6736e9109055c436feb3f65a | Other | Game Of Go | multipleChoice |
| 78 | 1751 | 6736f7bc980211368f0f94eb | Humanities/Social Science | Law | exactMatch |
| 79 | 1781 | 673722c82bfc8ab579ed111f | Humanities/Social Science | Economics | multipleChoice |
| 80 | 1789 | 6737309d1988146a57ffab18 | Math | Mathematics | exactMatch |
| 81 | 1812 | 67379aea6c946be458900f3f | Physics | Physics | exactMatch |
| 82 | 1832 | 67381b2862660a32c77bfe3d | Computer Science/AI | Computer Science | exactMatch |
| 83 | 1833 | 67381ce26a5242a22fe4681f | Math | Mathematics | exactMatch |
| 84 | 1886 | 67396c5bbb01e9373d18968e | Biology/Medicine | Biology | exactMatch |
| 85 | 1892 | 67399d0163641fb0bc110db9 | Other | Musicology | exactMatch |
| 86 | 1897 | 6739e82ba0d19bb8d127ad6c | Engineering | Materials Science | exactMatch |
| 87 | 1899 | 6739f5ede8b91dc9dac17cd9 | Biology/Medicine | Biology | exactMatch |
| 88 | 1900 | 6739fdd1e27e1ce214359d62 | Engineering | Computer Engineering | exactMatch |
| 89 | 1909 | 673a36936a38861184e67871 | Math | Mathematics | exactMatch |
| 90 | 1918 | 673a6a6c4c465c371379b670 | Biology/Medicine | Biochemistry | multipleChoice |
| 91 | 1955 | 673b955f2ddd80745b6bd232 | Computer Science/AI | Computer Science | exactMatch |
| 92 | 1957 | 673bd7048229809fa3ec5653 | Engineering | Electrical Engineering | exactMatch |
| 93 | 1964 | 673cc4885c871b3f9e026d02 | Physics | Astronomy | exactMatch |
| 94 | 1974 | 673e37db8d2811de2a83c135 | Humanities/Social Science | History | exactMatch |
| 95 | 1977 | 673e6cb6dd8bb70cf2be5e47 | Computer Science/AI | Computer Science | exactMatch |
| 96 | 1979 | 673eb1cfadce15d9254eb2ac | Humanities/Social Science | Philosophy | exactMatch |
| 97 | 2056 | 67581f18abd39842c40bd2fd | Computer Science/AI | Artificial Intelligence | exactMatch |
| 98 | 2067 | 675d7b901ded33d59eb2c94f | Math | Mathematics | exactMatch |
| 99 | 2086 | 6767969869b88c321c96665a | Biology/Medicine | Neuroscience | multipleChoice |
| 100 | 2097 | 6770832c6b74a5103a07f031 | Other | Logic | exactMatch |
