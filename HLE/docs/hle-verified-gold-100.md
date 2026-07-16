# HLE-Verified Gold 100 mini-benchmark

This manifest defines `hle-verified-gold-100-v1`, a fixed 100-question text-only subset of HLE-Verified Gold for repeatable model comparisons.
It is separate from the existing raw-HLE `hle-mini-100-v1` benchmark and does not replace its files or results.

## Selection contract

- Verification source: `skylenage-ai/HLE-Verified` at commit `0bc83643672d4f68a5f89998617a639d85e7318b`.
- Compatibility source: `cais/hle`, `test` split, local revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`.
- Conservative Gold definition: records classified `Gold subset` with `is_valid=1` for the problem, answer, and rationale components and no revision to the original HLE item.
- Source consistency safeguard: the five Gold shards contain 668 rows, but 27 rows have at least one component marked invalid. Those 27 are excluded, leaving 641 strict Gold records—the count reported in the paper abstract.
- Candidate pool: 554 text-only records from the 641 strict Gold records; the remaining 87 strict Gold records have a non-empty input `image` and are excluded.
- Balance: 100 questions across all eight broad HLE categories using the same quotas as `hle-mini-100-v1`.
- Deterministic selection: within each category, rank candidates by `SHA-256("hle-verified-gold-mini-bench-v1-2026-07-02:" + question_id)` and take the category quota.
- Final run order: ascending zero-based `sorted_index` from the pinned `cais/hle` text-only pool.
- Compatibility check: every selected question, answer, image field, answer type, category, and raw subject exactly matches the pinned `cais/hle` record with the same ID.
- Dataset changes: use `id` as the durable identifier; `sorted_index` is specific to the pinned raw HLE revision.
- Ordered-ID fingerprint: `sha256:a3d14b0508b1d6b1f324064104479b2978ddef0b89913d63dc03d8473fd29a85` (newline-delimited IDs with a trailing newline).
- Content fingerprint: `sha256:acc23697c41c36932a3074f98f99b70314f2a4a8ac95a33b56fa40412930da0b` (canonical JSON over ordered prompt/answer metadata).

## Category quotas

| Category | Gold text-only available | Selected |
|---|---:|---:|
| Biology/Medicine | 44 | 13 |
| Chemistry | 12 | 12 |
| Computer Science/AI | 73 | 13 |
| Engineering | 15 | 12 |
| Humanities/Social Science | 53 | 12 |
| Math | 262 | 13 |
| Other | 49 | 12 |
| Physics | 46 | 13 |
| **Total** | **554** | **100** |

## Running the subset

The CSV is the canonical content manifest. The command below extracts its IDs and runs the matching records from the pinned local `cais/hle` cache. Batch correctness judging runs after the response batch by default.

```bash
python3 -c 'import csv,sys; print(*(r["id"] for r in csv.DictReader(open(sys.argv[1]))))' \
  hle-verified-gold-100-questions.csv | \
  xargs .venv/bin/python bench.py \
    --endpoint-name hle-verified-gold-100-v1 \
    --runs 1 \
    --no-print-question \
    --max-tokens 100000 \
    --omit-temperature \
    --question-ids
```

To run only Gold100 questions 1–10, use the same command with this extraction line:

```bash
python3 -c 'import csv,itertools,sys; print(*(r["id"] for r in itertools.islice(csv.DictReader(open(sys.argv[1])), 10)))' \
  hle-verified-gold-100-questions.csv | \
  xargs .venv/bin/python bench.py \
    --endpoint-name hle-verified-gold-100-v1-q001-010 \
    --runs 1 \
    --no-print-question \
    --max-tokens 100000 \
    --omit-temperature \
    --question-ids
```

## Sources

- HLE-Verified dataset: https://huggingface.co/datasets/skylenage-ai/HLE-Verified
- HLE-Verified paper: https://arxiv.org/abs/2602.13964
- Original HLE dataset: https://huggingface.co/datasets/cais/hle

## Question manifest

| Gold100 # | sorted_index | question_id | Category | Raw subject | Answer type |
|---:|---:|---|---|---|---|
| 1 | 6 | 66b827b9b64deaedfbb997a2 | Physics | Physics | exactMatch |
| 2 | 13 | 66e883265ab37f0a7da089be | Other | Chess | multipleChoice |
| 3 | 27 | 66e8ccc4089d1e34c84c76c0 | Biology/Medicine | Biology | exactMatch |
| 4 | 55 | 66e9469b76b5b4f3e8a369d5 | Math | Mathematics | exactMatch |
| 5 | 69 | 66e9b2899cf8fcf41599246f | Humanities/Social Science | Linguistics | multipleChoice |
| 6 | 70 | 66e9b92578e89514d9ab6093 | Other | Trivia | exactMatch |
| 7 | 102 | 66eaa5ddc7a3252f0f3fe53f | Engineering | Materials Science | exactMatch |
| 8 | 134 | 66ebefa090db075818df99a9 | Humanities/Social Science | Psychology | multipleChoice |
| 9 | 142 | 66ec8ab4b489956467209e0c | Engineering | Electrical Engineering | exactMatch |
| 10 | 151 | 66ed3c5dae789e6253eedddd | Other | Games | exactMatch |
| 11 | 154 | 66ed5611d94c2fb87ded8826 | Other | Games | exactMatch |
| 12 | 156 | 66ed5f1e85adbeda9f978022 | Other | Geography | multipleChoice |
| 13 | 168 | 66edc60801af2a724035ad4b | Biology/Medicine | Biology | multipleChoice |
| 14 | 211 | 66efd054c04acd134cc4bb36 | Math | Mathematics | exactMatch |
| 15 | 214 | 66f0247d8777dcd82dba8253 | Computer Science/AI | Computer Science | exactMatch |
| 16 | 244 | 66f275c6e2a2b5d3594eae87 | Computer Science/AI | Computer Science | exactMatch |
| 17 | 245 | 66f27d65a40482f6012a4006 | Humanities/Social Science | Linguistics | multipleChoice |
| 18 | 282 | 66f57a7e9f9308128679f668 | Biology/Medicine | Neuroscience | multipleChoice |
| 19 | 298 | 66f6b73a1b586571e550784f | Math | Mathematics | exactMatch |
| 20 | 344 | 66fb8135483861eb2d0252a3 | Math | Mathematics | exactMatch |
| 21 | 350 | 66fbe64df560b62458a7b6a1 | Computer Science/AI | Computer Science | exactMatch |
| 22 | 367 | 66fc564eae059175d7cc3244 | Chemistry | Chemistry | exactMatch |
| 23 | 371 | 66fc5e611f5f3f3b48ae9566 | Biology/Medicine | Neuroscience | multipleChoice |
| 24 | 373 | 66fc5ed440e3b3e56869687f | Physics | Nuclear Physics | exactMatch |
| 25 | 382 | 66fc8b271d39fbf6d8bcdd0c | Biology/Medicine | Biology | exactMatch |
| 26 | 435 | 66fec7825e6051260840e060 | Chemistry | Chemistry | exactMatch |
| 27 | 520 | 670489fcedc6951c9585de8f | Engineering | Industrial Engineering | exactMatch |
| 28 | 545 | 6707425209b8f334446ed3e2 | Physics | Physics | exactMatch |
| 29 | 548 | 6707b8b6700263d6945e7b18 | Math | Applied Mathematics | exactMatch |
| 30 | 551 | 67082afe09cbab4e7c6f62aa | Computer Science/AI | Computer Science | multipleChoice |
| 31 | 555 | 670872c2f1b4c3641356feb0 | Chemistry | Chemistry | exactMatch |
| 32 | 557 | 670880520ed68fbdc467064e | Chemistry | Chemistry | exactMatch |
| 33 | 583 | 670b307567eb710437409184 | Other | Game Design | exactMatch |
| 34 | 586 | 670be48d7038d6936230870a | Physics | Classical Physics | exactMatch |
| 35 | 596 | 670c83ba4aece479236947cb | Biology/Medicine | Physical Medicine And Rehabilitation | multipleChoice |
| 36 | 597 | 670c8b10148f2a113537c8f6 | Math | Mathematics | multipleChoice |
| 37 | 598 | 670ca1456731aa001b9ba021 | Humanities/Social Science | Finance | exactMatch |
| 38 | 622 | 670e88d674a7c40e93dd1a5c | Physics | Physics | exactMatch |
| 39 | 628 | 670eb27fd2f45b1198c87766 | Engineering | Mechanical Engineering | exactMatch |
| 40 | 629 | 670edc9dbddc0cfe673272c8 | Chemistry | Chemistry | multipleChoice |
| 41 | 638 | 670f39dc1dcaeb830ff6231f | Humanities/Social Science | Linguistics | multipleChoice |
| 42 | 656 | 670fe86a7e294dc6ad20c1ba | Physics | Physics | exactMatch |
| 43 | 772 | 6717dd24e8666ff79cdd82b0 | Computer Science/AI | Computer Science | multipleChoice |
| 44 | 779 | 671805c78b88f01935b58255 | Computer Science/AI | Computer Science | multipleChoice |
| 45 | 788 | 6718577ca88093a75026b186 | Humanities/Social Science | Finance | multipleChoice |
| 46 | 802 | 671963d90f87e9920aff9d11 | Computer Science/AI | Computer Science | multipleChoice |
| 47 | 816 | 671a567961c380782c9eea17 | Engineering | Mechanical Engineering | exactMatch |
| 48 | 837 | 671ada4eed3d54e87368bc78 | Math | Mathematics | exactMatch |
| 49 | 850 | 671ba7b447e34cf4ed747b30 | Engineering | Mechanical Engineering | exactMatch |
| 50 | 855 | 671bc0c855449c636f4bbd36 | Chemistry | Chemistry | multipleChoice |
| 51 | 892 | 671d6502b996cf9936d1afd0 | Engineering | Electrical Engineering | exactMatch |
| 52 | 896 | 671d97e729e7fde7166e4743 | Engineering | Remote Sensing | multipleChoice |
| 53 | 897 | 671d999f18a4da3122fd2118 | Chemistry | Chemistry | exactMatch |
| 54 | 901 | 671db218fe1146e348ef1266 | Chemistry | Chemistry | exactMatch |
| 55 | 930 | 671ef4bd6edc2afd6995897b | Engineering | Electrical Engineering | multipleChoice |
| 56 | 948 | 671f2b0ee38f776acdad8aa1 | Biology/Medicine | Ecology | multipleChoice |
| 57 | 956 | 671f3d49d579cf064f22d3ce | Other | Graphic Design | exactMatch |
| 58 | 963 | 671f612d12bc18b3bf57dd89 | Math | Mathematics | exactMatch |
| 59 | 970 | 671f887676b11ce91b2887ce | Math | Mathematics | exactMatch |
| 60 | 973 | 671f8aa5e8fbfa3cf02ce3b6 | Computer Science/AI | Computer Science | exactMatch |
| 61 | 980 | 671fb0b7298c0d11670fc561 | Engineering | Electrical Engineering | exactMatch |
| 62 | 983 | 671fb84fc6abf8266c1892c8 | Math | Mathematics | exactMatch |
| 63 | 1027 | 67209100563d776c82113dba | Other | Classical Ballet | multipleChoice |
| 64 | 1095 | 6721c8e11b5a8e4cb0e9079b | Humanities/Social Science | Law | multipleChoice |
| 65 | 1161 | 672309a572e4abc960be3774 | Computer Science/AI | Computer Science | multipleChoice |
| 66 | 1170 | 67235bc3c0ae8158005244a9 | Math | Mathematics | exactMatch |
| 67 | 1181 | 6723a613f747d32c6b0b65dc | Chemistry | Chemistry | multipleChoice |
| 68 | 1199 | 6723ebefcf4ea65226eb6f9c | Other | Geology | exactMatch |
| 69 | 1200 | 6723ec50479384d8942cca75 | Humanities/Social Science | Law | multipleChoice |
| 70 | 1235 | 67242f1f911674ab1b5d904b | Biology/Medicine | Medicine | multipleChoice |
| 71 | 1237 | 67243887a7c5f8f463109d82 | Other | Go | exactMatch |
| 72 | 1253 | 67249d57d91756473725533a | Computer Science/AI | Artificial Intelligence | multipleChoice |
| 73 | 1288 | 6724ff0dea5926938a631b9e | Other | Musicology | multipleChoice |
| 74 | 1294 | 67250fde58b17ce5905f2cfe | Chemistry | Chemistry | exactMatch |
| 75 | 1296 | 6725145d97743d26179494d6 | Humanities/Social Science | History | exactMatch |
| 76 | 1303 | 6725267ae9d3782179d4a5ff | Physics | Physics | exactMatch |
| 77 | 1317 | 672538bc6d762e2b5184b6cf | Computer Science/AI | Cybersecurity | multipleChoice |
| 78 | 1345 | 67257157c16289d7e113915b | Chemistry | Chemistry | multipleChoice |
| 79 | 1422 | 6726941826b7fc6a39fbe581 | Humanities/Social Science | Linguistics | multipleChoice |
| 80 | 1606 | 67313652f659ba7b3fd1fe40 | Humanities/Social Science | Russian Literature | multipleChoice |
| 81 | 1678 | 673531cae26d0576765bd963 | Math | Mathematics | exactMatch |
| 82 | 1701 | 67364c441758ad568a4a2ac6 | Physics | Physics | exactMatch |
| 83 | 1715 | 6736945a5a7f4f59e4731c4d | Engineering | Civil Engineering | exactMatch |
| 84 | 1733 | 6736d8d1278519f8b18450a5 | Chemistry | Chemistry | exactMatch |
| 85 | 1736 | 6736ddc7ab70ca0b4ee6a2e6 | Computer Science/AI | Computer Science | exactMatch |
| 86 | 1750 | 6736f655e51e5ebba186a75d | Computer Science/AI | Computer Science | exactMatch |
| 87 | 1786 | 67372744600c9c0daa5d8f3f | Biology/Medicine | Medicine | multipleChoice |
| 88 | 1800 | 67374c79ccee19cce9664dd5 | Physics | Physics | exactMatch |
| 89 | 1830 | 673818e39b3842b34824240d | Physics | High Energy Physics And Nuclear Physics | multipleChoice |
| 90 | 1842 | 67383288f2df805520bc86b5 | Biology/Medicine | Genetics | multipleChoice |
| 91 | 1886 | 67396c5bbb01e9373d18968e | Biology/Medicine | Biology | exactMatch |
| 92 | 2013 | 67455cd07df215d3effe4f4e | Physics | Physics | exactMatch |
| 93 | 2014 | 67455f379dbdcf3802abd8f6 | Physics | Physics | exactMatch |
| 94 | 2049 | 6755d8a01c505b5224374708 | Math | Statistics | exactMatch |
| 95 | 2058 | 675aa6e703e9471764dfedd2 | Other | Trivia | multipleChoice |
| 96 | 2060 | 675b351deb7996cd4dfe804c | Humanities/Social Science | Poetry | exactMatch |
| 97 | 2084 | 67672352c393c4ff629cb820 | Biology/Medicine | Neuroscience | multipleChoice |
| 98 | 2101 | 677296942ebbac6133a1d618 | Physics | Quantum Physics | exactMatch |
| 99 | 2125 | 6781903d382cfca83d01b77f | Biology/Medicine | Bioengineering | exactMatch |
| 100 | 2146 | 67aacfd513ec9e1a16359d51 | Engineering | Computer Engineering | exactMatch |
