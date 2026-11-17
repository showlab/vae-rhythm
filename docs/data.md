# Data

Datasets are not redistributed with this repository. All of them are public on Hugging Face; follow each
dataset's licence and terms.

| Dataset | Role in the paper | Source |
|---|---|---|
| AIGVDBench | training (Open-Sora 14K + real 14K), validation, test (20 open-source + 11 closed-source generators) | `AIGVDBench/AIGVDBench` |
| DF26 | test (10 generators from 2025-26, paired real clips, 2AFC) | `DF26/DF26` (gated) |
| GenBuster-Bench++ | supplementary test (13 commercial generators, uniformly re-encoded) | `l8cv/GenBuster-Bench-plusplus` |
| ViF-Bench | supplementary test (re-encoded to 256p / 16 fps) | `JoeLeelyf/ViF-Bench` |
| MiniMax H3, Seedance 2.5 | 2026 closed-source test (generated only) | `wcxhimself/KIVI-Generated-Videos` |
| Seedance 2.0 | 2026 closed-source test (generated only) | `HelioAI/Open-Seedance-2.0-Videos` |

The 2026 sets contain only generated clips; the paper pairs them with the DF26 real clips (and, as a second
reading, with the AIGVDBench test reals).

```bash
hf download AIGVDBench/AIGVDBench --repo-type dataset --local-dir data/AIGVDBench
python scripts/prepare_aigvdbench.py --root data/AIGVDBench --out data/aigvd          # -> data/aigvd/manifest_aigvd.csv
hf download DF26/DF26 --repo-type dataset --local-dir data/DF26
python scripts/prepare_df26.py --root data/DF26 --out data/df26/manifest_df26.csv
```

## Manifest format

A manifest is a CSV with one row per video. Required columns depend on the command:

| column | meaning | used by |
|---|---|---|
| `video_id` | unique id (may contain `/`) | all |
| `path` | video file | scans, pairs, frame cache |
| `label` | 0 real, 1 generated | training, evaluation |
| `subset` | generator name (`Real` for real clips) | evaluation (one AUROC per generator) |
| `group` | `real`, `open_t2v`, `open_i2v`, `open_v2v` or `closed` | open / closed means |
| `split` | `train`, `val` or `test`; select with `file.csv:split` | training, calibration |
| `dataset` | dataset name; `synth` marks VAE-pair rows; frame-cache sub-folder | CaDet |
| `pair_id` | id shared by a generated clip and its real source | paired 2AFC |

Evaluation scores every generated subset against *all* real rows of the same manifest.

## Released VAE pairs and scores

A Hugging Face dataset (coming soon) will hold the 5,333 VAE pairs used in the paper
(4,818 for training, 515 for validation; derived from AIGVDBench training reals), their manifest with every
chain parameter, and per-video scores (phase-scan statistics and detector scores) behind the paper's tables.
