# Data Setup

Place processed inputs under `data/` (not tracked in git).

## Required for full pipeline

| Path | Source | Notes |
|------|--------|-------|
| `data/processed/GSE186505/human_blood_expr_processed.csv` | [GSE186505](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186505) | 10 TN + 10 control blood RNA-seq |
| `data/processed/ion_cci/` | IoN-CCI supplementary | Mouse TG bulk for module gene sets |
| `data/external/openneuro/ds005713/` | [OpenNeuro ds005713](https://openneuro.org/datasets/ds005713) | Imaging + MVD outcomes |
| `data/external/geo/GSE177034/` | GEO | NP meta cohort 1 |
| `data/external/geo/GSE124272/` | GEO | NP meta cohort 2 |
| `data/external/geo/GSE150408/` | GEO | NP meta cohort 3 |

## Optional

| Path | Source |
|------|--------|
| `data/processed/GSE197289/` | Human TG snRNA-seq (mechanistic hub) |
| `data/processed/GSE240432/` | Mouse TG scRNA-seq |
| `data/manual_cnv_pilot/masks/` | Expert CN V masks (supplementary pilot) |

## OpenNeuro git-annex

If DTI files are pointer stubs, see [OPENNEURO_ANNEX_RECOVERY.md](OPENNEURO_ANNEX_RECOVERY.md).

## Config

Copy `config/config_template.yaml` → `config/config.yaml` and set `data_dir` paths.
