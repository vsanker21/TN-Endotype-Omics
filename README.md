# TN Endotype Omics — Analysis Pipeline

Blood-derived **TN molecular endotypes** (k=2, GSE186505 discovery) integrated with **IoN-CCI cross-species modules**, **DTI-first OpenNeuro imaging**, and **external neuropathic pain validation**.

This repository contains **analysis code only** — not manuscript or publication figure generation scripts.

## Scientific focus (locked claims)

See [docs/CLAIM_TABLE_LOCKED.md](docs/CLAIM_TABLE_LOCKED.md).

| Supported in discovery | Requires external cohort |
|------------------------|-------------------------|
| k=2 TN blood endotypes (n=10) | Independent TN subtype replication (Stanford) |
| IoN-CCI module scores in blood | Matched blood–imaging–MVD (n=0 in public data) |
| NP cohort module **scoreability** | SEVB-Net CN V segmentation (no public weights) |
| DTI atlas ROI exploratory imaging | |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-analysis.txt
cp config/config_template.yaml config/config.yaml
# Download datasets — see docs/DATA_SETUP.md
python scripts/run_endotype_pipeline.py
```

## Pipeline steps

| Step | Script | Primary outputs |
|------|--------|-----------------|
| 1 | `recluster_tn_only_subtypes.py` | `results/molecular_subtypes.csv`, DE, pathways |
| 2 | `fix_deconvolution_and_regenerate_figures.py` | `results/blood_deconvolution_improved/` |
| 3 | `integrate_ion_cci_pathways_with_subtypes.py` | IoN-CCI × subtype tables |
| 4 | `run_high_impact_cross_dataset_analyses.py` | `results/high_impact_integration/` |
| 5 | `run_endotype_validation_tiers.py` | `results/endotype_validation/` |
| 6 | `run_endotype_framework_analyses.py` | `results/endotype_framework/` |
| 7 | `run_dti_nerve_adjacent_cohort.py` | `results/dti_nerve_adjacent/` |
| 8 | `run_imaging_qc_audit.py` | imaging QC tables |
| 9 | `run_manual_cnv_pilot_cohort.py --export-only` | manual pilot infrastructure |
| 10 | `run_natcomm_extension_analyses.py` | `results/natcomm_package/` |
| 11 | `run_k3_sensitivity_analysis.py` | `results/tn_only_subtyping/k3_sensitivity/` |

Run subsets:

```bash
python scripts/run_endotype_pipeline.py --only high_impact,endotype_framework
python scripts/run_endotype_pipeline.py --from-step 4
```

## Data requirements

Public cohorts: GSE186505, GSE240432, GSE197289, IoN-CCI, OpenNeuro [ds005713](https://openneuro.org/datasets/ds005713), GSE177034/GSE124272/GSE150408 (NP meta).

See [docs/DATA_SETUP.md](docs/DATA_SETUP.md).

## Stanford external validation

Pre-registered plan: [docs/STANFORD_VALIDATION_SAP.md](docs/STANFORD_VALIDATION_SAP.md).

Minimum ask: **≥25 pre-op TN blood RNA-seq** + clinical/imaging stratification axis.

## Author

[vsanker21](https://github.com/vsanker21)
