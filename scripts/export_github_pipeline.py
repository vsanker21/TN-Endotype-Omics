"""
Export focused TN endotype analysis pipeline for GitHub (no figure/manuscript scripts).

Creates a clean tree under ../_github_export/tn-endotype-omics/ ready for git push.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
EXPORT = project_root / "_github_export" / "tn-endotype-omics"

SCRIPTS = [
    "run_endotype_pipeline.py",
    "export_github_pipeline.py",
    "recluster_tn_only_subtypes.py",
    "fix_deconvolution_and_regenerate_figures.py",
    "integrate_ion_cci_pathways_with_subtypes.py",
    "run_ion_cci_de_analysis.py",
    "run_high_impact_cross_dataset_analyses.py",
    "run_endotype_validation_tiers.py",
    "run_endotype_framework_analyses.py",
    "run_dti_nerve_adjacent_cohort.py",
    "run_imaging_qc_audit.py",
    "run_manual_cnv_pilot_cohort.py",
    "run_natcomm_extension_analyses.py",
    "run_k3_sensitivity_analysis.py",
    "extract_nerve_region_imaging.py",
    "process_gse197289_final.py",
    "process_ion_cci_for_aim3.py",
    "monitor_ion_cci_progress.py",
]

AIM2_FILES = [
    "aim2_brain_networks/__init__.py",
    "aim2_brain_networks/nerve_rois/__init__.py",
    "aim2_brain_networks/nerve_rois/atlas_rois.py",
]

DOCS = [
    ("results/CLAIM_TABLE_LOCKED.md", "docs/CLAIM_TABLE_LOCKED.md"),
    ("results/natcomm_package/STANFORD_VALIDATION_SAP.md", "docs/STANFORD_VALIDATION_SAP.md"),
    ("docs/IMAGING_QC_SAP.md", "docs/IMAGING_QC_SAP.md"),
    ("docs/OPENNEURO_ANNEX_RECOVERY.md", "docs/OPENNEURO_ANNEX_RECOVERY.md"),
    ("docs/sevb_net/AUTHOR_WEIGHT_AND_DATA_REQUEST.md", "docs/sevb_net/AUTHOR_WEIGHT_AND_DATA_REQUEST.md"),
]

CONFIG = [
    "config/config_template.yaml",
    "config/analysis_standards.yaml",
    "config/aim2_config.yaml",
]

GITIGNORE = """# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/

# Data & large outputs (download separately — see docs/DATA_SETUP.md)
data/
results/
outputs/
*.nii
*.nii.gz
*.h5
*.h5ad

# Figures (analysis scripts may write optional PNGs locally)
*.png
*.pdf
*.svg
*.docx

# Secrets
config/config.yaml
.env
*.key

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db
logs/
*.log
"""

README = """# TN Endotype Omics — Analysis Pipeline

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
.venv\\Scripts\\activate   # Windows
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
"""

DATA_SETUP = """# Data Setup

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
"""

REQUIREMENTS = """# TN Endotype analysis pipeline (trimmed)
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
loguru>=0.6.0
pyyaml>=6.0
tqdm>=4.64.0
gseapy>=1.0.0
mygene>=3.2.2
nibabel>=3.2.0
dipy>=1.5.0
scanpy>=1.9.0
anndata>=0.8.0
h5py>=3.7.0
statsmodels>=0.13.0
openpyxl>=3.0.0
"""


def copy_file(src_rel: str, dst_rel: str) -> None:
    src = project_root / src_rel
    dst = EXPORT / dst_rel
    if not src.exists():
        print(f"  skip missing: {src_rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {dst_rel}")


def main() -> None:
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)

    (EXPORT / "scripts").mkdir()
    for s in SCRIPTS:
        copy_file(f"scripts/{s}", f"scripts/{s}")

    for rel in AIM2_FILES:
        copy_file(rel, rel)

    for src, dst in DOCS:
        copy_file(src, dst)

    for c in CONFIG:
        copy_file(c, c)

    (EXPORT / "README.md").write_text(README, encoding="utf-8")
    (EXPORT / "docs" / "DATA_SETUP.md").write_text(DATA_SETUP, encoding="utf-8")
    (EXPORT / "requirements-analysis.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (EXPORT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (EXPORT / "data").mkdir(parents=True, exist_ok=True)
    (EXPORT / "results").mkdir(parents=True, exist_ok=True)
    (EXPORT / "data" / ".gitkeep").write_text("", encoding="utf-8")
    (EXPORT / "results" / ".gitkeep").write_text("", encoding="utf-8")

    print(f"\nExport ready: {EXPORT}")
    print(f"Scripts: {len(SCRIPTS)}")


if __name__ == "__main__":
    main()
