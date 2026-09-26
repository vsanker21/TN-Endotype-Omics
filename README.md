# TN Endotype Omics — Analysis Pipeline

> **Status (revision, 2026).** The endotype claims of the original pipeline are **withdrawn**. The corrected
> re-analysis, *"Candidate blood transcriptomic subtypes of trigeminal neuralgia reflect erythroid library
> composition: a corrected re-analysis with trigeminal nerve-injury transcriptomes"*, finds that the k = 2 TN
> partition is not distinguishable from a no-structure null, tracks the globin share of each library, and does not
> correspond to IoN-CCI injury programmes once the IoN-CCI sample labels are corrected. The code for that analysis is
> described in [JHP revision: corrected re-analysis](#jhp-revision-corrected-re-analysis) below. The original
> pipeline is kept unchanged so that the errors disclosed in the revision can be checked against it; its outputs
> should not be used as evidence for TN subtypes.

## JHP revision: corrected re-analysis

All outputs are written to `results/jhp_revision/`. Seeds are fixed in every script.

| Order | Script | What it does | Outputs |
|------|--------|--------------|---------|
| 1 | `scripts/run_jhp_revision_analyses.py` | IoN-CCI label audit and within-tissue PyDESeq2 (PRJNA991739), Reactome ORA, revised modules; clusterability (single-Gaussian/SigClust-type null, gap statistic with k = 1), stability, full-pipeline and expression-matched random gene-set nulls, controls, ABIS NNLS deconvolution and marker scores, limma-trend TN vs control, preranked GSEA | `A*`, `B2`–`B9` |
| 2 | `scripts/run_jhp_revision_globin_check.py` | Globin fraction versus partition; globin depletion and globin residualisation | `B10_*` |
| 3 | `scripts/run_jhp_revision_submitted_randomsets.py` | Submitted gene sets as originally scored against matched random gene sets | `B4c_*` |
| 4 | `scripts/wsl/gse186505_salmon.sh` (Linux/WSL) | Downloads GSE186505 raw reads from ENA with MD5 check and quantifies with salmon 1.10.3 (GENCODE v26, k = 31) | `~/tn_gse186505/quant/` |
| 5 | `scripts/wsl/fastq_headers.sh` (Linux/WSL) | Reads instrument, run, flowcell and lane from the first read headers of each R1 FASTQ | `~/tn_gse186505/fastq_headers.tsv` |
| 6 | `scripts/run_jhp_revision_salmon_requant.py` | Salmon QC, PyDESeq2 TN vs control (sex; sex + globin; globin genes removed), clustering on salmon expression | `C_salmon/` |
| 7 | `scripts/run_jhp_revision_technical.py` | Read depth, mapping rate and flowcell versus partition; flowcell-adjusted TN vs control; TN vs control dispersion with and without globin adjustment | `D_technical/` |
| 8 | `scripts/run_jhp_revision_sp5c_ecm.py` | Sp5C-down extracellular-matrix/Schwann module (added after label correction) through the same tests | `E_sp5c_ecm/` |
| 9 | `scripts/make_jhp_revision_figures.py` | Figures 1–5 and S1–S7 | `figures/` |
| 10 | `scripts/build_jhp_supplementary_tables.py` | Additional file 2 (Tables S1–S17) | `.xlsx` |

**Inputs.**

- `data/processed/GSE186505/human_blood_expr_processed.csv`: the GEO FPKM matrix `GSE186505_fpkm.txt.gz` with gene symbols upper-cased and duplicates removed (19,225 genes × 20 samples; columns are the GEO sample titles).
- IoN-CCI counts in `results/aim3/ion_cci/de_analysis/` (`ion_cci_counts.csv`, `ensembl_to_symbol_mapping.csv`), produced by `process_ion_cci_for_aim3.py` and `run_ion_cci_de_analysis.py`.
- Reference files (Reactome 2022, GO BP 2023, MGI orthologs, ABIS signature matrix) are downloaded automatically to `results/jhp_revision/reference/`.
- `jhp_revision/inputs/` holds the small files that record what the original submission used. Copy them into place before running:
  - `molecular_subtypes.csv` goes to `results/`. These are the submitted partition labels.
  - `ion_cci_human_ortholog_mapping.csv` goes to `results/high_impact_integration/`. These are the submitted module gene lists.
  - `ion_cci_metadata.csv` goes to `results/aim3/ion_cci/de_analysis/`. This is the original, partly incorrect IoN-CCI sample sheet.
  - `sra_metadata.csv` goes to `data/external/ion_cci/`. These are the SRA records used to rebuild the labels.
- Salmon output location: set `TN_SALMON_HOME` to the home directory that contains `tn_gse186505/`, and `TN_TGMAP` to a GENCODE v26 transcript-to-gene map. The defaults are the authors' WSL paths.

**Python packages** (versions used): numpy 2.2.6, pandas 2.2.3, scipy 1.15.2, scikit-learn 1.7.2, statsmodels 0.14.6, pydeseq2 0.5.4, gseapy 1.1.4, matplotlib 3.10.8, openpyxl 3.1.5, requests 2.32.5, mygene 3.2.2. See `requirements-jhp-revision.txt`.

## Original pipeline (superseded; retained for transparency)

The sections below describe the original submission. Its central claims are withdrawn (see above).

### Scientific focus of the original submission (superseded)

See [docs/CLAIM_TABLE_LOCKED.md](docs/CLAIM_TABLE_LOCKED.md).

| Supported in discovery | Requires external cohort |
|------------------------|-------------------------|
| k=2 TN blood endotypes (n=10) | Independent TN subtype replication (Stanford) |
| IoN-CCI module scores in blood | Matched blood–imaging–MVD (n=0 in public data) |
| NP cohort module **scoreability** | SEVB-Net CN V segmentation (no public weights) |
| DTI atlas ROI exploratory imaging | |

### Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-analysis.txt
cp config/config_template.yaml config/config.yaml
# Download datasets — see docs/DATA_SETUP.md
python scripts/run_endotype_pipeline.py
```

### Pipeline steps

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

### Data requirements

Public cohorts: GSE186505, GSE240432, GSE197289, IoN-CCI, OpenNeuro [ds005713](https://openneuro.org/datasets/ds005713), GSE177034/GSE124272/GSE150408 (NP meta).

See [docs/DATA_SETUP.md](docs/DATA_SETUP.md).

### Stanford external validation

Pre-registered plan: [docs/STANFORD_VALIDATION_SAP.md](docs/STANFORD_VALIDATION_SAP.md).

Minimum ask: **≥25 pre-op TN blood RNA-seq** + clinical/imaging stratification axis.

## Author

[vsanker21](https://github.com/vsanker21)
