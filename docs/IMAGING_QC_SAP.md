# Imaging QC Standard Analysis Plan (Pre-specified)

**Version:** 1.0 (locked 2026-06-28)
**Reference:** `results/CLAIM_TABLE_LOCKED.md` (Tier 1 DTI atlas ROIs)

## Objective

Assess whether atlas nerve-adjacent DTI associations (especially cisternal MD/AD) 
are confounded by scan coverage, brain mask volume, or ROI voxel counts across imaging endotypes.

## QC metrics (per subject)

| Metric | Definition | Pass criterion |
|--------|------------|----------------|
| `qc_dti_success` | Tensor fit completed | Required for inclusion |
| `qc_brain_voxels` | Voxels in DTI brain mask | Report; flag bottom 10% |
| `qc_*_n_voxels` | Atlas ROI voxel count | Report; flag if <1,000 voxels |
| `qc_*_brain_fraction` | ROI voxels / brain voxels | Report; test vs endotype |

## Cohort QC summary

- Subjects in audit: **121**
- DTI successful: **121** (100.0%)

## Endotype confound screening

Mann–Whitney U: central vs non-central imaging endotype on each QC metric.
p<0.05 → flag as **potential confound** (adjust interpretation; do not claim CN V pathology).

No QC metric showed significant central vs non-central difference at α=0.05 in current audit.

## Manuscript language

Report DTI atlas ROI findings as **exploratory nerve-adjacent associations**. 
If QC confounds are present, state that associations may partly reflect coverage/completeness. 
Legacy T1/T2 intensity ROIs are supplementary only (Tier 3).

## Re-run

```bash
python scripts/run_dti_nerve_adjacent_cohort.py
python scripts/run_imaging_qc_audit.py
```