# OpenNeuro ds005713 — git-annex file recovery

Some BIDS files (e.g. `sub-132`) are **git-annex pointer files**, not loadable NIfTI. This caused 1/121 baseline DTI failures in initial audits.

## Recover missing volumes

From a shell with **git** and **git-annex** installed:

```bash
cd data/external/openneuro/ds005713
git annex get sub-132/anat/sub-132_T1w.nii.gz
git annex get sub-132/dwi/sub-132_dwi.nii.gz
# Or fetch entire dataset:
git annex get .
```

Then re-run:

```bash
python scripts/run_dti_nerve_adjacent_cohort.py
python scripts/run_imaging_qc_audit.py
```

## Current environment note

Git was not available in the automated pipeline shell (Windows PATH). Manual annex fetch required on a machine with git-annex configured for OpenNeuro.
