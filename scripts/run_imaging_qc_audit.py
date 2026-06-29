"""
Pre-specified imaging QC audit for atlas nerve-adjacent DTI metrics.

Rationale (CLAIM_TABLE_LOCKED.md): Cisternal ROI associations may confound with scan
coverage / posterior-fossa completeness. This script quantifies QC covariates and tests
whether ROI voxel counts or brain coverage differ by imaging endotype.

Outputs: results/dti_nerve_adjacent/imaging_qc_audit.csv
         results/dti_nerve_adjacent/IMAGING_QC_SAP.md (generated summary)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

OUT = project_root / "results" / "dti_nerve_adjacent"
DOCS = project_root / "docs" / "IMAGING_QC_SAP.md"


def main() -> None:
    dti_path = OUT / "dti_nerve_adjacent_full_cohort.csv"
    if not dti_path.exists():
        raise FileNotFoundError(f"Run run_dti_nerve_adjacent_cohort.py first: {dti_path}")

    df = pd.read_csv(dti_path)
    endo_path = project_root / "results" / "endotype_framework" / "imaging_endotype_labels.csv"
    if endo_path.exists():
        endo = pd.read_csv(endo_path)[["subject_id", "imaging_subtype", "imaging_endotype_label"]]
        df = df.merge(endo, on="subject_id", how="left")

    # QC metrics
    df["qc_dti_success"] = df.get("dti_computed", False).astype(bool)
    df["qc_brain_voxels"] = pd.to_numeric(df.get("dti_brain_voxels"), errors="coerce")
    for roi in ["cisternal_bilateral", "rez_bilateral", "pons_nuclei_proxy"]:
        col = f"fa_{roi}_n_voxels"
        if col in df.columns:
            df[f"qc_{roi}_n_voxels"] = pd.to_numeric(df[col], errors="coerce")
            df[f"qc_{roi}_brain_fraction"] = df[f"qc_{roi}_n_voxels"] / df["qc_brain_voxels"].replace(0, np.nan)

    qc_cols = [c for c in df.columns if c.startswith("qc_")]
    audit = df[["subject_id"] + qc_cols + [c for c in ("imaging_subtype", "imaging_endotype_label") if c in df.columns]]
    audit.to_csv(OUT / "imaging_qc_audit.csv", index=False)

    # Confound tests: QC metrics vs imaging endotype
    test_rows = []
    if "imaging_endotype_label" in audit.columns:
        for qc in qc_cols:
            if audit[qc].notna().sum() < 10:
                continue
            central = audit.loc[audit["imaging_endotype_label"] == "central", qc].dropna()
            non_c = audit.loc[audit["imaging_endotype_label"] != "central", qc].dropna()
            if len(central) >= 3 and len(non_c) >= 3:
                _, p = stats.mannwhitneyu(central, non_c, alternative="two-sided")
                test_rows.append({
                    "qc_metric": qc,
                    "comparison": "central_vs_noncentral_endotype",
                    "mean_central": central.mean(),
                    "mean_non_central": non_c.mean(),
                    "mannwhitney_p": p,
                    "interpretation": "potential_confound" if p < 0.05 else "no_endotype_qc_difference",
                })

    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "imaging_qc_endotype_tests.csv", index=False)

    n_ok = int(df["qc_dti_success"].sum())
    n_total = len(df)
    confounds = tests[tests.get("interpretation") == "potential_confound"] if len(tests) else pd.DataFrame()

    sap_lines = [
        "# Imaging QC Standard Analysis Plan (Pre-specified)",
        "",
        "**Version:** 1.0 (locked 2026-06-28)",
        "**Reference:** `results/CLAIM_TABLE_LOCKED.md` (Tier 1 DTI atlas ROIs)",
        "",
        "## Objective",
        "",
        "Assess whether atlas nerve-adjacent DTI associations (especially cisternal MD/AD) ",
        "are confounded by scan coverage, brain mask volume, or ROI voxel counts across imaging endotypes.",
        "",
        "## QC metrics (per subject)",
        "",
        "| Metric | Definition | Pass criterion |",
        "|--------|------------|----------------|",
        "| `qc_dti_success` | Tensor fit completed | Required for inclusion |",
        "| `qc_brain_voxels` | Voxels in DTI brain mask | Report; flag bottom 10% |",
        "| `qc_*_n_voxels` | Atlas ROI voxel count | Report; flag if <1,000 voxels |",
        "| `qc_*_brain_fraction` | ROI voxels / brain voxels | Report; test vs endotype |",
        "",
        f"## Cohort QC summary",
        "",
        f"- Subjects in audit: **{n_total}**",
        f"- DTI successful: **{n_ok}** ({100*n_ok/max(n_total,1):.1f}%)",
        "",
        "## Endotype confound screening",
        "",
        "Mann–Whitney U: central vs non-central imaging endotype on each QC metric.",
        "p<0.05 → flag as **potential confound** (adjust interpretation; do not claim CN V pathology).",
        "",
    ]
    if len(confounds):
        sap_lines.append("**Flagged QC metrics (p<0.05):**")
        for _, r in confounds.iterrows():
            sap_lines.append(f"- `{r['qc_metric']}`: p={r['mannwhitney_p']:.4g}")
    else:
        sap_lines.append("No QC metric showed significant central vs non-central difference at α=0.05 in current audit.")

    sap_lines.extend([
        "",
        "## Manuscript language",
        "",
        "Report DTI atlas ROI findings as **exploratory nerve-adjacent associations**. ",
        "If QC confounds are present, state that associations may partly reflect coverage/completeness. ",
        "Legacy T1/T2 intensity ROIs are supplementary only (Tier 3).",
        "",
        "## Re-run",
        "",
        "```bash",
        "python scripts/run_dti_nerve_adjacent_cohort.py",
        "python scripts/run_imaging_qc_audit.py",
        "```",
    ])

    sap_text = "\n".join(sap_lines)
    DOCS.write_text(sap_text, encoding="utf-8")
    (OUT / "IMAGING_QC_SAP.md").write_text(sap_text, encoding="utf-8")
    print(f"QC audit -> {OUT / 'imaging_qc_audit.csv'}")
    print(f"SAP -> {DOCS}")


if __name__ == "__main__":
    main()
