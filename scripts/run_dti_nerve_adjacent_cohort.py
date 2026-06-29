"""
Full-cohort DTI nerve-adjacent analysis (atlas-refined ROIs) for ds005713.

Primary metrics: FA/MD/RD/AD in atlas ROIs (rez, cisternal, pons nuclei proxy).
MVD subset analyzed separately for outcome associations.

Usage:
  python scripts/run_dti_nerve_adjacent_cohort.py
  python scripts/run_dti_nerve_adjacent_cohort.py --mvd-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.extract_nerve_region_imaging import extract_cohort_nerve_metrics, find_subjects_with_dwi

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

OUT = project_root / "results" / "dti_nerve_adjacent"
OPENEURO = project_root / "data" / "external" / "openneuro" / "ds005713"

# Primary DTI ROI columns (atlas-refined)
PRIMARY_DTI_ROIS = ["rez_bilateral", "cisternal_bilateral", "pons_nuclei_proxy"]
PRIMARY_METRICS = ["fa", "md", "rd", "ad"]


def _primary_columns(df: pd.DataFrame) -> list[str]:
    cols = [f"{m}_{r}_mean" for m in PRIMARY_METRICS for r in PRIMARY_DTI_ROIS if f"{m}_{r}_mean" in df.columns]
    return cols


def run_statistical_tests(merged: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """DTI-first tests: imaging endotype and MVD outcome."""
    rows = []
    dti_cols = _primary_columns(merged)

    for col in dti_cols:
        if merged[col].notna().sum() < 8:
            continue
        if "imaging_endotype_label" in merged.columns:
            central = merged.loc[merged["imaging_endotype_label"] == "central", col].dropna()
            non_c = merged.loc[merged["imaging_endotype_label"] != "central", col].dropna()
            if len(central) >= 3 and len(non_c) >= 3:
                _, p = stats.mannwhitneyu(central, non_c, alternative="two-sided")
                rows.append({
                    "metric": col,
                    "comparison": "central_vs_noncentral_imaging_endotype",
                    "mean_group_a": central.mean(),
                    "mean_group_b": non_c.mean(),
                    "mannwhitney_p": p,
                    "n_a": len(central),
                    "n_b": len(non_c),
                    "metric_tier": "primary_dti_atlas",
                })
        if "outcome_binary" in merged.columns:
            pos = merged.loc[merged["outcome_binary"] == 1, col].dropna()
            neg = merged.loc[merged["outcome_binary"] == 0, col].dropna()
            if len(pos) >= 2 and len(neg) >= 2:
                _, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
                rows.append({
                    "metric": col,
                    "comparison": "mvd_pos_vs_neg",
                    "mean_group_a": pos.mean(),
                    "mean_group_b": neg.mean(),
                    "mannwhitney_p": p,
                    "n_a": len(pos),
                    "n_b": len(neg),
                    "metric_tier": "primary_dti_atlas",
                })

    tests = pd.DataFrame(rows)
    tests.to_csv(out_dir / "dti_nerve_adjacent_tests.csv", index=False)
    return tests


def write_summary(df: pd.DataFrame, mvd_df: pd.DataFrame, out_dir: Path) -> None:
    n_dti = int(df["dti_computed"].sum()) if "dti_computed" in df.columns else 0
    lines = [
        "# DTI Nerve-Adjacent Cohort Analysis",
        "",
        "## Design",
        "- **Primary:** Atlas-refined ROIs on DTI (FA/MD/RD/AD)",
        "- **ROIs:** rez_bilateral, cisternal_bilateral, pons_nuclei_proxy",
        "- **Not claimed:** CN V-specific segmentation (SEVB-Net → future work / manual pilot supplementary)",
        "",
        f"- Full cohort with DWI files: {len(df)}",
        f"- DTI successfully computed: {n_dti}",
        f"- MVD subset with DTI: {int(mvd_df['dti_computed'].sum()) if 'dti_computed' in mvd_df.columns else len(mvd_df)}",
        "",
        "## Outputs",
        "- `dti_nerve_adjacent_full_cohort.csv`",
        "- `dti_nerve_adjacent_mvd_cohort.csv`",
        "- `dti_nerve_adjacent_mvd_merged.csv`",
        "- `dti_nerve_adjacent_tests.csv`",
    ]
    (out_dir / "DTI_NERVE_ADJACENT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvd-only", action="store_true", help="Process MVD subjects only")
    parser.add_argument("--skip-dti", action="store_true", help="Skip DTI computation (debug)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not OPENEURO.exists():
        logger.error(f"OpenNeuro not found: {OPENEURO}")
        sys.exit(1)

    part = pd.read_csv(OPENEURO / "participants.tsv", sep="\t")
    mvd_ids = part.loc[part["Surgery_type"] == "MVD", "BIDS_ID"].astype(str).tolist()

    if args.mvd_only:
        subject_ids = [s for s in find_subjects_with_dwi(OPENEURO) if s in mvd_ids and not s.endswith("fu")]
    else:
        subject_ids = [s for s in find_subjects_with_dwi(OPENEURO) if not s.endswith("fu")]

    logger.info(f"Processing {len(subject_ids)} baseline subjects with DWI")

    df = extract_cohort_nerve_metrics(
        subject_ids,
        OPENEURO,
        OUT / "dti_nerve_adjacent_full_cohort.csv",
        compute_dti=not args.skip_dti,
        roi_method="atlas",
        include_t1_t2=False,
    )

    mvd_baseline = [s for s in subject_ids if s in mvd_ids]
    mvd_df = df[df["subject_id"].isin(mvd_baseline)].copy()
    mvd_df.to_csv(OUT / "dti_nerve_adjacent_mvd_cohort.csv", index=False)

    # Merge with endotype labels
    endo_path = project_root / "results" / "endotype_framework" / "imaging_endotype_labels.csv"
    if endo_path.exists():
        endo = pd.read_csv(endo_path)
        part2 = part.rename(columns={"BIDS_ID": "subject_id"})
        merged = mvd_df.merge(endo, on="subject_id", how="left")
        merged = merged.merge(
            part2[["subject_id", "Surgery_outcome", "Surgery_type"]],
            on="subject_id",
            how="left",
        )
        merged["outcome_binary"] = (merged["Surgery_outcome"] == "Pos").astype(int)
        merged.to_csv(OUT / "dti_nerve_adjacent_mvd_merged.csv", index=False)
        run_statistical_tests(merged, OUT)

    write_summary(df, mvd_df, OUT)
    logger.info(f"Complete → {OUT}")


if __name__ == "__main__":
    main()
