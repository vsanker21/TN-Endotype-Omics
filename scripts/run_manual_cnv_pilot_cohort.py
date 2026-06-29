"""
n=10 manual CN V segmentation pilot cohort for ds005713.

Selects 10 MVD subjects (stratified by imaging endotype + outcome), exports T2 +
atlas ROI overlays for expert review, and computes DTI metrics when manual masks
are placed in data/manual_cnv_pilot/masks/.

Scientific framing: Supplementary validation only — not primary CN V claims until
expert masks + inter-rater QC exist. SEVB-Net remains future work.

Usage:
  python scripts/run_manual_cnv_pilot_cohort.py --export-only
  python scripts/run_manual_cnv_pilot_cohort.py  # after masks placed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aim2_brain_networks.nerve_rois.atlas_rois import build_roi_masks
from scripts.extract_nerve_region_imaging import _brain_mask, _find_files, _load_nifti, extract_subject_nerve_metrics

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

PILOT_DIR = project_root / "data" / "manual_cnv_pilot"
OUT = project_root / "results" / "manual_cnv_pilot"
OPENEURO = project_root / "data" / "external" / "openneuro" / "ds005713"
N_PILOT = 10


def select_pilot_subjects() -> list[str]:
    """Stratified n=10 from MVD cohort with DTI + imaging endotype labels."""
    part = pd.read_csv(OPENEURO / "participants.tsv", sep="\t").rename(columns={"BIDS_ID": "subject_id"})
    endo_path = project_root / "results" / "endotype_framework" / "imaging_endotype_labels.csv"
    if not endo_path.exists():
        raise FileNotFoundError("Run endotype framework first: imaging_endotype_labels.csv missing")

    endo = pd.read_csv(endo_path)
    mvd = part.merge(endo, on="subject_id", how="inner")
    mvd = mvd[(mvd["Surgery_type"] == "MVD") & (mvd["Surgery_outcome"].isin(["Pos", "Neg"]))]
    mvd = mvd[~mvd["subject_id"].astype(str).str.endswith("fu")]

    # Require loadable DWI
    eligible = []
    for sid in mvd["subject_id"].astype(str):
        files = _find_files(OPENEURO / sid)
        if files["dwi"] and files["bval"] and files["bvec"]:
            try:
                nib.load(str(files["dwi"]))
                eligible.append(sid)
            except Exception:
                pass
    mvd = mvd[mvd["subject_id"].astype(str).isin(eligible)].copy()

    selected: list[str] = []
    # ~3 per endotype label, balance outcome where possible
    for label in ["central", "peripheral", "intermediate"]:
        sub = mvd[mvd["imaging_endotype_label"] == label]
        if len(sub) == 0:
            continue
        pos = sub[sub["Surgery_outcome"] == "Pos"]
        neg = sub[sub["Surgery_outcome"] == "Neg"]
        quota = max(1, N_PILOT // 3)
        for frame in (pos, neg):
            for sid in frame["subject_id"].astype(str).head(quota // 2 + 1):
                if sid not in selected and len(selected) < N_PILOT:
                    selected.append(sid)

    for sid in mvd["subject_id"].astype(str):
        if len(selected) >= N_PILOT:
            break
        if sid not in selected:
            selected.append(sid)

    return selected[:N_PILOT]


def export_pilot_nifti(subject_ids: list[str]) -> None:
    """Export T2 + combined atlas ROI label map for manual annotation reference."""
    export_dir = PILOT_DIR / "export_for_annotation"
    export_dir.mkdir(parents=True, exist_ok=True)

    for sid in subject_ids:
        files = _find_files(OPENEURO / sid)
        t2_path = files["t2"]
        if not t2_path:
            logger.warning(f"{sid}: no T2 — skip export")
            continue
        t2 = _load_nifti(t2_path)
        if t2 is None:
            continue
        brain = _brain_mask(t2)
        masks = build_roi_masks(t2.shape, brain)
        label = np.zeros(t2.shape[:3], dtype=np.uint8)
        for i, name in enumerate(masks, start=1):
            label[masks[name]] = i

        ref_img = nib.load(str(t2_path))
        nib.save(nib.Nifti1Image(t2.astype(np.float32), ref_img.affine, ref_img.header), export_dir / f"{sid}_t2_reference.nii.gz")
        nib.save(nib.Nifti1Image(label, ref_img.affine, ref_img.header), export_dir / f"{sid}_atlas_roi_labels.nii.gz")

        # Axial montage for quick review
        z_mid = t2.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, z_off in zip(axes, (-20, 0, 20)):
            z = max(0, min(t2.shape[2] - 1, z_mid + z_off))
            ax.imshow(t2[:, :, z].T, cmap="gray", origin="lower")
            ax.contour(label[:, :, z].T, levels=[1, 2, 3], colors=["r", "g", "b"], linewidths=0.8)
            ax.set_title(f"z={z}")
            ax.axis("off")
        fig.suptitle(f"{sid} — T2 + atlas ROIs (R=REZ, G=cisternal, B=pons)")
        plt.tight_layout()
        plt.savefig(export_dir / f"{sid}_montage.png", dpi=150, bbox_inches="tight")
        plt.close()

    readme = PILOT_DIR / "ANNOTATION_GUIDE.md"
    readme.write_text(
        "\n".join(
            [
                "# Manual CN V Segmentation Pilot (n=10)",
                "",
                "## Purpose",
                "Supplementary expert segmentation for CN V on ds005713 T2 — **not** primary manuscript claim until QC complete.",
                "",
                "## Instructions",
                f"1. Subjects: {', '.join(subject_ids)}",
                "2. Reference: `export_for_annotation/{subject}_t2_reference.nii.gz`",
                "3. Atlas guide overlay: `{subject}_atlas_roi_labels.nii.gz` (labels 1=REZ, 2=cisternal, 3=pons)",
                "4. Save binary CN V mask as: `masks/{subject}_cnv_manual.nii.gz` (same space as T2)",
                "5. Re-run: `python scripts/run_manual_cnv_pilot_cohort.py`",
                "",
                "## SEVB-Net status",
                "Automated SEVB-Net segmentation → **future work** unless author weights obtained.",
                "Manual pilot results reported in Supplementary only.",
            ]
        ),
        encoding="utf-8",
    )


def run_pilot_metrics(subject_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in subject_ids:
        rows.append(
            extract_subject_nerve_metrics(
                sid,
                OPENEURO,
                compute_dti=True,
                roi_method="atlas",
                manual_pilot_dir=PILOT_DIR,
            )
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "manual_cnv_pilot_metrics.csv", index=False)

    n_manual = int(df.get("manual_cnv_mask", pd.Series(dtype=bool)).sum()) if "manual_cnv_mask" in df.columns else 0
    meta = {
        "n_pilot": len(subject_ids),
        "subjects": subject_ids,
        "n_manual_masks_present": n_manual,
        "sevb_status": "future_work",
        "claim_tier": "supplementary_only_if_manual_qc_complete",
    }
    (OUT / "pilot_cohort_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_DIR / "masks").mkdir(exist_ok=True)

    subject_ids = select_pilot_subjects()
    (PILOT_DIR / "pilot_subjects.json").write_text(json.dumps(subject_ids, indent=2), encoding="utf-8")
    logger.info(f"Pilot cohort (n={len(subject_ids)}): {subject_ids}")

    export_pilot_nifti(subject_ids)
    if not args.export_only:
        run_pilot_metrics(subject_ids)

    logger.info(f"Pilot complete → {OUT}")


if __name__ == "__main__":
    main()
