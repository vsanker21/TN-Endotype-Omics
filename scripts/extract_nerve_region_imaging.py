"""
Extract nerve-adjacent imaging metrics from OpenNeuro ds005713 NIfTI.

Primary (DTI-first): atlas-refined ROIs on FA/MD/RD/AD maps — nerve-adjacent
anatomical corridors, NOT validated CN V segmentations.

Secondary (de-emphasized): legacy percentile T1/T2 intensity ROIs (QC confounds).

SEVB-Net CN V masks used only when trained weights / manual pilot masks exist.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

from aim2_brain_networks.nerve_rois.atlas_rois import ATLAS_ROI_DEFINITIONS, build_roi_masks

try:
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity

    DIPY_AVAILABLE = True
except ImportError:
    DIPY_AVAILABLE = False

RoiMethod = Literal["atlas", "legacy", "both"]


def _load_nifti(path: Path) -> Optional[np.ndarray]:
    try:
        return nib.load(str(path)).get_fdata()
    except Exception as exc:
        logger.warning(f"Failed to load {path}: {exc}")
        return None


def _brain_mask(data: np.ndarray, percentile: float = 10) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return np.zeros_like(data, dtype=bool)
    pos = finite[finite > 0]
    thresh = np.percentile(pos, percentile) if len(pos) else 0
    return (data > thresh) & np.isfinite(data)


def _legacy_cisternal_mask(shape: tuple[int, ...]) -> np.ndarray:
    x, y, z = shape[:3]
    mask = np.zeros(shape[:3], dtype=bool)
    z_start = int(z * 0.75)
    mask[:, :, z_start:] = True
    lateral = int(x * 0.35)
    mask[lateral:-lateral, :, :] = False
    mask[:lateral, :, :] = True
    mask[-lateral:, :, :] = True
    return mask


def _legacy_brainstem_mask(shape: tuple[int, ...]) -> np.ndarray:
    x, y, z = shape[:3]
    mask = np.zeros(shape[:3], dtype=bool)
    z_start = int(z * 0.70)
    x_lo, x_hi = int(x * 0.30), int(x * 0.70)
    y_lo, y_hi = int(y * 0.30), int(y * 0.70)
    mask[x_lo:x_hi, y_lo:y_hi, z_start:] = True
    return mask


def _roi_stats(values: np.ndarray, mask: np.ndarray) -> dict:
    v = values[mask & np.isfinite(values)]
    if len(v) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "n_voxels": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "n_voxels": int(len(v)),
    }


def _find_files(subject_dir: Path) -> dict:
    out = {"t1": None, "t2": None, "dwi": None, "bval": None, "bvec": None}
    if not subject_dir.exists():
        return out
    t1 = list(subject_dir.rglob("*T1w*.nii.gz"))
    t2 = list(subject_dir.rglob("*T2w.nii.gz"))
    dwi = list(subject_dir.rglob("*dwi*.nii.gz"))
    if t1:
        out["t1"] = t1[0]
    if t2:
        out["t2"] = t2[0]
    if dwi:
        out["dwi"] = dwi[0]
        bval = list(dwi[0].parent.glob("*.bval"))
        bvec = list(dwi[0].parent.glob("*.bvec"))
        out["bval"] = bval[0] if bval else None
        out["bvec"] = bvec[0] if bvec else None
    return out


def _load_sevb_cnv_mask(subject_id: str, sevb_results_dir: Path) -> Optional[np.ndarray]:
    sub_dir = sevb_results_dir / subject_id
    for name in (
        f"{subject_id}_cnv_final_mask.nii.gz",
        f"{subject_id}_cnv_fine_mask.nii.gz",
        f"{subject_id}_cnv_coarse_mask.nii.gz",
    ):
        path = sub_dir / name
        if path.exists():
            data = _load_nifti(path)
            if data is not None:
                return (data > 0).astype(bool)
    return None


def _load_manual_cnv_mask(subject_id: str, manual_dir: Path) -> Optional[np.ndarray]:
    for name in (f"{subject_id}_cnv_manual.nii.gz", f"{subject_id}_cnv_manual.nii"):
        path = manual_dir / "masks" / name
        if path.exists():
            data = _load_nifti(path)
            if data is not None:
                return (data > 0.5).astype(bool)
    return None


def _compute_dti_maps(dwi_path: Path, bval_path: Path, bvec_path: Path) -> Optional[dict]:
    if not DIPY_AVAILABLE:
        return None
    try:
        dwi_data = _load_nifti(dwi_path)
        if dwi_data is None or dwi_data.ndim != 4:
            return None
        bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
        gtab = gradient_table(bvals, bvecs)
        fit = TensorModel(gtab).fit(dwi_data)
        fa = np.nan_to_num(fractional_anisotropy(fit.evals), nan=0.0)
        md = np.nan_to_num(mean_diffusivity(fit.evals), nan=0.0)
        evals = fit.evals
        rd = np.nan_to_num(np.mean(evals[..., 1:], axis=-1), nan=0.0)
        ad = np.nan_to_num(evals[..., 0], nan=0.0)
        return {"fa": fa, "md": md, "rd": rd, "ad": ad}
    except Exception as exc:
        logger.debug(f"DTI fit failed: {exc}")
        return None


def _append_dti_roi_metrics(rec: dict, maps: dict, roi_masks: dict[str, np.ndarray], prefix: str) -> None:
    brain_dwi = _brain_mask(maps["fa"])
    for roi_name, roi_mask in roi_masks.items():
        m = roi_mask & brain_dwi
        if m.shape != maps["fa"].shape:
            continue
        for metric, arr in maps.items():
            s = _roi_stats(arr, m)
            rec[f"{metric}_{prefix}{roi_name}_mean"] = s["mean"]
            rec[f"{metric}_{prefix}{roi_name}_n_voxels"] = s["n_voxels"]
        rec[f"roi_method_{prefix.rstrip('_') or 'primary'}"] = prefix.rstrip("_") or "atlas"


def extract_subject_nerve_metrics(
    subject_id: str,
    openeuro_dir: Path,
    compute_dti: bool = True,
    roi_method: RoiMethod = "atlas",
    include_t1_t2: bool = False,
    sevb_results_dir: Optional[Path] = None,
    manual_pilot_dir: Optional[Path] = None,
) -> dict:
    """Extract nerve-adjacent metrics for one subject."""
    subject_dir = openeuro_dir / subject_id
    files = _find_files(subject_dir)
    rec = {"subject_id": subject_id, "roi_method": roi_method}

    t1_data = _load_nifti(files["t1"]) if files["t1"] else None
    t2_data = _load_nifti(files["t2"]) if files["t2"] else None
    ref = t1_data if t1_data is not None else t2_data

    # Legacy T1/T2 (de-emphasized; optional)
    if include_t1_t2 and ref is not None:
        brain = _brain_mask(ref)
        for roi, mask_fn, tag in [
            ("cisternal", _legacy_cisternal_mask, "legacy"),
            ("brainstem", _legacy_brainstem_mask, "legacy"),
        ]:
            m = mask_fn(ref.shape) & brain
            if t1_data is not None:
                s = _roi_stats(t1_data, m)
                rec[f"t1_{tag}_{roi}_mean"] = s["mean"]
            if t2_data is not None:
                s = _roi_stats(t2_data, m)
                rec[f"t2_{tag}_{roi}_mean"] = s["mean"]

    # DTI + atlas ROIs (primary)
    dti_ok = False
    if compute_dti and files["dwi"] and files["bval"] and files["bvec"]:
        maps = _compute_dti_maps(files["dwi"], files["bval"], files["bvec"])
        if maps is not None:
            dti_ok = True
            rec["dti_computed"] = True
            brain_dti = _brain_mask(maps["fa"])
            rec["dti_brain_voxels"] = int(brain_dti.sum())

            if roi_method in ("atlas", "both"):
                atlas_masks = build_roi_masks(maps["fa"].shape, brain_dti)
                _append_dti_roi_metrics(rec, maps, atlas_masks, prefix="")

            if roi_method in ("legacy", "both"):
                legacy_masks = {
                    "cisternal": _legacy_cisternal_mask(maps["fa"].shape) & brain_dti,
                    "brainstem": _legacy_brainstem_mask(maps["fa"].shape) & brain_dti,
                }
                _append_dti_roi_metrics(rec, maps, legacy_masks, prefix="legacy_")

    else:
        rec["dti_computed"] = False

    # Manual pilot CN V mask (n=10 pilot; supplementary only)
    if manual_pilot_dir is not None:
        manual = _load_manual_cnv_mask(subject_id, manual_pilot_dir)
        if manual is not None and compute_dti and dti_ok:
            maps = _compute_dti_maps(files["dwi"], files["bval"], files["bvec"])
            if maps is not None and manual.shape == maps["fa"].shape:
                _append_dti_roi_metrics(rec, maps, {"manual_cnv": manual}, prefix="manual_")
                rec["manual_cnv_mask"] = True
        elif manual is not None:
            rec["manual_cnv_mask"] = True

    # SEVB-Net (future work — only if weights present)
    if sevb_results_dir is not None:
        cnv_mask = _load_sevb_cnv_mask(subject_id, sevb_results_dir)
        if cnv_mask is not None:
            rec["sevb_mask_available"] = True

    rec["atlas_roi_definitions"] = "|".join(ATLAS_ROI_DEFINITIONS.keys())
    return rec


def find_subjects_with_dwi(openeuro_dir: Path, subject_filter: list[str] | None = None) -> list[str]:
    """Return subject IDs with loadable DWI + bval/bvec."""
    out = []
    for sub_dir in sorted(openeuro_dir.glob("sub-*")):
        sid = sub_dir.name
        if subject_filter and sid not in subject_filter:
            continue
        files = _find_files(sub_dir)
        if not files["dwi"] or not files["bval"] or not files["bvec"]:
            continue
        try:
            nib.load(str(files["dwi"]))
            out.append(sid)
        except Exception:
            continue
    return out


def extract_cohort_nerve_metrics(
    subject_ids: list[str],
    openeuro_dir: Path,
    output_csv: Path,
    compute_dti: bool = True,
    roi_method: RoiMethod = "atlas",
    include_t1_t2: bool = False,
    manual_pilot_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Batch extract nerve-adjacent metrics."""
    rows = []
    for i, sid in enumerate(subject_ids, 1):
        logger.info(f"[{i}/{len(subject_ids)}] {sid}")
        rows.append(
            extract_subject_nerve_metrics(
                sid,
                openeuro_dir,
                compute_dti=compute_dti,
                roi_method=roi_method,
                include_t1_t2=include_t1_t2,
                manual_pilot_dir=manual_pilot_dir,
            )
        )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    n_dti = int(df["dti_computed"].sum()) if "dti_computed" in df.columns else 0
    logger.info(f"Saved {output_csv} — n={len(df)}, DTI computed={n_dti}")
    return df
