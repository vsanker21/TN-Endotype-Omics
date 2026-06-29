"""
Brain-bbox–anchored atlas-refined ROIs for nerve-adjacent DTI/T2 quantification.

Scientific rationale
--------------------
Percentile-of-FOV masks confound with scan coverage and head positioning.
We anchor ROIs to each subject's brain bounding box (tissue mask), placing
regions at literature-informed fractions of brain extent:

  - **rez_bilateral**: Root entry zone proxy at inferior pons (35–55% inferior→superior
    brain extent), bilateral lateral cisternal wings (Miller 1996; Jannetta REZ).
  - **cisternal_bilateral**: Cisternal CN V course — inferolateral posterior fossa
    (15–40% Z, outer 18–28% X bilaterally).
  - **pons_nuclei_proxy**: Trigeminal nucleus / brainstem core — inferior-central
    pons (30–50% Z, central 35–65% XY).

These are NOT validated CN V segmentations. They are nerve-adjacent anatomical
corridors suitable for exploratory DTI (FA/MD/RD/AD) when SEVB-Net weights are
unavailable.

References
----------
- Zhang et al. Front Neurosci 2023 (SEVB-Net; REZ/cisternal anatomy motivation)
- Miller JP et al. J Neurosurg 1996 (REZ localization)
- OpenNeuro ds005713 T2/DTI acquisition (variable coverage — QC required)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ATLAS_ROI_DEFINITIONS = {
    "rez_bilateral": "Bilateral root entry zone proxy (inferior pons, lateral cisternal)",
    "cisternal_bilateral": "Bilateral cisternal CN V course (inferolateral posterior fossa)",
    "pons_nuclei_proxy": "Central pons / trigeminal nucleus region proxy",
}


@dataclass(frozen=True)
class BrainBbox:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    @property
    def x_len(self) -> int:
        return self.x1 - self.x0

    @property
    def y_len(self) -> int:
        return self.y1 - self.y0

    @property
    def z_len(self) -> int:
        return self.z1 - self.z0


def brain_bbox_from_mask(brain_mask: np.ndarray) -> BrainBbox:
    coords = np.where(brain_mask)
    if len(coords[0]) == 0:
        s = brain_mask.shape[:3]
        return BrainBbox(0, s[0], 0, s[1], 0, s[2])
    return BrainBbox(
        int(coords[0].min()),
        int(coords[0].max()) + 1,
        int(coords[1].min()),
        int(coords[1].max()) + 1,
        int(coords[2].min()),
        int(coords[2].max()) + 1,
    )


def _rez_bilateral_mask(shape: tuple[int, ...], brain_mask: np.ndarray) -> np.ndarray:
    bb = brain_bbox_from_mask(brain_mask)
    mask = np.zeros(shape[:3], dtype=bool)
    z_lo = bb.z0 + int(0.35 * bb.z_len)
    z_hi = bb.z0 + int(0.55 * bb.z_len)
    y_lo = bb.y0 + int(0.25 * bb.y_len)
    y_hi = bb.y0 + int(0.75 * bb.y_len)
    wing = max(2, int(0.12 * bb.x_len))
    inner = max(1, int(0.18 * bb.x_len))
    # Left wing
    mask[bb.x0 + inner : bb.x0 + inner + wing, y_lo:y_hi, z_lo:z_hi] = True
    # Right wing
    mask[bb.x1 - inner - wing : bb.x1 - inner, y_lo:y_hi, z_lo:z_hi] = True
    return mask & brain_mask


def _cisternal_bilateral_mask(shape: tuple[int, ...], brain_mask: np.ndarray) -> np.ndarray:
    bb = brain_bbox_from_mask(brain_mask)
    mask = np.zeros(shape[:3], dtype=bool)
    z_lo = bb.z0 + int(0.15 * bb.z_len)
    z_hi = bb.z0 + int(0.42 * bb.z_len)
    y_lo = bb.y0 + int(0.20 * bb.y_len)
    y_hi = bb.y0 + int(0.80 * bb.y_len)
    wing = max(2, int(0.10 * bb.x_len))
    outer = max(1, int(0.04 * bb.x_len))
    mask[bb.x0 + outer : bb.x0 + outer + wing, y_lo:y_hi, z_lo:z_hi] = True
    mask[bb.x1 - outer - wing : bb.x1 - outer, y_lo:y_hi, z_lo:z_hi] = True
    return mask & brain_mask


def _pons_nuclei_proxy_mask(shape: tuple[int, ...], brain_mask: np.ndarray) -> np.ndarray:
    bb = brain_bbox_from_mask(brain_mask)
    mask = np.zeros(shape[:3], dtype=bool)
    z_lo = bb.z0 + int(0.30 * bb.z_len)
    z_hi = bb.z0 + int(0.52 * bb.z_len)
    x_lo = bb.x0 + int(0.32 * bb.x_len)
    x_hi = bb.x0 + int(0.68 * bb.x_len)
    y_lo = bb.y0 + int(0.32 * bb.y_len)
    y_hi = bb.y0 + int(0.68 * bb.y_len)
    mask[x_lo:x_hi, y_lo:y_hi, z_lo:z_hi] = True
    return mask & brain_mask


_ATLAS_BUILDERS: dict[str, Callable[[tuple[int, ...], np.ndarray], np.ndarray]] = {
    "rez_bilateral": _rez_bilateral_mask,
    "cisternal_bilateral": _cisternal_bilateral_mask,
    "pons_nuclei_proxy": _pons_nuclei_proxy_mask,
}


def build_roi_masks(
    shape: tuple[int, ...],
    brain_mask: np.ndarray,
    roi_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return atlas-refined ROI masks intersected with brain tissue."""
    names = roi_names or list(_ATLAS_BUILDERS.keys())
    out: dict[str, np.ndarray] = {}
    for name in names:
        fn = _ATLAS_BUILDERS.get(name)
        if fn is not None:
            out[name] = fn(shape, brain_mask)
    return out
