# SEVB-Net checkpoint and training data request

**Purpose:** Obtain pretrained SEVB-Net weights and/or de-identified training data for external validation on OpenNeuro ds005713 (trigeminal neuralgia T2 cohort).

**To:** Hanfeng Yang, MD (Corresponding author)  
**Email:** yhfctjr@yahoo.com  
**CC (optional):** Man Li, Shanghai United Imaging Intelligence Co., Ltd.

**Subject:** Request for SEVB-Net pretrained weights and/or de-identified T2 segmentation dataset (Front Neurosci 2023; doi:10.3389/fnins.2023.1265032)

---

Dear Dr. Yang and colleagues,

We are preparing an independent multimodal study of trigeminal neuralgia (TN) that integrates blood transcriptomic endotypes with structural MRI. For anatomically specific trigeminal nerve (CN V) quantification, we implemented your **SEVB-Net** architecture (coarse 1 mm + fine 0.5 mm cascade, ωDoubleLoss) following:

> Zhang C, Li M, Luo Z, et al. Deep learning-driven MRI trigeminal nerve segmentation with SEVB-net. *Front Neurosci.* 2023;17:1265032. doi:10.3389/fnins.2023.1265032

Our reimplementation matches the reported ~2.2M parameters per stage and paper-specified patch geometry. We plan **external validation** on the public OpenNeuro dataset **ds005713** (pre-operative T2-weighted MRI in TN patients undergoing microvascular decompression).

We could not locate publicly deposited model checkpoints or source code. Your data availability statement indicates that raw data can be made available on request. We respectfully request **either or both** of the following:

### Option A — Pretrained checkpoints (preferred for external validation)
- Coarse-stage weights (1 mm isotropic, patch 128×128×192)
- Fine-stage weights (0.5 mm isotropic, patch 64×192×160)
- Preferred format: PyTorch `state_dict` (`.pth`) compatible with a single-channel T2 input and binary CN V output

### Option B — De-identified training set (for local reproduction)
- Paired 3D T2 volumes and expert CN V segmentations used for training/validation
- Acquisition parameters (sequence type, e.g., STIR-SPACE) and preprocessing steps (N4, resampling)
- Train/validation/test split identifiers if available

### Intended use
- **Research only:** CN V morphological features (volume, cisternal segment intensity) linked to molecular endotypes and MVD outcome in ds005713
- **No commercial use**
- We will cite your work and report domain-shift performance (internal vs. OpenNeuro) transparently
- We can share our validation metrics (DSC vs. manual review on a subset) if helpful

### Ethics and data handling
We will comply with your institutional ethics approval (2023ER178-1) and any data transfer agreement. Our OpenNeuro analyses use publicly shared, de-identified BIDS data under its own terms.

Thank you for considering this request. We would be grateful for guidance on the appropriate contact at United Imaging Intelligence if checkpoints are distributed through the uAI research portal.

Sincerely,

[Your name]  
[Institution]  
[Email]

---

## After receiving checkpoints

1. Place files at:
   ```
   data/models/sevb_net/coarse_best.pth
   data/models/sevb_net/fine_best.pth
   ```
2. Validate:
   ```bash
   python scripts/acquire_sevb_net_weights.py
   ```
3. Run segmentation:
   ```bash
   python scripts/run_sevb_net_segmentation.py
   ```
4. Re-run nerve-region imaging integration:
   ```bash
   python scripts/extract_nerve_region_imaging.py
   ```

## Manuscript framing (honest claims)

| With SEVB weights + QC | Without weights |
|------------------------|-----------------|
| CN V-specific T2 metrics from automated segmentation | Posterior-fossa **proxy ROIs** only |
| Report DSC on manual review subset | Do not claim validated CN V pathology |
| Domain-shift caveat (STIR-SPACE vs. ds005713 T2) | Methods cite SEVB-Net as planned upgrade |
