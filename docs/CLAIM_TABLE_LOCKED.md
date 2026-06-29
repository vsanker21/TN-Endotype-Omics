# Locked Manuscript Claim Table

**Version:** 1.1 (locked 2026-06-28)  
**Scope:** Nature Communications package + endotype framework  
**Rule:** Main-text claims must appear in the **Supported** column. Everything else is Discussion, Supplementary, or Future Work.

---

## Primary claims (main text)

| Claim | Evidence | n / cohort | Strength |
|-------|----------|------------|----------|
| TN blood endotypes (k=2) discoverable in GSE186505 | Internal clustering + DE + module scores | n=10 TN | Discovery |
| Endotype signatures portable as frozen gene panels + IoN-CCI modules | `portable_endotype_signatures.json` | n=10 | Discovery |
| Human TG cell-type programs anchor blood DE mechanism | Tier 3 TG mapping | 38k cells | Mechanistic |
| IoN-CCI modules **scoreable** in external NP blood | 3-cohort meta (GSE177034, GSE124272, GSE150408) | 49–42/cohort | Boundary / scoreability |
| Population-level blood ↔ imaging endotype concordance (exploratory) | Centroid bridge + DTI-first imaging features | OpenNeuro ds005713 | Hypothesis-generating |
| MVD outcome differs by imaging endotype (exploratory, non-significant) | Fisher + logistic (non-circular) | n≈31 MVD | Exploratory |
| Drug / ligand–receptor translation hypotheses | Curated + Open Targets + TG LR pairs | In silico | Hypothesis |

---

## Supplementary-only claims

| Claim | Condition | Status |
|-------|-----------|--------|
| Manual CN V segmentation pilot (n=10) | Expert masks in `data/manual_cnv_pilot/masks/` + QC | **Infrastructure ready; masks pending** |
| Atlas-refined DTI nerve-adjacent ROI associations | `results/dti_nerve_adjacent/` | **Primary imaging metric path without SEVB** |
| k=3 sensitivity, permutation/bootstrap CIs | Existing supplementary analyses | Available |

---

## Future work (not claimed)

| Item | Rationale |
|------|-----------|
| **SEVB-Net automated CN V segmentation** | No public weights; author request outstanding → **future work** |
| Independent TN k=2 subtype replication | Requires Stanford / external TN blood cohort |
| Matched blood–imaging–MVD triangulation | n=0 matched patients (`patient_linkage_audit.json`) |
| Meta-analytic convergent peripheral axis at α=0.05 | Fisher p=0.135; inconsistent directions across NP cohorts |
| Clinical efficacy of repurposed drugs | In silico only |
| CN V-specific pathology from proxy ROIs | Anatomically adjacent only — not nerve-labeled |

---

## Imaging metric hierarchy (locked)

| Tier | Metrics | Manuscript use |
|------|---------|----------------|
| **1 — Primary** | FA, MD, RD, AD in atlas ROIs (`rez_bilateral`, `cisternal_bilateral`, `pons_nuclei_proxy`) | Main + Fig 5 (DTI) |
| **2 — Secondary** | Whole-brain DTI means (fa_mean, md_mean) | Bridge / MVD models |
| **3 — De-emphasized** | Legacy T1/T2 intensity in percentile ROIs | Supplementary only (QC confounds) |
| **4 — Future / Supp** | SEVB-Net or manual CN V masks | Future work / Supp if pilot QC complete |

---

## Wording templates

**Imaging Methods:**  
“Nerve-adjacent diffusion metrics were quantified in atlas-refined regions anchored to each subject's brain bounding box (bilateral REZ proxy, cisternal corridor, pons nuclei proxy). These regions are anatomically motivated but do not constitute validated trigeminal nerve segmentations.”

**SEVB-Net:**  
“Automated CN V segmentation with SEVB-Net (Zhang et al., 2023) was implemented but not applied in primary analyses because pretrained weights were unavailable. A manual segmentation pilot (n=10) is prepared for supplementary validation.”

**NP meta-analysis:**  
“External neuropathic pain cohorts validate module scoreability and context specificity; they do not replicate TN molecular subtypes.”

---

## Re-run pipeline

```bash
python scripts/run_endotype_framework_analyses.py
python scripts/run_dti_nerve_adjacent_cohort.py
python scripts/run_imaging_qc_audit.py
python scripts/run_manual_cnv_pilot_cohort.py --export-only
python scripts/run_natcomm_extension_analyses.py
```

**Canonical file:** `results/CLAIM_TABLE_LOCKED.md` — update version date if claims change.
