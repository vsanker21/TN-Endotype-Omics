# Stanford External Validation — Statistical Analysis Plan (SAP)

**Version:** 1.0 (locked prior to data access)  
**Discovery cohort:** GSE186505 (n=10 TN blood; hypothesis-generating)  
**Validation cohort:** Stanford TN blood (target n≥25)

## Primary endpoint
**Peripheral module convergent validity:** `peripheral_ecm_schwann` module score differs across validation groups defined by pre-specified clinical stratification (peripheral-dominant vs central-dominant clinical/imaging features, or continuous axis replication).

- Test: Spearman correlation between discovery-derived `central_axis` score and validation cohort axis (permutation n=9,999)
- Success criterion: r ≥ 0.30, permutation p < 0.05, direction concordant with discovery

## Secondary endpoints
1. Module score directionality: peripheral_ecm_schwann, central_synaptic, ntrk_signaling (Welch t-test / Mann–Whitney with BH-FDR)
2. Portable 25-gene panel classification AUC (if labels available)
3. Matched blood–imaging–MVD outcome (if n≥10 matched): logistic regression MVD ~ central_axis + nerve ROI

## Excluded / non-claims
- Independent k=2 TN subtype replication is **not** a primary endpoint (underpowered in discovery)
- GSE177034/GSE124272/GSE150408 are convergent NP cohorts, not TN validation

## Frozen assets (do not modify after lock)
- `results/endotype_framework/portable_endotype_signatures.json`
- `results/high_impact_integration/ion_cci_human_ortholog_mapping.csv`
- Module direction signs: peripheral=up; central/ntrk=down

## Analysis code
`scripts/run_endotype_validation_tiers.py` + `scripts/run_natcomm_extension_analyses.py`
