"""
Endotype framework analyses for high-impact TN publication.

1. Portable endotype signatures (gene panels + module scores)
2. Population-level multimodal bridge (blood endotypes ↔ imaging endotypes)
3. MVD outcome by central vs peripheral imaging endotype (+ nerve-region metrics)
4. Human TG atlas as mechanistic hub (blood DE → ganglion cell programs)
5. Clinical utility / treatment mapping (non-circular)
6. Therapeutic hypotheses (discussion-ready)
7. Five main figures + summary tables
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.extract_nerve_region_imaging import extract_cohort_nerve_metrics

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

OUT = project_root / "results" / "endotype_framework"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Canonical TG cell types from GSE197289 author annotations
TG_CELL_MAP = {
    "PEP": "Neurons",
    "NF": "Neurons",
    "NP": "Neurons",
    "SST": "Neurons",
    "TRPM8": "Neurons",
    "cLTMR": "Neurons",
    "Schwann": "Schwann_cells",
    "Satglia": "Satellite_glial_cells",
    "Immune": "Immune_cells",
    "Fibroblast": "Fibroblasts",
    "Vascular": "Endothelial_cells",
}


def _load_blood_expression() -> pd.DataFrame:
    expr = pd.read_csv(project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv", index_col=0)
    expr.index = expr.index.astype(str).str.upper()
    return expr


def analysis_1_portable_signatures() -> dict:
    """Define portable endotype gene panels and module score templates."""
    logger.info("ANALYSIS 1: Portable endotype signatures")

    subtypes = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")
    de0 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_0.csv")
    de1 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_1.csv")
    modules = pd.read_csv(project_root / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")
    module_scores = pd.read_csv(project_root / "results" / "high_impact_integration" / "human_blood_module_scores.csv")

    lfc, padj = "log2fc", "p_adj"

    def _panel(de_df, subtype, direction, top_n=25):
        sig = de_df[(de_df[padj] < 0.05) & (de_df[lfc].abs() > 0.5)].copy()
        if direction == "up":
            sig = sig[sig[lfc] > 0]
        else:
            sig = sig[sig[lfc] < 0]
        sig = sig.nlargest(top_n, sig[lfc].abs().name if "abs_log2fc" not in sig.columns else "abs_log2fc")
        return sig[["gene", lfc, padj]].assign(subtype=subtype, direction=direction)

    # Subtype 0 = peripheral endotype (ECM/injury up in IoN-CCI)
    peripheral_panel = _panel(de0, 0, "up", 25)
    # Subtype 1 = central endotype
    central_panel = _panel(de1, 1, "up", 25)

    # Module-based portable scores
    key_modules = ["peripheral_ecm_schwann", "central_synaptic", "ntrk_signaling"]
    module_genes = {}
    for mod in key_modules:
        g = modules.loc[modules["module"] == mod, "human_symbol"].dropna().astype(str).str.upper().unique().tolist()
        module_genes[mod] = g

    signature = {
        "version": "1.0",
        "cohort": "GSE186505",
        "n_tn_discovery": int(len(subtypes)),
        "endotypes": {
            "peripheral_subtype_0": {
                "blood_subtype": 0,
                "n_samples": int((subtypes["subtype"] == 0).sum()),
                "description": "Peripheral ECM/Schwann injury-remodeling endotype",
                "primary_modules": ["peripheral_ecm_schwann"],
                "gene_panel": peripheral_panel["gene"].tolist(),
                "portable_score_formula": "mean(z-score(g)) for module peripheral_ecm_schwann orthologs",
            },
            "central_subtype_1": {
                "blood_subtype": 1,
                "n_samples": int((subtypes["subtype"] == 1).sum()),
                "description": "Central synaptic/NTRK dysregulation endotype",
                "primary_modules": ["central_synaptic", "ntrk_signaling"],
                "gene_panel": central_panel["gene"].tolist(),
                "portable_score_formula": "mean(-z(g)) for central_synaptic + ntrk_signaling orthologs",
            },
        },
        "module_ortholog_genes": module_genes,
    }

    (OUT / "portable_endotype_signatures.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    pd.concat([peripheral_panel, central_panel], ignore_index=True).to_csv(OUT / "endotype_gene_panels.csv", index=False)

    # Wide module scores for reference cohort
    wide = module_scores.pivot(index="sample_id", columns="module", values="module_score")
    wide = wide.merge(subtypes[["sample_id", "subtype"]], left_index=True, right_on="sample_id")
    wide.to_csv(OUT / "reference_endotype_module_scores.csv", index=False)

    logger.info(f"Portable signatures: {len(peripheral_panel)} peripheral + {len(central_panel)} central genes")
    return signature


def analysis_2_population_bridge() -> pd.DataFrame:
    """Population-level bridge: blood endotype axis ↔ imaging endotypes."""
    logger.info("ANALYSIS 2: Population-level multimodal bridge")

    module_assoc = pd.read_csv(project_root / "results" / "high_impact_integration" / "module_subtype_association_tests.csv")
    img = pd.read_csv(project_root / "results" / "aim2" / "real_processing" / "imaging_subtypes_all_subjects.csv")
    img_pre = img[~img["subject_id"].astype(str).str.endswith("fu")].copy()

    centroids = img_pre.groupby("imaging_subtype").agg({
        "fa_mean": "mean", "md_mean": "mean", "t1_mean": "mean", "t2_mean": "mean", "fmri_mean": "mean",
    }).reset_index()

    # Enrich centroids with atlas DTI ROI means when available
    nerve_path = OUT / "nerve_region_imaging_metrics.csv"
    if nerve_path.exists():
        nerve = pd.read_csv(nerve_path)
        atlas_cols = [
            c for c in nerve.columns
            if c.startswith(("fa_", "md_", "rd_", "ad_")) and c.endswith("_mean") and "legacy" not in c
        ]
        for col in atlas_cols:
            if col not in nerve.columns or nerve[col].notna().sum() < 5:
                continue
            merged_n = img_pre.merge(nerve[["subject_id", col]], on="subject_id", how="left")
            if col not in merged_n.columns:
                continue
            by_st = merged_n.groupby("imaging_subtype", dropna=False)[col].mean()
            centroids[col] = centroids["imaging_subtype"].map(by_st)

    # Endotype axis scores from blood modules
    blood_peripheral = module_assoc.loc[module_assoc["module"] == "peripheral_ecm_schwann", "delta_1_minus_0"].iloc[0]
    blood_central_syn = module_assoc.loc[module_assoc["module"] == "central_synaptic", "delta_1_minus_0"].iloc[0]
    blood_central_ntrk = module_assoc.loc[module_assoc["module"] == "ntrk_signaling", "delta_1_minus_0"].iloc[0]

    rows = []
    # DTI-first priors (atlas nerve-adjacent > legacy T1/T2 intensity)
    priors = [
        ("peripheral_ecm_schwann", "fa_mean", "imaging_subtype_0", "peripheral/compressive", "positive"),
        ("peripheral_ecm_schwann", "fa_rez_bilateral_mean", "imaging_subtype_0", "peripheral/compressive", "positive"),
        ("peripheral_ecm_schwann", "md_rez_bilateral_mean", "imaging_subtype_0", "peripheral/compressive", "negative"),
        ("central_synaptic", "fmri_mean", "imaging_subtype_2", "central/sensitized", "positive"),
        ("central_synaptic", "md_cisternal_bilateral_mean", "imaging_subtype_2", "central/sensitized", "positive"),
        ("ntrk_signaling", "fmri_mean", "imaging_subtype_2", "central/sensitized", "positive"),
        ("ntrk_signaling", "fa_cisternal_bilateral_mean", "imaging_subtype_2", "central/sensitized", "negative"),
    ]

    for mod, feat, img_endotype, endotype_label, expected in priors:
        if feat not in centroids.columns:
            continue
        rho, p = stats.spearmanr(centroids["imaging_subtype"], centroids[feat]) if len(centroids) >= 3 else (np.nan, np.nan)
        blood_delta = module_assoc.loc[module_assoc["module"] == mod, "delta_1_minus_0"]
        blood_delta = blood_delta.iloc[0] if len(blood_delta) else np.nan
        rows.append({
            "blood_module": mod,
            "blood_delta_subtype1_minus_0": blood_delta,
            "imaging_feature": feat,
            "imaging_endotype_prior": img_endotype,
            "endotype_label": endotype_label,
            "spearman_rho_imaging_subtype": rho,
            "spearman_p": p,
            "imaging_subtype_0_mean": centroids.loc[centroids.imaging_subtype == 0, feat].values[0] if 0 in centroids.imaging_subtype.values else np.nan,
            "imaging_subtype_1_mean": centroids.loc[centroids.imaging_subtype == 1, feat].values[0] if 1 in centroids.imaging_subtype.values else np.nan,
            "imaging_subtype_2_mean": centroids.loc[centroids.imaging_subtype == 2, feat].values[0] if 2 in centroids.imaging_subtype.values else np.nan,
            "concordant": (np.sign(rho) == (1 if expected == "positive" else -1)) if pd.notna(rho) and rho != 0 else np.nan,
        })

    bridge = pd.DataFrame(rows)
    bridge.to_csv(OUT / "population_multimodal_bridge.csv", index=False)

    # Assign imaging endotype labels
    # DTI-first central imaging score (replace T2 intensity)
    fa_z = stats.zscore(img_pre["fa_mean"].fillna(img_pre["fa_mean"].median()), nan_policy="omit")
    md_z = stats.zscore(img_pre["md_mean"].fillna(img_pre["md_mean"].median()), nan_policy="omit")
    fmri_z = stats.zscore(img_pre["fmri_mean"].fillna(img_pre["fmri_mean"].median()), nan_policy="omit")
    img_pre["central_imaging_score"] = fmri_z - fa_z + md_z
    img_pre["imaging_endotype_label"] = np.where(
        img_pre["imaging_subtype"] == 2, "central",
        np.where(img_pre["imaging_subtype"] == 0, "peripheral", "intermediate"),
    )
    img_pre.to_csv(OUT / "imaging_endotype_labels.csv", index=False)

    # Figure 5: DTI-first centroids (de-emphasize T1/T2 intensity)
    dti_feat = [c for c in centroids.columns if c in ("fa_mean", "md_mean", "fmri_mean")]
    nerve_path = OUT / "nerve_region_imaging_metrics.csv"
    if nerve_path.exists():
        nerve = pd.read_csv(nerve_path)
        atlas_cols = [
            c for c in nerve.columns
            if c.startswith(("fa_", "md_", "rd_", "ad_")) and c.endswith("_mean") and "legacy" not in c
        ]
        for col in atlas_cols[:4]:
            if col not in nerve.columns or nerve[col].notna().sum() < 5:
                continue
            merged_n = img_pre.merge(nerve[["subject_id", col]], on="subject_id", how="left")
            if col not in merged_n.columns:
                continue
            by_st = merged_n.groupby("imaging_subtype", dropna=False)[col].mean()
            centroids[col] = centroids["imaging_subtype"].map(by_st)
            if col not in dti_feat:
                dti_feat.append(col)
    dti_feat = [c for c in dti_feat if c in centroids.columns]
    heat_data = centroids.set_index("imaging_subtype")[dti_feat]
    heat_z = heat_data.apply(stats.zscore, axis=0)
    fig, ax = plt.subplots(figsize=(max(6, len(dti_feat) * 1.2), 4))
    sns.heatmap(heat_z.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Imaging centroids by subtype (DTI-first features, z-scored)")
    ax.set_xlabel("Imaging subtype")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_imaging_endotype_centroids.png", dpi=300, bbox_inches="tight")
    plt.close()

    n_con = int(bridge["concordant"].sum()) if "concordant" in bridge.columns else 0
    logger.info(f"Population bridge: {n_con}/{len(bridge)} concordant priors")
    return bridge


def analysis_3_mvd_outcome_by_endotype(nerve_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prospective-style: central imaging endotype vs MVD outcome (+ nerve metrics)."""
    logger.info("ANALYSIS 3: MVD outcome by imaging endotype")

    part = pd.read_csv(project_root / "data" / "external" / "openneuro" / "ds005713" / "participants.tsv", sep="\t")
    part = part.rename(columns={"BIDS_ID": "subject_id"})
    img = pd.read_csv(OUT / "imaging_endotype_labels.csv")

    mvd = part.merge(img, on="subject_id", how="inner")
    mvd = mvd[(mvd["Surgery_type"] == "MVD") & (mvd["Surgery_outcome"].isin(["Pos", "Neg"]))].copy()
    mvd["outcome_binary"] = (mvd["Surgery_outcome"] == "Pos").astype(int)
    mvd["Sindou_grade"] = pd.to_numeric(mvd["Sindou_grade"], errors="coerce")

    if nerve_df is not None and len(nerve_df):
        mvd = mvd.merge(nerve_df, on="subject_id", how="left")

    # Outcome by imaging endotype label
    outcome_rows = []
    for label in ["central", "peripheral", "intermediate"]:
        sub = mvd[mvd["imaging_endotype_label"] == label]
        if len(sub) == 0:
            continue
        n_pos = (sub["Surgery_outcome"] == "Pos").sum()
        outcome_rows.append({
            "imaging_endotype": label,
            "n_mvd": len(sub),
            "n_responders": int(n_pos),
            "n_nonresponders": int(len(sub) - n_pos),
            "response_rate": n_pos / len(sub),
        })
    outcome_by_endotype = pd.DataFrame(outcome_rows)
    outcome_by_endotype.to_csv(OUT / "mvd_outcome_by_imaging_endotype.csv", index=False)

    # Fisher: central vs non-central response
    central = mvd[mvd["imaging_endotype_label"] == "central"]
    non_central = mvd[mvd["imaging_endotype_label"].isin(["peripheral", "intermediate"])]
    fisher_p = np.nan
    if len(central) >= 2 and len(non_central) >= 2:
        table = [
            [(central["Surgery_outcome"] == "Pos").sum(), (central["Surgery_outcome"] == "Neg").sum()],
            [(non_central["Surgery_outcome"] == "Pos").sum(), (non_central["Surgery_outcome"] == "Neg").sum()],
        ]
        _, fisher_p = stats.fisher_exact(table)

    # Non-circular logistic: clinical + nerve-region + central score (exclude imaging_subtype label)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_cols = ["age", "Sindou_grade", "Pain_severity (average score)", "Disease_duration (years)", "central_imaging_score"]
    # DTI-first nerve-adjacent features (atlas ROIs preferred)
    nerve_cols = [
        c for c in mvd.columns
        if any(x in c for x in ["fa_rez", "fa_cisternal", "md_rez", "md_cisternal", "rd_rez", "ad_rez"])
        and c.endswith("_mean") and "legacy" not in c
    ]
    if not nerve_cols:
        nerve_cols = [c for c in mvd.columns if any(x in c for x in ["fa_mean", "md_mean"]) and c.endswith("_mean")]
    feature_cols += [c for c in nerve_cols if mvd[c].notna().sum() >= 8]

    X = mvd[feature_cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep]
    y = mvd["outcome_binary"].values

    model_results = {"fisher_central_vs_other_p": fisher_p, "n_mvd": len(mvd)}
    if len(np.unique(y)) == 2 and len(mvd) >= 10:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ])
        cv = StratifiedKFold(n_splits=min(4, min(np.bincount(y))), shuffle=True, random_state=42)
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        model_results["cv_auc_non_circular"] = float(roc_auc_score(y, proba))
        model_results["features_used"] = keep
        pd.DataFrame([model_results]).to_csv(OUT / "mvd_non_circular_model.csv", index=False)

    mvd.to_csv(OUT / "mvd_cohort_with_endotypes.csv", index=False)

    # Bar plot: response rate by endotype
    if len(outcome_by_endotype):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=outcome_by_endotype, x="imaging_endotype", y="response_rate", ax=ax, palette="Set2")
        ax.set_ylabel("MVD response rate")
        ax.set_xlabel("Imaging endotype")
        ax.set_title(f"MVD outcome by imaging endotype (Fisher p={fisher_p:.3f})" if pd.notna(fisher_p) else "MVD outcome by imaging endotype")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIG / "fig4_mvd_by_imaging_endotype.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"MVD endotype analysis: n={len(mvd)}, Fisher p={fisher_p}")
    return outcome_by_endotype


def analysis_4_human_tg_hub() -> pd.DataFrame:
    """Map blood DE genes to human TG cell-type programs."""
    logger.info("ANALYSIS 4: Human TG atlas mechanistic hub")

    ref_path = project_root / "results" / "aim1_enhanced" / "human_tg_integration" / "human_tg_reference_signatures.csv"
    sig = pd.read_csv(ref_path, index_col=0)
    sig.index = sig.index.astype(str).str.upper()

    # Collapse Leiden/unknown columns to canonical TG cell types (GSE197289 author taxonomy)
    canonical_map = {
        "Neurons": ["Neurons", "Unknown_3", "Unknown_14"],
        "Schwann_cells": ["Schwann_cells", "Unknown_8"],
        "Satellite_glial_cells": ["Unknown_11"],
        "Immune_cells": ["Unknown_7"],
        "Fibroblasts": ["Fibroblasts", "Unknown_12", "Unknown_2"],
        "Endothelial_cells": ["Endothelial_cells"],
    }
    canon_expr = {}
    for ct, cols in canonical_map.items():
        present = [c for c in cols if c in sig.columns]
        if present:
            canon_expr[ct] = sig[present].max(axis=1)
    sig_df = pd.DataFrame(canon_expr)

    # Author cell-type proportions for context
    meta = pd.read_csv(project_root / "data" / "external" / "geo" / "GSE197289" / "GSE197289" / "human_meta.csv")
    meta["canonical_cell"] = meta["cellID"].map(TG_CELL_MAP).fillna("Other")
    meta_props = meta["canonical_cell"].value_counts(normalize=True).rename("tg_cohort_fraction")
    meta_props.to_csv(OUT / "human_tg_celltype_composition.csv")

    # Blood DE genes
    de0 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_0.csv")
    de1 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_1.csv")
    de0["gene_u"] = de0["gene"].astype(str).str.upper()
    de1["gene_u"] = de1["gene"].astype(str).str.upper()

    def _map_de(de_df, subtype_name):
        sig = de_df[(de_df["p_adj"] < 0.05) & (de_df["log2fc"].abs() > 0.5)].copy()
        sig = sig.nlargest(100, "abs_log2fc" if "abs_log2fc" in sig.columns else sig["log2fc"].abs().name)
        rows = []
        for _, r in sig.iterrows():
            g = r["gene_u"]
            if g not in sig_df.index:
                continue
            expr = sig_df.loc[g]
            top_ct = expr.idxmax()
            rows.append({
                "gene": g,
                "blood_subtype": subtype_name,
                "log2fc_blood": r["log2fc"],
                "top_tg_cell_type": top_ct,
                "tg_expression": float(expr[top_ct]),
            })
        return pd.DataFrame(rows)

    mapped = pd.concat([_map_de(de0, "peripheral_0"), _map_de(de1, "central_1")], ignore_index=True)
    mapped.to_csv(OUT / "blood_de_to_tg_celltype_mapping.csv", index=False)

    # Enrichment: which cell types dominate each endotype DE set
    enrich_rows = []
    for st in ["peripheral_0", "central_1"]:
        sub = mapped[mapped["blood_subtype"] == st]
        if len(sub) == 0:
            continue
        counts = sub["top_tg_cell_type"].value_counts(normalize=True)
        for ct, frac in counts.items():
            enrich_rows.append({"blood_endotype": st, "tg_cell_type": ct, "fraction_top_celltype": frac, "n_genes": len(sub)})

    enrich = pd.DataFrame(enrich_rows)
    enrich.to_csv(OUT / "blood_endotype_tg_celltype_enrichment.csv", index=False)

    # Figure 3: stacked bar of cell type enrichment
    if len(enrich):
        pivot = enrich.pivot(index="blood_endotype", columns="tg_cell_type", values="fraction_top_celltype").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
        ax.set_title("Blood endotype DE genes mapped to human TG cell-type programs")
        ax.set_ylabel("Fraction of DE genes with highest TG expression")
        ax.set_xlabel("Blood endotype")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / "fig3_blood_de_tg_celltypes.png", dpi=300, bbox_inches="tight")
        plt.close()

    sig_df.to_csv(OUT / "human_tg_celltype_signatures_computed.csv")
    logger.info(f"TG hub: mapped {len(mapped)} DE genes to cell types")
    return enrich


def analysis_5_clinical_utility() -> pd.DataFrame:
    """Treatment mapping hypothesis table (non-circular, clinically motivated)."""
    logger.info("ANALYSIS 5: Clinical utility / treatment mapping")

    rows = [
        {
            "blood_endotype": "Subtype 0 (peripheral)",
            "imaging_endotype_prior": "Imaging subtype 0 (peripheral/compressive)",
            "mechanism": "ECM/Schwann injury-remodeling; IoN-CCI TG-up",
            "hypothesized_phenotype": "Nerve compression-dominant TN",
            "recommended_intervention": "MVD / surgical decompression",
            "rationale": "Peripheral structural pathology; higher pre-op FA/T1 load in imaging subtype 0",
            "evidence_level": "Hypothesis (population-level concordance)",
        },
        {
            "blood_endotype": "Subtype 1 (central)",
            "imaging_endotype_prior": "Imaging subtype 2 (central/sensitized)",
            "mechanism": "Synaptic/NTRK dysregulation; IoN-CCI TG/Sp5C-down",
            "hypothesized_phenotype": "Central sensitization-dominant TN",
            "recommended_intervention": "Neuromodulation + pharmacologic (gabapentinoids, SNRIs)",
            "rationale": "Central module enrichment; highest fMRI/T2 in imaging subtype 2; may predict lower MVD response",
            "evidence_level": "Hypothesis (population-level concordance)",
        },
    ]
    utility = pd.DataFrame(rows)
    utility.to_csv(OUT / "clinical_utility_treatment_mapping.csv", index=False)

    # Therapeutic hypotheses (discussion box — not primary claims)
    drugs = pd.read_csv(project_root / "results" / "high_impact_integration" / "drug_repurposing_ntrk_synaptic_shortlist.csv")
    top = drugs.sort_values("priority_score", ascending=False).head(8)
    top = top.assign(
        claim_strength="Exploratory hypothesis",
        literature_support="Central sensitization / neuropathic pain pharmacology; NTRK biology in chronic pain",
    )
    top.to_csv(OUT / "therapeutic_hypotheses_discussion.csv", index=False)

    logger.info("Clinical utility table saved")
    return utility


def generate_main_figures():
    """Assemble 5 main publication figures."""
    logger.info("Generating 5 main figures")

    subtypes = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")
    module_scores = pd.read_csv(project_root / "results" / "high_impact_integration" / "human_blood_module_scores.csv")
    de1 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_1.csv")

    # Fig 1: Concept schematic (matplotlib diagram)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "GSE186505\nBlood RNA-seq\nn=10 TN, k=2"),
        (0.35, 0.55, "GSE197289\nHuman TG snRNA-seq\n38k cells"),
        (0.65, 0.55, "IoN-CCI\nRodent TG/Sp5C\nMechanism modules"),
        (0.05, 0.15, "OpenNeuro ds005713\nImaging + MVD outcome\nn=263 processed"),
        (0.45, 0.15, "Population bridge\nEndotype signatures"),
        (0.75, 0.15, "Clinical utility\nTreatment mapping"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.28, fill=True, facecolor="#e8f4fc", edgecolor="#333"))
        ax.text(x + 0.11, y + 0.14, txt, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.45, 0.4), xytext=(0.16, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.45, 0.4), xytext=(0.46, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.45, 0.4), xytext=(0.76, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.56, 0.28), xytext=(0.16, 0.15), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Fig 1. Cross-species endotype framework for trigeminal neuralgia", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_concept_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 2: Module heatmap + subtype distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    subtypes["subtype"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("TN molecular subtypes (k=2)")
    axes[0].set_xlabel("Subtype")
    key = module_scores[module_scores["module"].isin(["peripheral_ecm_schwann", "central_synaptic", "ntrk_signaling"])]
    sns.boxplot(data=key, x="module", y="module_score", hue="subtype", ax=axes[1])
    axes[1].set_title("IoN-CCI module scores by subtype")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_blood_endotypes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3 already from TG analysis; copy cross-species if module heatmap exists
    hi_fig = project_root / "results" / "high_impact_integration" / "figures" / "module_scores_by_subtype.png"
    if hi_fig.exists():
        import shutil
        shutil.copy(hi_fig, FIG / "fig3_cross_species_modules.png")

    # Fig 4 from MVD analysis
    mvd_roc = project_root / "results" / "high_impact_integration" / "figures" / "mvd_outcome_roc.png"
    if mvd_roc.exists():
        import shutil
        shutil.copy(mvd_roc, FIG / "fig4_mvd_outcome_roc.png")

    # Fig 5 integrated axis
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.1, 0.7, "PERIPHERAL ENDOTYPE (Subtype 0)", fontsize=12, weight="bold", color="#1f77b4")
    ax.text(0.1, 0.55, "• ECM/Schwann modules ↑\n• IoN-CCI TG-up\n• Imaging subtype 0\n→ MVD candidate", fontsize=10)
    ax.text(0.55, 0.7, "CENTRAL ENDOTYPE (Subtype 1)", fontsize=12, weight="bold", color="#ff7f0e")
    ax.text(0.55, 0.55, "• Synaptic/NTRK modules ↑\n• IoN-CCI TG/Sp5C-down\n• Imaging subtype 2\n→ Neuromodulation/pharmacologic", fontsize=10)
    ax.annotate("", xy=(0.5, 0.35), xytext=(0.25, 0.45), arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(0.35, 0.25, "Population-level\nconcordance", ha="center", fontsize=9, style="italic")
    ax.set_title("Fig 5. Integrated peripheral–central TN endotype model")
    plt.tight_layout()
    plt.savefig(FIG / "fig5_integrated_endotype_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_summary():
    lines = [
        "# Endotype Framework Analysis Summary",
        "",
        "## Deliverables",
        "- Portable endotype signatures (JSON + gene panels)",
        "- Population multimodal bridge table",
        "- MVD outcome by imaging endotype",
        "- Human TG cell-type mapping for blood DE",
        "- Nerve-adjacent DTI metrics (atlas-refined ROIs; primary imaging path)",
        "- Legacy T1/T2 intensity ROIs de-emphasized (QC confounds)",
        "- SEVB-Net / manual CN V pilot → supplementary / future work (see CLAIM_TABLE_LOCKED.md)",
        "- Clinical utility + therapeutic hypotheses (discussion)",
        "- Five main figures in `figures/`",
        "",
        "## Publication framing",
        "Framework and discovery paper: peripheral vs central TN endotypes with cross-species anchoring.",
        "Explicit n=10 blood discovery; cohort-independent integration; validation roadmap required.",
    ]
    (OUT / "ENDOTYPE_FRAMEWORK_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    logger.info("=" * 70)
    logger.info("ENDOTYPE FRAMEWORK ANALYSES")
    logger.info("=" * 70)

    analysis_1_portable_signatures()

    # DTI nerve-adjacent metrics BEFORE population bridge (DTI-first features)
    part = pd.read_csv(project_root / "data" / "external" / "openneuro" / "ds005713" / "participants.tsv", sep="\t")
    mvd_ids = part.loc[part["Surgery_type"] == "MVD", "BIDS_ID"].astype(str).tolist()
    openeuro = project_root / "data" / "external" / "openneuro" / "ds005713"
    mvd_baseline = [s for s in mvd_ids if not str(s).endswith("fu")]
    nerve_df = extract_cohort_nerve_metrics(
        mvd_baseline,
        openeuro,
        OUT / "nerve_region_imaging_metrics.csv",
        compute_dti=True,
        roi_method="atlas",
        include_t1_t2=False,
    )
    dti_pre = project_root / "results" / "aim2" / "real_processing" / "dti_metrics_all_subjects.csv"
    if dti_pre.exists():
        dti = pd.read_csv(dti_pre)
        merge_cols = [c for c in ["subject_id", "fa_mean", "md_mean", "rd_mean", "ad_mean"] if c in dti.columns]
        nerve_df = nerve_df.merge(dti[merge_cols], on="subject_id", how="left", suffixes=("", "_wb"))
        nerve_df.to_csv(OUT / "nerve_region_imaging_metrics.csv", index=False)

    analysis_2_population_bridge()

    analysis_3_mvd_outcome_by_endotype(nerve_df)
    analysis_4_human_tg_hub()
    analysis_5_clinical_utility()
    generate_main_figures()
    write_summary()

    logger.info(f"Complete → {OUT}")
    return True


if __name__ == "__main__":
    main()
