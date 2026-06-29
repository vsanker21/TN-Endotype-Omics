"""
High-impact cross-dataset analyses:
1. IoN-CCI module scoring + ortholog mapping → human blood subtypes (k=2, n=10)
2. OpenNeuro MVD outcome model (pre-op imaging + clinical covariates)
3. Pre→post neuroplasticity (fu sessions)
4. Mechanism–imaging concordance table
5. Drug repurposing shortlist (NTRK/synaptic modules + human DE direction)
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

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

OUT = project_root / "results" / "high_impact_integration"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Curated IoN-CCI mechanism modules (literature-anchored + DE/pathway derived)
# ---------------------------------------------------------------------------
MODULE_DEFINITIONS = {
    "peripheral_ecm_schwann": {
        "description": "Peripheral injury/remodeling: ECM, Schwann cell myelination (IoN-CCI up in TG)",
        "pathway_terms": [
            "Schwann Cell Myelination",
            "Extracellular Matrix Organization",
            "Collagen Formation",
            "Collagen Chain Trimerization",
        ],
        "direction_ion_cci": "up",
        "extra_genes": ["Mpz", "Prx", "Egr2", "Lama2", "Lamb1", "Col1a1", "Mmp13", "Dcn"],
    },
    "central_synaptic": {
        "description": "Central synaptic/neuronal dysfunction (IoN-CCI down in TG/Sp5C)",
        "pathway_terms": [
            "Neuronal System",
            "Transmission Across Chemical Synapses",
            "Neurotransmitter Release Cycle",
            "GABA Synthesis",
        ],
        "direction_ion_cci": "down",
        "extra_genes": ["Snap25", "Stxbp1", "Syn1", "Vamp2", "Syt1", "Cplx1", "Rab3a"],
    },
    "ntrk_signaling": {
        "description": "Neurotrophin/NTRK-TRKA signaling (IoN-CCI down)",
        "pathway_terms": ["Signaling By NTRKs", "Signaling By NTRK1"],
        "direction_ion_cci": "down",
        "extra_genes": ["Ntrk1", "Ntrk2", "Grb2", "Irs2", "Bdnf", "Ngfr"],
    },
}

# Mechanistic priors for imaging concordance
MECHANISM_IMAGING_PRIORS = [
    ("peripheral_ecm_schwann", "fa_mean", "positive", "Peripheral injury → local microstructural DTI changes"),
    ("peripheral_ecm_schwann", "t1_mean", "positive", "Structural/nerve region intensity"),
    ("peripheral_ecm_schwann", "compression_index", "positive", "Compressive subtype hypothesis"),
    ("central_synaptic", "fmri_mean", "positive", "Centralized pain → elevated BOLD/network load"),
    ("central_synaptic", "modularity", "negative", "Network dysregulation reduces modularity"),
    ("central_synaptic", "global_efficiency", "negative", "Central subtype network disruption"),
    ("ntrk_signaling", "fmri_mean", "positive", "NTRK disruption associated with central sensitization"),
    ("ntrk_signaling", "t2_mean", "positive", "Inflammatory/edema proxy on T2"),
]

# Curated druggable targets for NTRK/synaptic modules
CURATED_DRUGS = [
    {"drug": "Entrectinib", "targets": ["NTRK1", "NTRK2", "NTRK3"], "mechanism": "TRK inhibitor", "ion_cci_logic": "down", "human_logic": "down_in_subtype_needs_activation"},
    {"drug": "Larotrectinib", "targets": ["NTRK1", "NTRK2", "NTRK3"], "mechanism": "TRK inhibitor", "ion_cci_logic": "down", "human_logic": "down_in_subtype_needs_activation"},
    {"drug": "Gabapentin", "targets": ["CACNA2D1", "CACNA2D2"], "mechanism": "Calcium channel modulator (synaptic)", "ion_cci_logic": "down", "human_logic": "symptomatic"},
    {"drug": "Pregabalin", "targets": ["CACNA2D1"], "mechanism": "Calcium channel modulator", "ion_cci_logic": "down", "human_logic": "symptomatic"},
    {"drug": "Baclofen", "targets": ["GABBR1", "GABBR2"], "mechanism": "GABA-B agonist", "ion_cci_logic": "down", "human_logic": "down_in_subtype_needs_activation"},
    {"drug": "Duloxetine", "targets": ["SLC6A2", "SLC6A4"], "mechanism": "SNRI", "ion_cci_logic": "down", "human_logic": "symptomatic"},
    {"drug": "Milnacipran", "targets": ["SLC6A2", "SLC6A4"], "mechanism": "SNRI", "ion_cci_logic": "down", "human_logic": "symptomatic"},
    {"drug": "Memantine", "targets": ["GRIN1", "GRIN2B"], "mechanism": "NMDA antagonist", "ion_cci_logic": "down", "human_logic": "symptomatic"},
    {"drug": "Acetylcholinesterase inhibitors (Donepezil)", "targets": ["ACHE"], "mechanism": "AChE inhibitor", "ion_cci_logic": "down", "human_logic": "down_in_subtype_needs_activation"},
    {"drug": "Semaglutide (exploratory)", "targets": ["GLP1R"], "mechanism": "Metabolic-neuroimmune modulator", "ion_cci_logic": "down", "human_logic": "exploratory"},
]


def _parse_pathway_genes(pathway_df: pd.DataFrame, terms: list[str]) -> set[str]:
    genes = set()
    for _, row in pathway_df.iterrows():
        term = str(row.get("Term", ""))
        if any(t.lower() in term.lower() for t in terms):
            raw = str(row.get("Genes", ""))
            for g in raw.split(";"):
                g = g.strip()
                if g:
                    genes.add(g)
    return genes


def _load_pathway_files() -> dict[str, pd.DataFrame]:
    base = project_root / "results" / "aim3" / "ion_cci" / "de_review"
    files = {
        "tg_up": base / "tg_upregulated_pathways.csv",
        "tg_down": base / "tg_downregulated_pathways.csv",
        "sp5c_up": base / "sp5c_upregulated_pathways.csv",
        "sp5c_down": base / "sp5c_downregulated_pathways.csv",
    }
    out = {}
    for k, p in files.items():
        out[k] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def build_mouse_module_gene_sets(pathways: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Assemble mouse gene sets per module from pathways + DE support."""
    de_tg = pd.read_csv(project_root / "results" / "aim3" / "ion_cci" / "de_analysis" / "de_ion_cci_vs_sham_tg_with_symbols.csv")
    de_sp5c = pd.read_csv(project_root / "results" / "aim3" / "ion_cci" / "de_analysis" / "de_ion_cci_vs_sham_sp5c_with_symbols.csv")

    modules = {}
    for name, meta in MODULE_DEFINITIONS.items():
        genes = _parse_pathway_genes(pathways["tg_up" if meta["direction_ion_cci"] == "up" else "tg_down"], meta["pathway_terms"])
        genes.update(g.upper() if g.islower() else g for g in meta["extra_genes"])
        genes = {g for g in genes if g and g != "nan"}

        de = de_tg if meta["direction_ion_cci"] == "up" else de_tg
        sig = de[(de["padj"] < 0.05) & (de["log2FoldChange"].abs() > 0.5)]
        if meta["direction_ion_cci"] == "up":
            sig = sig[sig["log2FoldChange"] > 0]
        else:
            sig = sig[sig["log2FoldChange"] < 0]
        sig = sig.assign(abs_lfc=sig["log2FoldChange"].abs())
        top_de = set(sig.nlargest(50, "abs_lfc")["gene_symbol"].dropna().astype(str))
        genes |= top_de

        modules[name] = {
            "mouse_genes": sorted(genes),
            "direction_ion_cci": meta["direction_ion_cci"],
            "description": meta["description"],
            "n_genes": len(genes),
        }

    # Tissue-specific modules: TG vs Sp5C IoN-CCI DE divergence
    tg_sig = de_tg[(de_tg["padj"] < 0.05) & (de_tg["log2FoldChange"].abs() > 1)]
    sp_sig = de_sp5c[(de_sp5c["padj"] < 0.05) & (de_sp5c["log2FoldChange"].abs() > 1)]
    tg_ids = set(tg_sig["gene_symbol"].astype(str))
    sp_ids = set(sp_sig["gene_symbol"].astype(str))
    modules["tg_enriched"] = {
        "mouse_genes": sorted(list(tg_ids - sp_ids))[:100] if len(tg_ids - sp_ids) > 0 else sorted(list(tg_ids))[:100],
        "direction_ion_cci": "mixed",
        "description": "TG-enriched IoN-CCI response (vs Sp5C)",
        "n_genes": min(100, len(tg_ids - sp_ids) or len(tg_ids)),
    }
    modules["sp5c_enriched"] = {
        "mouse_genes": sorted(list(sp_ids - tg_ids))[:100] if len(sp_ids - tg_ids) > 0 else sorted(list(sp_ids))[:100],
        "direction_ion_cci": "mixed",
        "description": "Sp5C-enriched IoN-CCI response (vs TG)",
        "n_genes": min(100, len(sp_ids - tg_ids) or len(sp_ids)),
    }
    return modules


def map_mouse_to_human_orthologs(mouse_genes: list[str]) -> pd.DataFrame:
    """Map mouse symbols to human via mygene homologene + project Ensembl table + symbol fallback."""
    mouse_genes = [g for g in mouse_genes if g and str(g) != "nan"]
    mapping_rows = []
    seen_mouse = set()

    ens_file = project_root / "results" / "aim3" / "ion_cci" / "de_analysis" / "ensembl_to_symbol_mapping.csv"
    ens_map = {}
    if ens_file.exists():
        ens_df = pd.read_csv(ens_file)
        ens_map = dict(zip(ens_df["gene_symbol"].astype(str).str.upper(), ens_df["gene_symbol"].astype(str)))

    try:
        import mygene

        mg = mygene.MyGeneInfo()
        batch_size = 1000
        for i in range(0, len(mouse_genes), batch_size):
            batch = mouse_genes[i : i + batch_size]
            res = mg.querymany(batch, scopes="symbol,alias", fields="homologene", species="mouse", returnall=True)
            for hit in res.get("out", []):
                if "notfound" in hit:
                    continue
                mouse_sym = str(hit.get("query", ""))
                human_sym = None
                hom = hit.get("homologene", {})
                if isinstance(hom, dict) and "genes" in hom:
                    for g in hom["genes"]:
                        if len(g) >= 3 and g[0] == 9606:
                            human_sym = str(g[2]).upper()
                            break
                if mouse_sym and human_sym:
                    mapping_rows.append({"mouse_symbol": mouse_sym, "human_symbol": human_sym, "method": "mygene_homologene"})
                    seen_mouse.add(mouse_sym.upper())
    except Exception as e:
        logger.warning(f"mygene ortholog mapping partial/failed ({e})")

    for g in mouse_genes:
        gu = g.upper()
        if gu in seen_mouse:
            continue
        human = ens_map.get(gu, gu)
        mapping_rows.append({"mouse_symbol": g, "human_symbol": str(human).upper(), "method": "symbol_fallback"})

    return pd.DataFrame(mapping_rows).drop_duplicates(subset=["mouse_symbol"], keep="first")


def score_modules_on_human_blood(modules: dict, ortholog_maps: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-sample module scores on human blood expression."""
    expr = pd.read_csv(project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv", index_col=0)
    expr.index = expr.index.astype(str).str.upper()
    subtypes = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")
    tn_ids = subtypes["sample_id"].tolist()
    tn_ids = [s for s in tn_ids if s in expr.columns]

    z_expr = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    score_rows = []
    detail_rows = []

    for mod_name, ortho in ortholog_maps.items():
        meta = modules[mod_name]
        human_genes = ortho["human_symbol"].str.upper().unique().tolist()
        available = [g for g in human_genes if g in z_expr.index]
        direction = meta["direction_ion_cci"]

        for sample in tn_ids:
            if not available:
                score = np.nan
            else:
                vals = z_expr.loc[available, sample]
                raw = vals.mean()
                if direction == "down":
                    score = -raw
                elif direction == "up":
                    score = raw
                else:
                    score = raw
            st = subtypes.loc[subtypes["sample_id"] == sample, "subtype"].iloc[0]
            score_rows.append({
                "sample_id": sample,
                "subtype": int(st),
                "module": mod_name,
                "module_score": score,
                "n_ortholog_genes": len(human_genes),
                "n_expressed_genes": len(available),
                "coverage": len(available) / max(len(human_genes), 1),
            })

        detail_rows.append({
            "module": mod_name,
            "description": meta["description"],
            "direction_ion_cci": direction,
            "n_mouse_genes": meta["n_genes"],
            "n_human_orthologs": len(human_genes),
            "n_expressed_in_blood": len(available),
            "example_genes": "; ".join(available[:15]),
        })

    scores = pd.DataFrame(score_rows)
    details = pd.DataFrame(detail_rows)
    return scores, details


def test_module_subtype_associations(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mod in scores["module"].unique():
        sub = scores[scores["module"] == mod]
        g0 = sub[sub["subtype"] == 0]["module_score"].dropna()
        g1 = sub[sub["subtype"] == 1]["module_score"].dropna()
        if len(g0) >= 2 and len(g1) >= 2:
            t, p = stats.ttest_ind(g0, g1, equal_var=False)
            d = (g1.mean() - g0.mean()) / np.sqrt((g0.var() + g1.var()) / 2) if (g0.var() + g1.var()) > 0 else 0
        else:
            t, p, d = np.nan, np.nan, np.nan
        rows.append({
            "module": mod,
            "mean_subtype_0": g0.mean(),
            "mean_subtype_1": g1.mean(),
            "delta_1_minus_0": g1.mean() - g0.mean(),
            "cohens_d": d,
            "welch_t": t,
            "p_value": p,
            "n_subtype_0": len(g0),
            "n_subtype_1": len(g1),
        })
    return pd.DataFrame(rows)


def permutation_test_module_subtype(scores: pd.DataFrame, n_perm: int = 9999, seed: int = 42) -> pd.DataFrame:
    """Label-shuffle permutation test for module score differences between subtypes."""
    rng = np.random.default_rng(seed)
    rows = []
    for mod in scores["module"].unique():
        sub = scores[scores["module"] == mod].dropna(subset=["module_score", "subtype"]).copy()
        g0 = sub[sub["subtype"] == 0]["module_score"]
        g1 = sub[sub["subtype"] == 1]["module_score"]
        if len(g0) < 2 or len(g1) < 2:
            continue

        obs_delta = g1.mean() - g0.mean()
        pooled_sd = np.sqrt((g0.var(ddof=1) + g1.var(ddof=1)) / 2)
        obs_d = obs_delta / pooled_sd if pooled_sd > 0 else 0.0

        vals = sub["module_score"].to_numpy()
        labels = sub["subtype"].to_numpy()
        n0 = int((labels == 0).sum())
        count = 0
        for _ in range(n_perm):
            perm_labels = rng.permutation(labels)
            p0 = vals[perm_labels == 0]
            p1 = vals[perm_labels == 1]
            if len(p0) < 2 or len(p1) < 2:
                continue
            perm_delta = p1.mean() - p0.mean()
            if abs(perm_delta) >= abs(obs_delta):
                count += 1

        rows.append({
            "module": mod,
            "observed_delta_1_minus_0": obs_delta,
            "observed_cohens_d": obs_d,
            "permutation_p_value": (count + 1) / (n_perm + 1),
            "n_permutations": n_perm,
            "n_subtype_0": len(g0),
            "n_subtype_1": len(g1),
        })
    return pd.DataFrame(rows)


def bootstrap_mvd_auc_ci(
    y: np.ndarray,
    proba: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for AUC from cross-validated out-of-fold probabilities."""
    from sklearn.metrics import roc_auc_score

    obs_auc = roc_auc_score(y, proba)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_boot, proba[idx]))

    if not boot_aucs:
        return {"observed_auc": obs_auc, "bootstrap_mean_auc": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_bootstrap": 0}

    boot_aucs = np.array(boot_aucs)
    return {
        "observed_auc": obs_auc,
        "bootstrap_mean_auc": float(boot_aucs.mean()),
        "ci_lower": float(np.percentile(boot_aucs, 2.5)),
        "ci_upper": float(np.percentile(boot_aucs, 97.5)),
        "n_bootstrap": int(len(boot_aucs)),
    }


def analysis_1_module_scoring():
    logger.info("=" * 70)
    logger.info("ANALYSIS 1: IoN-CCI module scoring + ortholog mapping")
    logger.info("=" * 70)

    pathways = _load_pathway_files()
    modules = build_mouse_module_gene_sets(pathways)

    all_ortholog = []
    ortholog_maps = {}
    for mod_name, meta in modules.items():
        ortho = map_mouse_to_human_orthologs(meta["mouse_genes"])
        ortho["module"] = mod_name
        ortholog_maps[mod_name] = ortho
        all_ortholog.append(ortho)

    ortholog_df = pd.concat(all_ortholog, ignore_index=True)
    ortholog_df.to_csv(OUT / "ion_cci_human_ortholog_mapping.csv", index=False)

    module_genes_df = pd.DataFrame([
        {"module": k, "mouse_gene": g, **{kk: vv for kk, vv in v.items() if kk != "mouse_genes"}}
        for k, v in modules.items() for g in v["mouse_genes"]
    ])
    module_genes_df.to_csv(OUT / "ion_cci_module_gene_sets.csv", index=False)

    scores, details = score_modules_on_human_blood(modules, ortholog_maps)
    scores.to_csv(OUT / "human_blood_module_scores.csv", index=False)
    details.to_csv(OUT / "human_blood_module_details.csv", index=False)

    assoc = test_module_subtype_associations(scores)
    assoc.to_csv(OUT / "module_subtype_association_tests.csv", index=False)

    perm = permutation_test_module_subtype(scores)
    perm.to_csv(OUT / "module_subtype_permutation_tests.csv", index=False)
    if len(perm) and len(assoc):
        assoc = assoc.merge(
            perm[["module", "permutation_p_value", "n_permutations"]],
            on="module",
            how="left",
        )
        assoc.to_csv(OUT / "module_subtype_association_tests.csv", index=False)

    # Heatmap
    pivot = scores.pivot(index="sample_id", columns="module", values="module_score")
    subtype_map = pd.read_csv(project_root / "results" / "molecular_subtypes.csv").set_index("sample_id")["subtype"]
    pivot = pivot.loc[pivot.index.intersection(subtype_map.index)]
    col_colors = subtype_map.reindex(pivot.index).map({0: "#1f77b4", 1: "#ff7f0e"})

    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Module score (z-mean, direction-adjusted)"})
        ax.set_title("Supplementary Figure S14. IoN-CCI module scores on TN blood (n=10, k=2)")
        plt.tight_layout()
        plt.savefig(FIG / "module_scores_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Boxplot: key modules by subtype
    key_mods = ["peripheral_ecm_schwann", "central_synaptic", "ntrk_signaling"]
    plot_scores = scores[scores["module"].isin(key_mods)]
    if len(plot_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=plot_scores, x="module", y="module_score", hue="subtype", ax=ax)
        ax.set_title("Supplementary Figure S15. IoN-CCI module scores by TN molecular subtype (k=2)")
        ax.set_xlabel("")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / "module_scores_by_subtype.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"Module association tests saved ({len(assoc)} modules)")
    return scores, assoc, modules, ortholog_maps


def _load_participants_imaging_merged() -> pd.DataFrame:
    part = pd.read_csv(project_root / "data" / "external" / "openneuro" / "ds005713" / "participants.tsv", sep="\t")
    part = part.rename(columns={"BIDS_ID": "subject_id"})

    img = pd.read_csv(project_root / "results" / "aim2" / "real_processing" / "imaging_subtypes_all_subjects.csv")
    img_pre = img[~img["subject_id"].astype(str).str.endswith("fu")].copy()

    merged = part.merge(img_pre, on="subject_id", how="inner")
    return merged


def analysis_2_mvd_outcome_model():
    logger.info("=" * 70)
    logger.info("ANALYSIS 2: OpenNeuro MVD outcome model")
    logger.info("=" * 70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = _load_participants_imaging_merged()
    mvd = df[df["Surgery_type"] == "MVD"].copy()
    mvd = mvd[mvd["Surgery_outcome"].isin(["Pos", "Neg"])].copy()
    mvd["outcome_binary"] = (mvd["Surgery_outcome"] == "Pos").astype(int)

    clinical_cols = ["age", "sex"]
    for c in ["Sindou_grade", "Pain_severity (average score)", "Disease_duration (years)"]:
        if c in mvd.columns:
            mvd[c] = pd.to_numeric(mvd[c], errors="coerce")
            clinical_cols.append(c)

    imaging_cols = [c for c in ["fa_mean", "md_mean", "rd_mean", "ad_mean", "t1_mean", "t2_mean", "fmri_mean",
                                 "compression_index", "tortuosity", "nerve_volume", "imaging_subtype",
                                 "clustering_coefficient", "modularity", "global_efficiency"] if c in mvd.columns]

    feature_cols = clinical_cols + imaging_cols
    X = mvd[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Drop all-NaN imaging columns (avoids imputer/sklearn shape mismatch)
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep]
    feature_cols = keep
    y = mvd["outcome_binary"].values

    logger.info(f"MVD cohort: n={len(mvd)}, responders={y.sum()}, non-responders={(1-y).sum()}")

    if len(np.unique(y)) < 2 or len(mvd) < 8:
        logger.error("Insufficient MVD outcome diversity for modeling")
        return None

    n_splits = min(4, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=42)),
        ]),
    }

    results = []
    boot_rows = []
    pred_store = mvd[["subject_id", "Surgery_outcome"]].copy()
    pred_store["outcome_binary"] = y

    for name, pipe in models.items():
        try:
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
            pred_store[f"pred_proba_{name}"] = proba
            pipe.fit(X, y)
            if name == "RandomForest":
                imp = pipe.named_steps["clf"].feature_importances_
            elif name == "LogisticRegression":
                imp = np.abs(pipe.named_steps["clf"].coef_[0])
            else:
                imp = np.zeros(len(feature_cols))
            if len(imp) != len(feature_cols):
                imp = np.pad(imp, (0, max(0, len(feature_cols) - len(imp))))[: len(feature_cols)]
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
            imp_df.to_csv(OUT / f"mvd_model_{name}_feature_importance.csv", index=False)

            boot = bootstrap_mvd_auc_ci(y, proba, n_boot=2000, seed=42)
            boot_rows.append({"model": name, **boot})
            results.append({
                "model": name,
                "cv_auc": auc,
                "bootstrap_mean_auc": boot["bootstrap_mean_auc"],
                "auc_ci_lower": boot["ci_lower"],
                "auc_ci_upper": boot["ci_upper"],
                "n_bootstrap": boot["n_bootstrap"],
                "n_samples": len(mvd),
                "n_features": len(feature_cols),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
            })
        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "mvd_outcome_model_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "mvd_outcome_bootstrap_ci.csv", index=False)
    pred_store.to_csv(OUT / "mvd_outcome_predictions.csv", index=False)
    mvd.to_csv(OUT / "mvd_modeling_cohort.csv", index=False)

    if results and "pred_proba_LogisticRegression" in pred_store.columns:
        lr_row = results_df[results_df["model"] == "LogisticRegression"].iloc[0]
        lr_auc = lr_row["cv_auc"]
        lr_ci = f"{lr_row['auc_ci_lower']:.3f}–{lr_row['auc_ci_upper']:.3f}" if pd.notna(lr_row.get("auc_ci_lower")) else "NA"
        fpr, tpr, _ = roc_curve(y, pred_store["pred_proba_LogisticRegression"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"Logistic CV AUC={lr_auc:.3f} (95% bootstrap CI: {lr_ci})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"Supplementary Figure S16. MVD outcome prediction (n={len(mvd)} MVD patients)")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG / "mvd_outcome_roc.png", dpi=300, bbox_inches="tight")
        plt.close()

    return results_df


def analysis_3_neuroplasticity():
    logger.info("=" * 70)
    logger.info("ANALYSIS 3: Pre→post neuroplasticity (fu sessions)")
    logger.info("=" * 70)

    img = pd.read_csv(project_root / "results" / "aim2" / "real_processing" / "imaging_subtypes_all_subjects.csv")
    part = pd.read_csv(project_root / "data" / "external" / "openneuro" / "ds005713" / "participants.tsv", sep="\t")
    part = part.rename(columns={"BIDS_ID": "subject_id"})

    numeric_feats = ["fa_mean", "md_mean", "rd_mean", "ad_mean", "t1_mean", "t2_mean", "fmri_mean",
                     "clustering_coefficient", "modularity", "global_efficiency"]

    pre = img[~img["subject_id"].astype(str).str.endswith("fu")].set_index("subject_id")
    post = img[img["subject_id"].astype(str).str.endswith("fu")].copy()
    post["base_id"] = post["subject_id"].str.replace("fu", "", regex=False)

    rows = []
    for _, prow in post.iterrows():
        base = prow["base_id"]
        if base not in pre.index:
            continue
        pre_row = pre.loc[base]
        rec = {"subject_id": base, "post_id": prow["subject_id"]}
        for f in numeric_feats:
            if f not in img.columns:
                continue
            pre_v = pd.to_numeric(pre_row.get(f), errors="coerce")
            post_v = pd.to_numeric(prow.get(f), errors="coerce")
            if pd.notna(pre_v) and pd.notna(post_v):
                rec[f"pre_{f}"] = pre_v
                rec[f"post_{f}"] = post_v
                rec[f"delta_{f}"] = post_v - pre_v
                rec[f"pct_delta_{f}"] = 100 * (post_v - pre_v) / (abs(pre_v) + 1e-9)
        rows.append(rec)

    delta_df = pd.DataFrame(rows)
    if len(delta_df) == 0:
        logger.warning("No pre/post pairs found")
        return None

    # Restrict to participants with known surgical outcome (MVD/PI)
    delta_df = delta_df.merge(
        part[["subject_id", "Surgery_outcome", "Postsurgery_pain_reduction (%)", "Surgery_type"]],
        on="subject_id",
        how="inner",
    )
    delta_df = delta_df[delta_df["Surgery_outcome"].isin(["Pos", "Neg"])].copy()
    delta_df.to_csv(OUT / "pre_post_neuroplasticity_deltas.csv", index=False)

    # Stats: delta vs outcome
    stat_rows = []
    delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]
    for col in delta_cols:
        sub = delta_df.dropna(subset=[col, "Surgery_outcome"])
        pos = sub[sub["Surgery_outcome"] == "Pos"][col]
        neg = sub[sub["Surgery_outcome"] == "Neg"][col]
        if len(pos) >= 2 and len(neg) >= 1:
            u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            stat_rows.append({
                "feature_delta": col.replace("delta_", ""),
                "mean_delta_responders": pos.mean(),
                "mean_delta_nonresponders": neg.mean(),
                "n_responders": len(pos),
                "n_nonresponders": len(neg),
                "mannwhitney_p": p,
            })
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT / "pre_post_delta_outcome_tests.csv", index=False)

    # Figure: top delta feature by outcome
    if len(stat_df) > 0:
        best = stat_df.sort_values("mannwhitney_p").iloc[0]["feature_delta"]
        col = f"delta_{best}"
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_df = delta_df.dropna(subset=[col, "Surgery_outcome"])
        sns.boxplot(data=plot_df, x="Surgery_outcome", y=col, ax=ax, palette={"Pos": "#2ca02c", "Neg": "#d62728"})
        ax.set_title(f"Supplementary Figure S17. Pre→post Δ {best} by MVD outcome (n={len(plot_df)} pairs)")
        ax.set_xlabel("Surgery outcome")
        plt.tight_layout()
        plt.savefig(FIG / "pre_post_neuroplasticity_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"Pre/post pairs: {len(delta_df)}")
    return delta_df, stat_df


def analysis_4_mechanism_imaging_concordance(scores: pd.DataFrame, modules: dict):
    logger.info("=" * 70)
    logger.info("ANALYSIS 4: Mechanism–imaging concordance")
    logger.info("=" * 70)

    img = pd.read_csv(project_root / "results" / "aim2" / "real_processing" / "imaging_subtypes_all_subjects.csv")
    img_pre = img[~img["subject_id"].astype(str).str.endswith("fu")]

    imaging_feats = ["fa_mean", "md_mean", "t1_mean", "t2_mean", "fmri_mean", "compression_index",
                     "modularity", "global_efficiency", "clustering_coefficient"]
    imaging_feats = [f for f in imaging_feats if f in img_pre.columns]

    subtype_centroids = img_pre.groupby("imaging_subtype")[imaging_feats].mean(numeric_only=True)

    # Blood module enrichment by molecular subtype (k=2)
    blood_mod = scores.groupby(["module", "subtype"])["module_score"].mean().unstack()

    rows = []
    for module, feat, expected_sign, rationale in MECHANISM_IMAGING_PRIORS:
        if module not in blood_mod.index or feat not in subtype_centroids.columns:
            continue

        # Imaging: correlate imaging subtype index with feature centroid (3 subtypes)
        cent = subtype_centroids[feat].dropna()
        if len(cent) < 2:
            continue
        x = cent.index.astype(float).values
        y = cent.values
        rho_img, p_img = stats.spearmanr(x, y) if len(cent) >= 3 else (np.nan, np.nan)

        # Blood: subtype 1 vs 0 module score difference
        if 0 in blood_mod.columns and 1 in blood_mod.columns:
            blood_delta = blood_mod.loc[module, 1] - blood_mod.loc[module, 0]
        else:
            blood_delta = np.nan

        expected_rho = 1.0 if expected_sign == "positive" else -1.0
        concordant = (np.sign(rho_img) == np.sign(expected_rho)) if pd.notna(rho_img) and rho_img != 0 else np.nan

        rows.append({
            "ion_cci_module": module,
            "module_description": modules.get(module, {}).get("description", ""),
            "imaging_feature": feat,
            "expected_association": expected_sign,
            "rationale": rationale,
            "blood_score_delta_subtype1_minus_0": blood_delta,
            "imaging_subtype_spearman_rho": rho_img,
            "imaging_subtype_spearman_p": p_img,
            "imaging_subtype_0_mean": cent.get(0, np.nan),
            "imaging_subtype_1_mean": cent.get(1, np.nan),
            "imaging_subtype_2_mean": cent.get(2, np.nan),
            "concordant_with_prior": concordant,
        })

    concordance = pd.DataFrame(rows)
    concordance.to_csv(OUT / "mechanism_imaging_concordance_table.csv", index=False)

    # MVD outcome correlation with imaging features (clinical validation layer)
    mvd = _load_participants_imaging_merged()
    mvd = mvd[(mvd["Surgery_type"] == "MVD") & (mvd["Surgery_outcome"].isin(["Pos", "Neg"]))].copy()
    mvd["outcome_binary"] = (mvd["Surgery_outcome"] == "Pos").astype(int)
    outcome_rows = []
    for feat in imaging_feats:
        sub = mvd.dropna(subset=[feat])
        if len(sub) >= 6:
            rho, p = stats.pointbiserialr(sub["outcome_binary"], sub[feat])
            outcome_rows.append({"imaging_feature": feat, "pointbiserial_r": rho, "p_value": p, "n_mvd": len(sub)})
    outcome_corr = pd.DataFrame(outcome_rows)
    outcome_corr.to_csv(OUT / "imaging_feature_mvd_outcome_correlations.csv", index=False)

    logger.info(f"Concordance table: {len(concordance)} rows")
    return concordance, outcome_corr


def analysis_5_drug_repurposing(ortholog_maps: dict):
    logger.info("=" * 70)
    logger.info("ANALYSIS 5: Drug repurposing shortlist")
    logger.info("=" * 70)

    de0 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_0.csv")
    de1 = pd.read_csv(project_root / "results" / "subtype_characterization" / "de_subtype_1.csv")
    lfc = "log2fc" if "log2fc" in de0.columns else "log2FoldChange"
    padj = "p_adj" if "p_adj" in de0.columns else "padj"
    de0["gene_u"] = de0["gene"].astype(str).str.upper()
    de1["gene_u"] = de1["gene"].astype(str).str.upper()

    # Module genes (human orthologs) from NTRK/synaptic modules
    module_genes = {}
    for mod in ["ntrk_signaling", "central_synaptic", "peripheral_ecm_schwann"]:
        if mod in ortholog_maps:
            module_genes[mod] = set(ortholog_maps[mod]["human_symbol"].str.upper())

    rows = []
    for drug_info in CURATED_DRUGS:
        for target in drug_info["targets"]:
            tu = target.upper()
            in_mod = [m for m, genes in module_genes.items() if tu in genes]
            r0 = de0[de0["gene_u"] == tu]
            r1 = de1[de1["gene_u"] == tu]
            lfc0 = r0[lfc].iloc[0] if len(r0) else np.nan
            lfc1 = r1[lfc].iloc[0] if len(r1) else np.nan
            p0 = r0[padj].iloc[0] if len(r0) else np.nan
            p1 = r1[padj].iloc[0] if len(r1) else np.nan

            # Direction concordance: IoN-CCI down → expect human down in at least one subtype
            ion_dir = drug_info["ion_cci_logic"]
            human_concord = ""
            if ion_dir == "down":
                if pd.notna(lfc0) and lfc0 < -0.5:
                    human_concord += "concordant_subtype0_down;"
                if pd.notna(lfc1) and lfc1 < -0.5:
                    human_concord += "concordant_subtype1_down;"
                if not human_concord:
                    human_concord = "not_significant_or_discordant"

            priority = 0
            if in_mod:
                priority += 3
            if "concordant" in human_concord:
                priority += 2
            if drug_info["drug"] in ("Gabapentin", "Pregabalin", "Duloxetine"):
                priority += 1

            rows.append({
                "drug": drug_info["drug"],
                "target_gene": target,
                "mechanism": drug_info["mechanism"],
                "ion_cci_module": ";".join(in_mod) if in_mod else "",
                "ion_cci_expected_direction": ion_dir,
                "human_log2fc_subtype0": lfc0,
                "human_padj_subtype0": p0,
                "human_log2fc_subtype1": lfc1,
                "human_padj_subtype1": p1,
                "human_de_concordance": human_concord,
                "priority_score": priority,
                "rationale": f"IoN-CCI {ion_dir}-regulated module target; check human TN subtype DE direction",
            })

    drugs = pd.DataFrame(rows).sort_values(["priority_score", "drug"], ascending=[False, True])
    drugs.to_csv(OUT / "drug_repurposing_ntrk_synaptic_shortlist.csv", index=False)

    # Top module genes with human DE for manual review
    gene_rows = []
    for mod, genes in module_genes.items():
        for g in sorted(genes):
            r0 = de0[de0["gene_u"] == g]
            r1 = de1[de1["gene_u"] == g]
            gene_rows.append({
                "module": mod,
                "human_gene": g,
                "log2fc_subtype0": r0[lfc].iloc[0] if len(r0) else np.nan,
                "padj_subtype0": r0[padj].iloc[0] if len(r0) else np.nan,
                "log2fc_subtype1": r1[lfc].iloc[0] if len(r1) else np.nan,
                "padj_subtype1": r1[padj].iloc[0] if len(r1) else np.nan,
            })
    pd.DataFrame(gene_rows).to_csv(OUT / "module_gene_human_de_review.csv", index=False)

    top = drugs[drugs["priority_score"] >= 3].head(15)
    logger.info(f"Drug shortlist: {len(drugs)} entries, {len(top)} high-priority")
    return drugs


def write_summary(scores, assoc, mvd_results, delta_df, concordance, drugs, perm=None):
    lines = [
        "# High-Impact Cross-Dataset Integration Summary",
        "",
        "## 1. IoN-CCI module scoring → human blood subtypes (k=2, n=10 TN)",
        "",
    ]
    if assoc is not None and len(assoc):
        for _, r in assoc.iterrows():
            perm_p = r.get("permutation_p_value", np.nan)
            perm_str = f", perm p={perm_p:.4f}" if pd.notna(perm_p) else ""
            lines.append(
                f"- **{r['module']}**: Δ(subtype1−0)={r['delta_1_minus_0']:.3f}, Cohen's d={r['cohens_d']:.2f}, Welch p={r['p_value']:.4f}{perm_str}"
            )

    lines.extend(["", "## 2. OpenNeuro MVD outcome model", ""])
    if mvd_results is not None and len(mvd_results):
        for _, r in mvd_results.iterrows():
            ci = ""
            if pd.notna(r.get("auc_ci_lower")) and pd.notna(r.get("auc_ci_upper")):
                ci = f", 95% bootstrap CI [{r['auc_ci_lower']:.3f}, {r['auc_ci_upper']:.3f}]"
            lines.append(
                f"- **{r['model']}**: CV AUC={r['cv_auc']:.3f}{ci} (n={int(r['n_samples'])}, pos={int(r['n_pos'])}, neg={int(r['n_neg'])})"
            )

    lines.extend(["", "## 3. Pre→post neuroplasticity", ""])
    if delta_df is not None:
        lines.append(f"- Pre/post pairs analyzed: **{len(delta_df)}**")

    lines.extend(["", "## 4. Mechanism–imaging concordance", ""])
    if concordance is not None and len(concordance):
        n_con = concordance["concordant_with_prior"].sum() if "concordant_with_prior" in concordance else 0
        lines.append(f"- Prior-hypothesis concordant rows: **{int(n_con)}/{len(concordance)}**")

    lines.extend(["", "## 5. Drug repurposing shortlist", ""])
    if drugs is not None:
        top = drugs.sort_values("priority_score", ascending=False).head(10)
        for _, r in top.iterrows():
            lines.append(f"- **{r['drug']}** → {r['target_gene']} (priority={int(r['priority_score'])}, {r['human_de_concordance']})")

    (OUT / "HIGH_IMPACT_INTEGRATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    logger.info("Starting high-impact cross-dataset analyses")
    scores, assoc, modules, ortholog_maps = analysis_1_module_scoring()
    perm = pd.read_csv(OUT / "module_subtype_permutation_tests.csv") if (OUT / "module_subtype_permutation_tests.csv").exists() else None
    mvd_results = analysis_2_mvd_outcome_model()
    delta_out = analysis_3_neuroplasticity()
    delta_df = delta_out[0] if delta_out else None
    concordance, _ = analysis_4_mechanism_imaging_concordance(scores, modules)
    drugs = analysis_5_drug_repurposing(ortholog_maps)
    write_summary(scores, assoc, mvd_results, delta_df, concordance, drugs, perm=perm)

    manifest = {
        "output_dir": str(OUT),
        "analyses": [
            "ion_cci_human_ortholog_mapping.csv",
            "human_blood_module_scores.csv",
            "module_subtype_association_tests.csv",
            "module_subtype_permutation_tests.csv",
            "mvd_outcome_model_results.csv",
            "mvd_outcome_bootstrap_ci.csv",
            "pre_post_neuroplasticity_deltas.csv",
            "mechanism_imaging_concordance_table.csv",
            "drug_repurposing_ntrk_synaptic_shortlist.csv",
            "HIGH_IMPACT_INTEGRATION_SUMMARY.md",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"All analyses complete → {OUT}")
    return True


if __name__ == "__main__":
    main()
