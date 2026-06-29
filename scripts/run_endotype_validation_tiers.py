"""
Three-tier endotype validation:

Tier 1: External convergent validation — GSE177034 (persistent vs resolved NP blood)
        Secondary: GSE124272 (IDD pain vs healthy)
Tier 2: Internal validation — GSE186505 LOO, split-half stability, TN vs control panels
Tier 3: Cross-tissue mechanistic — IoN-CCI ↔ human TG (GSE197289 + PMC6326384 bulk)
"""

from __future__ import annotations

import gzip
import json
import re
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

OUT = project_root / "results" / "endotype_validation"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODULES = ["peripheral_ecm_schwann", "central_synaptic", "ntrk_signaling"]
MODULE_DIRECTION = {
    "peripheral_ecm_schwann": "up",
    "central_synaptic": "down",
    "ntrk_signaling": "down",
}


def load_frozen_human_module_genes() -> dict[str, list[str]]:
    """Human ortholog gene lists from IoN-CCI mapping (frozen discovery set)."""
    ortho = pd.read_csv(project_root / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")
    out = {}
    for mod in MODULES:
        genes = ortho.loc[ortho["module"] == mod, "human_symbol"].dropna().astype(str).str.upper().unique().tolist()
        # Keep only plausible human symbols (exclude mouse-only IDs)
        genes = [g for g in genes if not g.startswith("GM") and not g.endswith("RIK") and "ENSMUS" not in g]
        out[mod] = genes
    return out


def score_modules(expr: pd.DataFrame, module_genes: dict[str, list[str]]) -> pd.DataFrame:
    """Direction-adjusted module scores. expr: genes × samples."""
    expr = expr.copy()
    expr.index = expr.index.astype(str).str.upper()
    z = expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    rows = []
    for sample in z.columns:
        for mod, genes in module_genes.items():
            avail = [g for g in genes if g in z.index]
            if not avail:
                score = np.nan
            else:
                raw = z.loc[avail, sample].mean()
                score = raw if MODULE_DIRECTION[mod] == "up" else -raw
            rows.append({
                "sample_id": sample,
                "module": mod,
                "module_score": score,
                "n_genes_used": len(avail),
                "coverage": len(avail) / max(len(genes), 1),
            })
    wide = pd.DataFrame(rows).pivot(index="sample_id", columns="module", values="module_score")
    wide["central_axis"] = wide[["central_synaptic", "ntrk_signaling"]].mean(axis=1) - wide["peripheral_ecm_schwann"]
    return wide.reset_index()


def parse_geo_series_matrix_metadata(path: Path) -> pd.DataFrame:
    """Parse !Sample_* rows from GEO series matrix into sample × metadata."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sample_keys = []
    rows = {}

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue
            parts = line.strip().split("\t")
            field = parts[0].replace("!Sample_", "")
            vals = [p.strip('"') for p in parts[1:]]
            if field == "geo_accession":
                sample_keys = vals
            else:
                for sid, val in zip(sample_keys, vals):
                    rows.setdefault(sid, {})[field] = val

    meta = pd.DataFrame.from_dict(rows, orient="index")
    meta.index.name = "geo_accession"
    meta = meta.reset_index()
    if "title" in meta.columns:
        meta["sample_id"] = meta["title"]
    elif "geo_accession" in meta.columns:
        meta["sample_id"] = meta["geo_accession"]
    return meta


def load_gse177034_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSE177034 featureCounts matrix + phenotype metadata."""
    base = project_root / "data" / "external" / "geo" / "GSE177034"
    counts_gz = base / "GSE177034_lbp_featurecounts.txt.gz"
    matrix_gz = base / "GSE177034_series_matrix.txt.gz"
    if not counts_gz.exists():
        raise FileNotFoundError(f"Missing {counts_gz}; run download first")

    df = pd.read_csv(counts_gz, sep="\t", compression="gzip")
    gene_col = "Geneid"
    meta_cols = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
    sample_cols = [c for c in df.columns if c not in meta_cols]
    expr = df.set_index(gene_col)[sample_cols]
    expr.index = expr.index.astype(str).str.upper()
    # log2 CPM
    lib = expr.sum(axis=0).replace(0, np.nan)
    cpm = expr.div(lib, axis=1) * 1e6
    logcpm = np.log2(cpm + 1)

    meta = parse_geo_series_matrix_metadata(matrix_gz)
    # Parse characteristics into columns
    for col in meta.columns:
        if col.startswith("characteristics_ch1"):
            pass
    # Consolidate characteristics (multiple ch1 columns get merged in dict - need fix)
    # Re-parse all characteristics lines
    char_rows = {}
    with gzip.open(matrix_gz, "rt", encoding="utf-8", errors="replace") as f:
        titles = None
        for line in f:
            if line.startswith("!Sample_title"):
                titles = [p.strip('"') for p in line.strip().split("\t")[1:]]
                for t in titles:
                    char_rows.setdefault(t, {})
            if line.startswith("!Sample_characteristics_ch1") and titles:
                vals = [p.strip('"') for p in line.strip().split("\t")[1:]]
                for t, v in zip(titles, vals):
                    if ":" in v:
                        key, val = v.split(":", 1)
                        char_rows.setdefault(t, {})[key.strip()] = val.strip()

    pheno = pd.DataFrame.from_dict(char_rows, orient="index").reset_index().rename(columns={"index": "sample_id"})
    if "geo_accession" in meta.columns and "title" in meta.columns:
        acc_map = dict(zip(meta["title"], meta["geo_accession"]))
        pheno["geo_accession"] = pheno["sample_id"].map(acc_map)
    return logcpm, pheno


def load_gse124272_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSE124272 series matrix expression + labels."""
    path = project_root / "data" / "external" / "geo" / "GSE124272" / "GSE124272_series_matrix.txt.gz"
    if not path.exists():
        raise FileNotFoundError(path)

    meta = parse_geo_series_matrix_metadata(path)
    # Read expression table (after series table header)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    table = pd.read_csv(pd.io.common.StringIO("".join(lines[start + 1 : end])), sep="\t")
    table = table.set_index(table.columns[0])
    table.index = table.index.astype(str).str.upper()
    table = table.apply(pd.to_numeric, errors="coerce")
    table = np.log2(table + 1)

    labels = []
    for _, r in meta.iterrows():
        title = r.get("title", "")
        grp = "NP_pain" if "patient" in title.lower() or "idd" in title.lower() else "control"
        labels.append({"sample_id": title, "group": grp})
    pheno = pd.DataFrame(labels)
    # Align columns
    table.columns = meta["title"].values if "title" in meta.columns else table.columns
    return table, pheno


def tier1_external_validation(module_genes: dict) -> dict:
    logger.info("TIER 1: External convergent validation (GSE177034)")
    logcpm, pheno = load_gse177034_counts()

    # Clean sample names: sample.0098.t0.bam -> sample.0098.t0
    col_map = {c: re.sub(r"\.bam$", "", c) for c in logcpm.columns}
    logcpm = logcpm.rename(columns=col_map)
    pheno["sample_id"] = pheno["sample_id"].astype(str)

    scores = score_modules(logcpm, module_genes)
    merged = scores.merge(pheno, on="sample_id", how="inner")
    merged.to_csv(OUT / "tier1_gse177034_module_scores.csv", index=False)

    results = {"cohort": "GSE177034", "claim_scope": "convergent_neuropathic_pain_biology_not_TN_replication"}

    # Primary: t0 baseline — Persistent vs Resolved (3-month outcome)
    t0 = merged[merged.get("timepoint", merged.get("timepoint", pd.Series())) == "t0"] if "timepoint" in merged.columns else merged[merged["sample_id"].str.endswith(".t0")]
    if "timepoint" not in merged.columns:
        t0 = merged[merged["sample_id"].str.contains(r"\.t0$", regex=True)]

    for time_label, sub in [("t0_baseline", t0), ("t1_followup", merged[merged["sample_id"].str.contains(r"\.t1$", regex=True)])]:
        if "paingroup" not in sub.columns or len(sub) < 10:
            continue
        rows = []
        for mod in MODULES + ["central_axis"]:
            pers = sub[sub["paingroup"] == "Persistent"][mod].dropna()
            res = sub[sub["paingroup"] == "Resolved"][mod].dropna()
            if len(pers) >= 3 and len(res) >= 3:
                u, p = stats.mannwhitneyu(pers, res, alternative="two-sided")
                d = (pers.mean() - res.mean()) / np.sqrt((pers.var() + res.var()) / 2) if (pers.var() + res.var()) > 0 else 0
                rows.append({
                    "timepoint": time_label,
                    "module": mod,
                    "mean_persistent": pers.mean(),
                    "mean_resolved": res.mean(),
                    "delta_persistent_minus_resolved": pers.mean() - res.mean(),
                    "cohens_d": d,
                    "mannwhitney_p": p,
                    "n_persistent": len(pers),
                    "n_resolved": len(res),
                })
        test_df = pd.DataFrame(rows)
        test_df.to_csv(OUT / f"tier1_gse177034_{time_label}_tests.csv", index=False)

        # ROC: central_axis predicting Persistent
        if "central_axis" in sub.columns and sub["paingroup"].nunique() == 2:
            y = (sub["paingroup"] == "Persistent").astype(int)
            x = sub["central_axis"].fillna(0)
            try:
                auc = roc_auc_score(y, x)
                results[f"{time_label}_central_axis_auc_persistent"] = float(auc)
            except Exception:
                pass

        # Figure
        if len(t0) > 0 and time_label == "t0_baseline":
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, mod in zip(axes, MODULES):
                plot_df = sub.dropna(subset=[mod, "paingroup"])
                sns.boxplot(data=plot_df, x="paingroup", y=mod, ax=ax, order=["Resolved", "Persistent"])
                ax.set_title(mod.replace("_", " "))
            plt.suptitle("GSE177034: IoN-CCI modules by pain outcome (t0 baseline)")
            plt.tight_layout()
            plt.savefig(FIG / "tier1_gse177034_module_boxplots.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Secondary GSE124272
    try:
        expr124, pheno124 = load_gse124272_expression()
        scores124 = score_modules(expr124, module_genes)
        m124 = scores124.merge(pheno124, on="sample_id", how="left")
        m124.to_csv(OUT / "tier1_gse124272_module_scores.csv", index=False)
        rows124 = []
        for mod in MODULES + ["central_axis"]:
            np_g = m124[m124["group"] == "NP_pain"][mod].dropna()
            ctrl = m124[m124["group"] == "control"][mod].dropna()
            if len(np_g) >= 3 and len(ctrl) >= 3:
                u, p = stats.mannwhitneyu(np_g, ctrl, alternative="two-sided")
                rows124.append({"module": mod, "mean_np": np_g.mean(), "mean_control": ctrl.mean(), "mannwhitney_p": p})
        pd.DataFrame(rows124).to_csv(OUT / "tier1_gse124272_tests.csv", index=False)
        results["gse124272"] = "completed"
    except Exception as exc:
        logger.warning(f"GSE124272 skipped: {exc}")
        results["gse124272"] = f"skipped: {exc}"

    pd.DataFrame([results]).to_csv(OUT / "tier1_summary.csv", index=False)
    return results


def tier2_internal_validation(module_genes: dict) -> dict:
    logger.info("TIER 2: Internal validation (GSE186505)")
    expr = pd.read_csv(project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv", index_col=0)
    expr.index = expr.index.astype(str).str.upper()
    subtypes = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")
    sig = json.loads((project_root / "results" / "endotype_framework" / "portable_endotype_signatures.json").read_text())

    tn_samples = subtypes["sample_id"].tolist()
    control_samples = [c for c in expr.columns if c.startswith("Control")]
    all_samples = tn_samples + control_samples

    # --- LOO module scoring (TN only) ---
    loo_rows = []
    z_full = expr[tn_samples].sub(expr[tn_samples].mean(axis=1), axis=0).div(expr[tn_samples].std(axis=1).replace(0, np.nan), axis=0).fillna(0)

    for held_out in tn_samples:
        ref = [s for s in tn_samples if s != held_out]
        ref_expr = expr[ref]
        z_ref = ref_expr.sub(ref_expr.mean(axis=1), axis=0).div(ref_expr.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
        for mod, genes in module_genes.items():
            avail = [g for g in genes if g in z_ref.index]
            if not avail:
                continue
            raw = z_ref.loc[avail, held_out].mean() if held_out in z_ref.columns else np.nan
            if held_out not in expr.columns:
                continue
            # score held-out using ref-derived z embedded in ref z matrix - need held-out expr vs ref stats
            mu = ref_expr.loc[avail].mean(axis=1)
            sd = ref_expr.loc[avail].std(axis=1).replace(0, np.nan)
            z_ho = ((expr.loc[avail, held_out] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0).mean()
            score = z_ho if MODULE_DIRECTION[mod] == "up" else -z_ho
            st = subtypes.loc[subtypes["sample_id"] == held_out, "subtype"].iloc[0]
            loo_rows.append({"sample_id": held_out, "subtype": st, "module": mod, "loo_module_score": score})

    loo_df = pd.DataFrame(loo_rows)
    loo_wide = loo_df.pivot(index="sample_id", columns="module", values="loo_module_score")
    loo_wide["central_axis"] = loo_wide[["central_synaptic", "ntrk_signaling"]].mean(axis=1) - loo_wide["peripheral_ecm_schwann"]
    loo_wide = loo_wide.merge(subtypes, left_index=True, right_on="sample_id")
    loo_df.to_csv(OUT / "tier2_loo_module_scores.csv", index=False)

    loo_tests = []
    for mod in MODULES + ["central_axis"]:
        g0 = loo_wide[loo_wide["subtype"] == 0][mod].dropna()
        g1 = loo_wide[loo_wide["subtype"] == 1][mod].dropna()
        if len(g0) >= 2 and len(g1) >= 2:
            u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            loo_tests.append({"module": mod, "mannwhitney_p": p, "mean_subtype_0": g0.mean(), "mean_subtype_1": g1.mean()})
    pd.DataFrame(loo_tests).to_csv(OUT / "tier2_loo_subtype_tests.csv", index=False)

    # --- Split-half clustering stability ---
    expr_tn = expr[tn_samples].T
    scaler = StandardScaler()
    X = scaler.fit_transform(expr_tn)
    pca = PCA(n_components=min(5, len(tn_samples) - 1), random_state=42)
    Xp = pca.fit_transform(X)

    ari_scores = []
    axis_corr = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        idx = rng.permutation(len(tn_samples))
        h1, h2 = idx[:5], idx[5:]
        if len(h1) < 4 or len(h2) < 4:
            continue
        km1 = KMeans(n_clusters=2, random_state=42, n_init=20).fit(Xp[h1])
        km2 = KMeans(n_clusters=2, random_state=42, n_init=20).fit(Xp[h2])
        # Module axis consistency: cluster 0 vs 1 central-peripheral difference same sign?
        s1 = score_modules(expr[[tn_samples[i] for i in h1]], module_genes)
        s1["cluster"] = km1.labels_
        s2 = score_modules(expr[[tn_samples[i] for i in h2]], module_genes)
        s2["cluster"] = km2.labels_
        if "central_axis" in s1.columns:
            d1 = s1.groupby("cluster")["central_axis"].mean().diff().iloc[-1] if s1["cluster"].nunique() == 2 else 0
            d2 = s2.groupby("cluster")["central_axis"].mean().diff().iloc[-1] if s2["cluster"].nunique() == 2 else 0
            axis_corr.append(np.sign(d1) == np.sign(d2))
        # ARI on overlapping labels not defined across splits - use full sample label agreement via centroid assignment
        ari_scores.append(adjusted_rand_score(km1.labels_, km2.labels_) if len(h1) == len(h2) else np.nan)

    split_half = {
        "n_iterations": 500,
        "axis_sign_consistency_frac": float(np.mean(axis_corr)) if axis_corr else np.nan,
        "mean_ari_random_halves": float(np.nanmean(ari_scores)),
    }
    pd.DataFrame([split_half]).to_csv(OUT / "tier2_split_half_stability.csv", index=False)

    # --- TN vs control gene panel direction ---
    panel_rows = []
    peripheral_genes = [g.upper() for g in sig["endotypes"]["peripheral_subtype_0"]["gene_panel"]]
    central_genes = [g.upper() for g in sig["endotypes"]["central_subtype_1"]["gene_panel"]]
    for panel_name, genes in [("peripheral_panel", peripheral_genes), ("central_panel", central_genes)]:
        avail = [g for g in genes if g in expr.index]
        for g in avail:
            tn_v = expr.loc[g, tn_samples].astype(float)
            ctrl_v = expr.loc[g, control_samples].astype(float)
            u, p = stats.mannwhitneyu(tn_v, ctrl_v, alternative="two-sided")
            panel_rows.append({
                "panel": panel_name,
                "gene": g,
                "mean_tn": tn_v.mean(),
                "mean_control": ctrl_v.mean(),
                "log2fc_tn_vs_control": np.log2((tn_v.mean() + 1) / (ctrl_v.mean() + 1)),
                "mannwhitney_p": p,
            })
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(OUT / "tier2_gene_panel_tn_vs_control.csv", index=False)

    # Module scores TN vs control
    scores_all = score_modules(expr[all_samples], module_genes)
    scores_all["group"] = scores_all["sample_id"].apply(lambda s: "TN" if s in tn_samples else "control")
    scores_all.to_csv(OUT / "tier2_gse186505_module_scores_all.csv", index=False)
    tc_tests = []
    for mod in MODULES + ["central_axis"]:
        tn_sc = scores_all[scores_all["group"] == "TN"][mod]
        ct_sc = scores_all[scores_all["group"] == "control"][mod]
        u, p = stats.mannwhitneyu(tn_sc, ct_sc, alternative="two-sided")
        tc_tests.append({"module": mod, "mean_tn": tn_sc.mean(), "mean_control": ct_sc.mean(), "mannwhitney_p": p})
    pd.DataFrame(tc_tests).to_csv(OUT / "tier2_module_tn_vs_control.csv", index=False)

    # Figure: LOO module scores by subtype
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_loo = loo_df[loo_df["module"] == "central_synaptic"]
    sns.boxplot(data=plot_loo, x="subtype", y="loo_module_score", ax=ax)
    ax.set_title("Tier 2: LOO central_synaptic module score by subtype")
    plt.tight_layout()
    plt.savefig(FIG / "tier2_loo_module_by_subtype.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"loo_tests": len(loo_tests), "split_half": split_half}


def tier3_cross_tissue(module_genes: dict) -> dict:
    logger.info("TIER 3: Cross-tissue mechanistic validation")
    results = {}

    # --- GSE197289 reference signatures ---
    ref = pd.read_csv(project_root / "results" / "aim1_enhanced" / "human_tg_integration" / "human_tg_reference_signatures.csv", index_col=0)
    ref.index = ref.index.astype(str).str.upper()
    canon_map = {
        "Neurons": ["Neurons", "Unknown_3", "Unknown_14"],
        "Schwann_cells": ["Schwann_cells", "Unknown_8"],
        "Satellite_glial_cells": ["Unknown_11"],
        "Immune_cells": ["Unknown_7"],
        "Fibroblasts": ["Fibroblasts", "Unknown_12", "Unknown_2"],
    }
    canon = {}
    for ct, cols in canon_map.items():
        present = [c for c in cols if c in ref.columns]
        if present:
            canon[ct] = ref[present].max(axis=1)

    tg_rows = []
    for mod, genes in module_genes.items():
        avail = [g for g in genes if g in ref.index]
        if not avail:
            continue
        means = {ct: float(canon[ct].reindex(avail).mean()) for ct in canon}
        top = max(means, key=means.get)
        tg_rows.append({
            "module": mod,
            "top_tg_celltype": top,
            "n_genes": len(avail),
            **{f"mean_{ct}": means[ct] for ct in means},
        })
    tg_df = pd.DataFrame(tg_rows)
    tg_df.to_csv(OUT / "tier3_module_tg_celltype_enrichment.csv", index=False)
    results["gse197289_modules"] = len(tg_rows)

    # --- IoN-CCI DE concordance with human TG bulk ---
    de_tg = pd.read_csv(project_root / "results" / "aim3" / "ion_cci" / "de_analysis" / "de_ion_cci_vs_sham_tg_with_symbols.csv")
    de_sp5c = pd.read_csv(project_root / "results" / "aim3" / "ion_cci" / "de_analysis" / "de_ion_cci_vs_sham_sp5c_with_symbols.csv")
    ortho = pd.read_csv(project_root / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")

    tg_bulk = pd.read_csv(project_root / "data" / "processed" / "human_tg_pmc6326384" / "human_tg_bulk_expression_processed.csv", index_col=0)
    tg_bulk.index = tg_bulk.index.astype(str).str.upper()
    blood = pd.read_csv(project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv", index_col=0)
    blood.index = blood.index.astype(str).str.upper()

    concord_rows = []
    for mod in MODULES:
        human_genes = module_genes[mod]
        mouse_up = set(de_tg[(de_tg["padj"] < 0.05) & (de_tg["log2FoldChange"] > 0.5)]["gene_symbol"].astype(str))
        mouse_down = set(de_tg[(de_tg["padj"] < 0.05) & (de_tg["log2FoldChange"] < -0.5)]["gene_symbol"].astype(str))
        if mod == "peripheral_ecm_schwann":
            mouse_set = mouse_up
        else:
            mouse_set = mouse_down
        ortho_sub = ortho[(ortho["module"] == mod) & (ortho["mouse_symbol"].isin(mouse_set))]
        hg = ortho_sub["human_symbol"].str.upper().unique()
        hg = [g for g in hg if g in tg_bulk.index and g in blood.index]
        if len(hg) < 5:
            continue
        tg_mean = tg_bulk.loc[hg].mean(axis=1).mean()
        blood_mean = blood.loc[hg].mean(axis=1).mean()
        concord_rows.append({
            "module": mod,
            "n_orthologs_overlap_de": len(hg),
            "mean_expr_human_tg_bulk": tg_mean,
            "mean_expr_human_blood": blood_mean,
            "tg_vs_blood_fold": tg_mean / (blood_mean + 1e-9),
            "interpretation": "Module genes expressed in human TG bulk (mechanistic plausibility)",
        })
    pd.DataFrame(concord_rows).to_csv(OUT / "tier3_ion_cci_tg_bulk_concordance.csv", index=False)
    results["pmc6326384"] = len(concord_rows)

    # --- IoN-CCI TG vs Sp5C tissue specificity ---
    spec_rows = []
    for mod in MODULES:
        if mod == "peripheral_ecm_schwann":
            sig_tg = de_tg[(de_tg["padj"] < 0.05) & (de_tg["log2FoldChange"] > 0.5)]
            direction = "up_in_TG"
        else:
            sig_tg = de_tg[(de_tg["padj"] < 0.05) & (de_tg["log2FoldChange"] < -0.5)]
            direction = "down_in_TG"
        genes_m = set(sig_tg["gene_symbol"].astype(str))
        sp5c_match = de_sp5c[de_sp5c["gene_symbol"].astype(str).isin(genes_m)]
        spec_rows.append({
            "module": mod,
            "n_sig_tg": len(sig_tg),
            "n_overlap_sp5c": len(sp5c_match),
            "mean_lfc_sp5c": sp5c_match["log2FoldChange"].mean() if len(sp5c_match) else np.nan,
            "direction": direction,
        })
    pd.DataFrame(spec_rows).to_csv(OUT / "tier3_ion_cci_tg_sp5c_specificity.csv", index=False)

    return results


def write_summary(t1, t2, t3):
    lines = [
        "# Endotype Validation Summary (Tiers 1–3)",
        "",
        "## Tier 1: External convergent validation",
        "- **GSE177034**: Persistent vs Resolved neuropathic pain (whole blood RNA-seq)",
        "- **Claim allowed**: IoN-CCI modules generalize to blood neuropathic pain biology",
        "- **Claim NOT allowed**: Independent replication of TN k=2 subtypes",
        "",
        "See: `tier1_gse177034_t0_baseline_tests.csv`, `tier1_gse177034_t1_followup_tests.csv`",
        "",
        "## Tier 2: Internal validation (GSE186505)",
        "- Leave-one-out module scoring",
        "- Split-half clustering stability",
        "- TN vs control gene panel + module direction checks",
        "",
        "See: `tier2_loo_subtype_tests.csv`, `tier2_split_half_stability.csv`, `tier2_gene_panel_tn_vs_control.csv`",
        "",
        "## Tier 3: Cross-tissue mechanistic validation",
        "- IoN-CCI modules → human TG cell types (GSE197289 reference)",
        "- IoN-CCI DE → human TG bulk expression (PMC6326384)",
        "- TG vs Sp5C tissue specificity",
        "",
        "See: `tier3_module_tg_celltype_enrichment.csv`, `tier3_ion_cci_tg_bulk_concordance.csv`",
    ]
    (OUT / "ENDOTYPE_VALIDATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    logger.info("=" * 70)
    logger.info("ENDOTYPE VALIDATION TIERS 1–3")
    logger.info("=" * 70)
    module_genes = load_frozen_human_module_genes()
    for mod, genes in module_genes.items():
        logger.info(f"  {mod}: {len(genes)} human genes")

    t1 = tier1_external_validation(module_genes)
    t2 = tier2_internal_validation(module_genes)
    t3 = tier3_cross_tissue(module_genes)
    write_summary(t1, t2, t3)
    logger.info(f"Complete → {OUT}")


if __name__ == "__main__":
    main()
