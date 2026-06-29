"""
k=3 sensitivity analysis for TN-only molecular subtyping (supplementary materials).

Primary analysis uses k=2 (data-driven). This script clusters TN samples at k=3,
generates DE, figures, and comparison tables for supplementary reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

EXPR_FILE = project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv"
OUT_DIR = project_root / "results" / "tn_only_subtyping" / "k3_sensitivity"
FIG_DIR = project_root / "results" / "ultra_hd_visualizations"


def cluster_k3(expr_tn: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    data = expr_tn.T
    scaled = StandardScaler().fit_transform(data)
    n_comp = max(1, min(data.shape[0] - 1, data.shape[1] - 1, 50))
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(scaled)
    model = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = model.fit_predict(reduced)
    sil = silhouette_score(reduced, labels)
    ch = calinski_harabasz_score(reduced, labels)
    subtypes = pd.DataFrame({
        "sample_id": expr_tn.columns,
        "diagnosis": "TN",
        "subtype_k3": labels,
        "subtype_name": [f"subtype_k3_{l}" for l in labels],
    })
    metrics = {"k": 3, "silhouette": sil, "calinski_harabasz": ch, "inertia": model.inertia_}
    return subtypes, reduced, pca, metrics


def de_analysis(expr_tn: pd.DataFrame, subtypes: pd.DataFrame, label_col: str = "subtype_k3") -> dict[int, pd.DataFrame]:
    results = {}
    for subtype in sorted(subtypes[label_col].unique()):
        in_s = subtypes[subtypes[label_col] == subtype]["sample_id"].tolist()
        out_s = subtypes[subtypes[label_col] != subtype]["sample_id"].tolist()
        rows = []
        for gene in expr_tn.index:
            a = expr_tn.loc[gene, in_s].values.astype(float)
            b = expr_tn.loc[gene, out_s].values.astype(float)
            if np.std(a) == 0 and np.std(b) == 0:
                continue
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            mean_a, mean_b = np.mean(a), np.mean(b)
            log2fc = np.log2((mean_a + 1e-10) / (mean_b + 1e-10))
            rows.append({
                "gene": gene, "subtype_k3": int(subtype),
                "mean_subtype": mean_a, "mean_other": mean_b,
                "log2fc": log2fc, "t_statistic": t_stat, "p_value": p_val,
            })
        de_df = pd.DataFrame(rows)
        p = de_df["p_value"].values
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        q = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
        padj = np.empty_like(q)
        padj[order] = np.clip(q, 0, 1)
        de_df["p_adj"] = padj
        de_df["abs_log2fc"] = de_df["log2fc"].abs()
        de_df = de_df.sort_values(["p_adj", "abs_log2fc"])
        results[int(subtype)] = de_df
        de_df.to_csv(OUT_DIR / f"de_subtype_k3_{int(subtype)}.csv", index=False)
        sig = de_df[(de_df["p_adj"] < 0.05) & (de_df["abs_log2fc"] > 1)]
        logger.info(f"k=3 Subtype {subtype}: n={len(in_s)} vs {len(out_s)}; sig genes={len(sig)}")
    return results


def plot_pca_k3(reduced, subtypes, pca, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", 3)
    for i, st in enumerate(sorted(subtypes["subtype_k3"].unique())):
        mask = (subtypes["subtype_k3"] == st).values
        y = reduced[mask, 1] if reduced.shape[1] > 1 else np.zeros(mask.sum())
        ax.scatter(reduced[mask, 0], y, c=[colors[i]], s=120, label=f"k=3 Subtype {st} (n={mask.sum()})",
                   edgecolors="black", linewidths=0.5)
        for j in subtypes.index[mask]:
            ax.annotate(subtypes.loc[j, "sample_id"], (reduced[j, 0], reduced[j, 1] if reduced.shape[1] > 1 else 0),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)" if len(var) > 1 else "PC2")
    ax.set_title("Supplementary Figure S12. Sensitivity analysis: k=3 K-means (TN-only, n=10)\nNot the primary subtyping solution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_k_comparison():
    primary = pd.read_csv(project_root / "results" / "tn_only_subtyping" / "k_selection_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = primary["k"].astype(str)
    ax.bar(x, primary["silhouette"], color=["#4472C4" if s else "#B4C7E7" for s in primary["selected"]])
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Supplementary Figure S11. Cluster number selection (TN-only, n=10)")
    for i, (_, row) in enumerate(primary.iterrows()):
        label = " (primary)" if row["selected"] else ""
        ax.text(i, row["silhouette"] + 0.005, f"{row['silhouette']:.3f}{label}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "supplementary_k_selection_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "k_selection_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano_k3(de_results: dict):
    n = len(de_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for idx, (st, de) in enumerate(sorted(de_results.items())):
        sig = (de["p_adj"] < 0.05) & (de["abs_log2fc"] > 1)
        ax = axes[idx]
        ax.scatter(de.loc[~sig, "log2fc"], -np.log10(de.loc[~sig, "p_adj"].clip(1e-300)), c="gray", s=12, alpha=0.5)
        up = sig & (de["log2fc"] > 0)
        down = sig & (de["log2fc"] < 0)
        ax.scatter(de.loc[up, "log2fc"], -np.log10(de.loc[up, "p_adj"].clip(1e-300)), c="red", s=18)
        ax.scatter(de.loc[down, "log2fc"], -np.log10(de.loc[down, "p_adj"].clip(1e-300)), c="blue", s=18)
        ax.axhline(-np.log10(0.05), ls="--", color="black", lw=0.8)
        ax.axvline(-1, ls="--", color="black", lw=0.8)
        ax.axvline(1, ls="--", color="black", lw=0.8)
        ax.set_title(f"k=3 Subtype {st} vs other TN")
        ax.set_xlabel("log2 FC")
        ax.set_ylabel("-log10(FDR)")
    plt.suptitle("Supplementary Figure S13. Sensitivity analysis: differential expression under k=3 clustering", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "supplementary_volcano_k3_sensitivity.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / "volcano_k3_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(project_root / "results" / "gse186505_sample_metadata.csv")
    tn_ids = metadata[metadata["diagnosis"] == "TN"]["sample_id"].tolist()
    expr = pd.read_csv(EXPR_FILE, index_col=0)
    expr_tn = expr[[c for c in tn_ids if c in expr.columns]]

    subtypes_k3, reduced, pca, metrics_k3 = cluster_k3(expr_tn)
    subtypes_k3.to_csv(OUT_DIR / "molecular_subtypes_k3_sensitivity.csv", index=False)

    # Cluster size summary
    counts = subtypes_k3["subtype_k3"].value_counts().sort_index()
    summary = pd.DataFrame([{
        "analysis": "primary_k2",
        **{f"subtype_{i}_n": c for i, c in pd.read_csv(project_root / "results" / "molecular_subtypes.csv")["subtype"].value_counts().sort_index().items()},
    }, {
        "analysis": "sensitivity_k3",
        **{f"subtype_{i}_n": counts.get(i, 0) for i in range(3)},
    }])
    # Simpler assignment comparison table
    primary = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")
    compare = primary.merge(
        subtypes_k3[["sample_id", "subtype_k3"]],
        on="sample_id", how="left",
    ).rename(columns={"subtype": "primary_k2_subtype", "subtype_k3": "sensitivity_k3_subtype"})
    compare.to_csv(OUT_DIR / "k2_vs_k3_assignment_comparison.csv", index=False)

    cluster_summary = []
    for st in sorted(subtypes_k3["subtype_k3"].unique()):
        ids = subtypes_k3[subtypes_k3["subtype_k3"] == st]["sample_id"].tolist()
        cluster_summary.append({
            "k3_cluster": int(st),
            "n_tn": len(ids),
            "sample_ids": ", ".join(sorted(ids)),
        })
    pd.DataFrame(cluster_summary).to_csv(OUT_DIR / "k3_cluster_summary.csv", index=False)

    pd.DataFrame([metrics_k3]).to_csv(OUT_DIR / "k3_clustering_metrics.csv", index=False)

    de_results = de_analysis(expr_tn, subtypes_k3)
    plot_pca_k3(reduced, subtypes_k3.reset_index(drop=True), pca, OUT_DIR / "pca_k3_subtypes.png")
    plot_pca_k3(reduced, subtypes_k3.reset_index(drop=True), pca, FIG_DIR / "supplementary_pca_k3_sensitivity.png")
    plot_k_comparison()
    plot_volcano_k3(de_results)

    # Narrative summary for supplementary (no error framing)
    primary_metrics = pd.read_csv(project_root / "results" / "tn_only_subtyping" / "k_selection_metrics.csv")
    k2_sil = primary_metrics.loc[primary_metrics["selected"], "silhouette"].iloc[0]
    text = f"""# k=3 Sensitivity Analysis (Supplementary)

## Overview
Primary molecular subtyping used k=2 (silhouette={k2_sil:.3f}). As a sensitivity analysis, we repeated K-means clustering on the same TN-only discovery cohort (n=10) with k=3 (silhouette={metrics_k3['silhouette']:.3f}).

## k=3 cluster sizes
{chr(10).join([f"- Subtype {int(r['k3_cluster'])}: n={int(r['n_tn'])} ({r['sample_ids']})" for _, r in pd.DataFrame(cluster_summary).iterrows()])}

## DE summary (k=3, TN subtype vs other TN)
{chr(10).join([f"- k=3 Subtype {st}: {(de[(de['p_adj']<0.05)&(de['abs_log2fc']>1)]).shape[0]} genes (FDR<0.05, |log2FC|>1)" for st, de in de_results.items()])}

## Figures
- `supplementary_k_selection_comparison.png` — silhouette by k
- `supplementary_pca_k3_sensitivity.png` — PCA with k=3 labels
- `supplementary_volcano_k3_sensitivity.png` — DE volcano plots (k=3)
"""
    (OUT_DIR / "SUPPLEMENTARY_K3_SENSITIVITY.md").write_text(text, encoding="utf-8")
    logger.info(f"k=3 sensitivity analysis complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
