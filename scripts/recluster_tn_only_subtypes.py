"""
Recluster GSE186505 molecular subtypes using TN samples only.

Corrects the prior analysis that clustered all 20 samples (10 TN + 10 controls).
Generates k-selection metrics, control-composition audit table, updated subtype
assignments, DE results, pathway enrichment, and manuscript language draft.
"""

from __future__ import annotations

import shutil
import sys
import warnings
from datetime import datetime
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

warnings.filterwarnings("ignore")

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

EXPR_FILE = project_root / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv"
LEGACY_SUBTYPES = project_root / "results" / "molecular_subtypes.csv"
OUTPUT_DIR = project_root / "results" / "tn_only_subtyping"
DE_DIR = project_root / "results" / "subtype_characterization"
PATHWAY_DIR = project_root / "results" / "pathway_enrichment"
FIG_DIR = project_root / "results" / "ultra_hd_visualizations"


def load_sample_metadata() -> pd.DataFrame:
    """Build sample metadata from GSE186505 sample IDs."""
    expr = pd.read_csv(EXPR_FILE, index_col=0, nrows=1)
    samples = list(expr.columns)
    rows = []
    for s in samples:
        if s.startswith("Control"):
            diagnosis = "Control"
        elif s.startswith("TN"):
            diagnosis = "TN"
        else:
            diagnosis = "Unknown"
        rows.append({"sample_id": s, "diagnosis": diagnosis})
    return pd.DataFrame(rows)


def load_legacy_subtypes() -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_SUBTYPES)
    legacy = legacy.rename(columns={
        "subtype": "legacy_subtype",
        "subtype_name": "legacy_subtype_name",
    })
    return legacy


def select_k(data_reduced: np.ndarray, k_range: range) -> pd.DataFrame:
    """Evaluate clustering metrics across candidate k values."""
    metrics = []
    for k in k_range:
        if k >= len(data_reduced):
            continue
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(data_reduced)
        sil = silhouette_score(data_reduced, labels) if len(set(labels)) > 1 else np.nan
        ch = calinski_harabasz_score(data_reduced, labels) if len(set(labels)) > 1 else np.nan
        metrics.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": sil,
            "calinski_harabasz": ch,
        })
    return pd.DataFrame(metrics)


def choose_k(metrics: pd.DataFrame) -> int:
    """Pick k by highest silhouette among k=2..4 (n=10 TN samples)."""
    valid = metrics.dropna(subset=["silhouette"])
    if valid.empty:
        return 2
    best = valid.loc[valid["silhouette"].idxmax(), "k"]
    return int(best)


def cluster_tn_only(expr_tn: pd.DataFrame, k: int) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """PCA + K-means on TN-only expression matrix (genes x samples)."""
    data = expr_tn.T.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    n_components = min(data.shape[0] - 1, data.shape[1] - 1, 50)
    n_components = max(1, n_components)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(reduced)

    subtypes = pd.DataFrame({
        "sample_id": expr_tn.columns,
        "diagnosis": "TN",
        "subtype": labels,
        "subtype_name": [f"subtype_{l}" for l in labels],
    })
    return subtypes, pca, reduced


def build_control_composition_table(
    metadata: pd.DataFrame,
    legacy: pd.DataFrame,
    new_tn: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit table: legacy mixed clusters vs diagnosis; new TN-only clusters."""
    full = metadata.merge(legacy, on="sample_id", how="left")
    full = full.merge(
        new_tn[["sample_id", "subtype", "subtype_name"]].rename(columns={
            "subtype": "tn_only_subtype",
            "subtype_name": "tn_only_subtype_name",
        }),
        on="sample_id",
        how="left",
    )

    summary_rows = []
    for legacy_k in sorted(full["legacy_subtype"].dropna().unique()):
        subset = full[full["legacy_subtype"] == legacy_k]
        n_control = (subset["diagnosis"] == "Control").sum()
        n_tn = (subset["diagnosis"] == "TN").sum()
        summary_rows.append({
            "legacy_cluster": int(legacy_k),
            "legacy_cluster_label": f"subtype_{int(legacy_k)}",
            "n_total": len(subset),
            "n_control": n_control,
            "n_tn": n_tn,
            "pct_control": round(100 * n_control / len(subset), 1),
            "pct_tn": round(100 * n_tn / len(subset), 1),
            "interpretation": (
                "Mixed TN/control cluster (invalid for TN subtype discovery)"
                if n_control > 0 and n_tn > 0
                else ("Control-enriched" if n_control > n_tn else "TN-enriched")
            ),
        })
    summary = pd.DataFrame(summary_rows)

    new_summary_rows = []
    for new_k in sorted(new_tn["subtype"].unique()):
        tn_ids = new_tn[new_tn["subtype"] == new_k]["sample_id"].tolist()
        new_summary_rows.append({
            "tn_only_cluster": int(new_k),
            "tn_only_cluster_label": f"subtype_{int(new_k)}",
            "n_tn": len(tn_ids),
            "tn_sample_ids": ", ".join(sorted(tn_ids)),
        })
    new_summary = pd.DataFrame(new_summary_rows)

    return full, summary, new_summary


def differential_expression_tn_only(
    expr_tn: pd.DataFrame,
    subtypes: pd.DataFrame,
    output_dir: Path,
) -> dict[int, pd.DataFrame]:
    """Subtype vs other TN samples (Welch t-test + Benjamini-Hochberg)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    de_results = {}

    for subtype in sorted(subtypes["subtype"].unique()):
        in_samples = subtypes[subtypes["subtype"] == subtype]["sample_id"].tolist()
        out_samples = subtypes[subtypes["subtype"] != subtype]["sample_id"].tolist()
        if len(in_samples) < 2 or len(out_samples) < 2:
            logger.warning(f"Skipping DE for subtype {subtype}: insufficient TN samples")
            continue

        logger.info(f"DE subtype {subtype}: n={len(in_samples)} vs n={len(out_samples)} (TN-only)")
        rows = []
        for gene in expr_tn.index:
            a = expr_tn.loc[gene, in_samples].values.astype(float)
            b = expr_tn.loc[gene, out_samples].values.astype(float)
            if np.std(a) == 0 and np.std(b) == 0:
                continue
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            mean_a, mean_b = np.mean(a), np.mean(b)
            log2fc = np.log2((mean_a + 1e-10) / (mean_b + 1e-10))
            rows.append({
                "gene": gene,
                "subtype": int(subtype),
                "mean_subtype": mean_a,
                "mean_other": mean_b,
                "log2fc": log2fc,
                "t_statistic": t_stat,
                "p_value": p_val,
            })

        de_df = pd.DataFrame(rows)
        if de_df.empty:
            continue

        # Benjamini-Hochberg FDR
        p = de_df["p_value"].values
        order = np.argsort(p)
        ranked = p[order]
        m = len(ranked)
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        padj = np.empty_like(q)
        padj[order] = q
        de_df["p_adj"] = padj
        de_df["abs_log2fc"] = de_df["log2fc"].abs()
        de_df = de_df.sort_values(["p_adj", "abs_log2fc"], ascending=[True, False])

        de_results[int(subtype)] = de_df
        de_df.to_csv(output_dir / f"de_subtype_{int(subtype)}.csv", index=False)
        sig = de_df[(de_df["p_adj"] < 0.05) & (de_df["abs_log2fc"] > 1)]
        logger.info(f"  Significant genes (FDR<0.05, |log2FC|>1): {len(sig)}")

    return de_results


def pathway_enrichment(de_results: dict[int, pd.DataFrame], output_dir: Path) -> None:
    """Run gseapy enrichr for up/down regulated genes per subtype."""
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed; skipping pathway enrichment")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_sets = ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2021"]

    for subtype, de_df in de_results.items():
        for direction, mask in [
            ("upregulated", (de_df["p_adj"] < 0.05) & (de_df["log2fc"] > 1)),
            ("downregulated", (de_df["p_adj"] < 0.05) & (de_df["log2fc"] < -1)),
        ]:
            genes = de_df.loc[mask, "gene"].head(200).tolist()
            if len(genes) < 5:
                continue
            for gs in gene_sets:
                try:
                    enr = gp.enrichr(
                        gene_list=genes,
                        gene_sets=[gs],
                        organism="Human",
                        outdir=None,
                        verbose=False,
                    )
                    if enr is not None and hasattr(enr, "results") and len(enr.results) > 0:
                        out = output_dir / f"subtype_{subtype}_{direction}_{gs}_enrichment.csv"
                        enr.results.to_csv(out, index=False)
                        logger.info(f"  Pathways {subtype} {direction} {gs}: {len(enr.results)} terms")
                except Exception as exc:
                    logger.debug(f"Pathway error {subtype}/{direction}/{gs}: {exc}")


def case_control_de(expr: pd.DataFrame, metadata: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """TN vs Control DE (separate from subtype analysis)."""
    tn = metadata[metadata["diagnosis"] == "TN"]["sample_id"].tolist()
    ctrl = metadata[metadata["diagnosis"] == "Control"]["sample_id"].tolist()
    rows = []
    for gene in expr.index:
        a = expr.loc[gene, tn].values.astype(float)
        b = expr.loc[gene, ctrl].values.astype(float)
        if np.std(a) == 0 and np.std(b) == 0:
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        mean_tn, mean_ctrl = np.mean(a), np.mean(b)
        log2fc = np.log2((mean_tn + 1e-10) / (mean_ctrl + 1e-10))
        rows.append({
            "gene": gene,
            "mean_tn": mean_tn,
            "mean_control": mean_ctrl,
            "log2fc": log2fc,
            "t_statistic": t_stat,
            "p_value": p_val,
        })
    de_df = pd.DataFrame(rows)
    p = de_df["p_value"].values
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    padj = np.empty_like(q)
    padj[order] = np.clip(q, 0, 1)
    de_df["p_adj"] = padj
    de_df["abs_log2fc"] = de_df["log2fc"].abs()
    de_df = de_df.sort_values(["p_adj", "abs_log2fc"])
    output_dir.mkdir(parents=True, exist_ok=True)
    de_df.to_csv(output_dir / "tn_vs_control_de.csv", index=False)
    sig = de_df[(de_df["p_adj"] < 0.05)]
    logger.info(f"TN vs Control DE: {len(sig)} genes at FDR<0.05")
    return de_df


def plot_k_selection(metrics: pd.DataFrame, chosen_k: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(metrics["k"], metrics["inertia"], "o-", color="#4472C4")
    axes[0].axvline(chosen_k, color="red", ls="--", label=f"Selected k={chosen_k}")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow (Inertia)")
    axes[0].legend()

    axes[1].plot(metrics["k"], metrics["silhouette"], "o-", color="#ED7D31")
    axes[1].axvline(chosen_k, color="red", ls="--")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette")

    axes[2].plot(metrics["k"], metrics["calinski_harabasz"], "o-", color="#70AD47")
    axes[2].axvline(chosen_k, color="red", ls="--")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Calinski-Harabasz")
    axes[2].set_title("Calinski-Harabasz")

    plt.tight_layout()
    fig.savefig(output_dir / "k_selection_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_clusters(
    reduced: np.ndarray,
    subtypes: pd.DataFrame,
    pca: PCA,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette("Set2", n_colors=subtypes["subtype"].nunique())
    for i, st in enumerate(sorted(subtypes["subtype"].unique())):
        mask = (subtypes["subtype"] == st).values
        idx = subtypes.index[mask]
        y = reduced[mask, 1] if reduced.shape[1] > 1 else np.zeros(mask.sum())
        ax.scatter(
            reduced[mask, 0],
            y,
            c=[colors[i]],
            s=120,
            label=f"Subtype {st} (n={(mask).sum()})",
            edgecolors="black",
            linewidths=0.5,
        )
        for j in idx:
            ax.annotate(
                subtypes.loc[j, "sample_id"],
                (reduced[j, 0], reduced[j, 1] if reduced.shape[1] > 1 else 0),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}% var)" if len(var) > 1 else "PC2")
    ax.set_title("TN-only molecular subtypes (PCA)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "tn_only_pca_subtypes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_legacy_vs_tn_only(full_audit: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar: control vs TN composition in legacy clusters."""
    legacy = full_audit.dropna(subset=["legacy_subtype"])
    counts = legacy.groupby(["legacy_subtype", "diagnosis"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["Control", "TN"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#95A5A6", "#E74C3C"])
    ax.set_xlabel("Legacy mixed-cohort cluster")
    ax.set_ylabel("Sample count")
    ax.set_title("Legacy clusters contained both controls and TN samples")
    ax.set_xticklabels([f"Subtype {int(i)}" for i in counts.index], rotation=0)
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    fig.savefig(output_dir / "legacy_cluster_diagnosis_composition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def regenerate_subtype_figures(subtypes: pd.DataFrame, de_results: dict) -> None:
    """Regenerate core publication figures using TN-only subtypes."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    deconv_file = project_root / "results" / "blood_deconvolution_improved" / "corrected_deconvolution_proportions.csv"
    if not deconv_file.exists():
        deconv_file = project_root / "results" / "blood_deconvolution_improved" / "improved_deconvolution_proportions.csv"
    if not deconv_file.exists():
        logger.warning("Deconvolution file not found; skipping figure regeneration")
        return

    deconv = pd.read_csv(deconv_file, index_col=0)
    tn_ids = subtypes["sample_id"].tolist()
    deconv_tn = deconv.loc[[s for s in tn_ids if s in deconv.index]]

    # Figure: subtype distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = subtypes["subtype"].value_counts().sort_index()
    bars = ax.bar([f"Subtype {i}" for i in counts.index], counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.set_ylabel("TN samples (n=10)")
    ax.set_title("TN-only molecular subtype distribution")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(v), ha="center")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure_tn_only_subtype_distribution.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Figure: deconvolution heatmap (TN only)
    if not deconv_tn.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(deconv_tn.T, cmap="viridis", ax=ax, cbar_kws={"label": "Proportion"})
        ax.set_title("Cell type deconvolution — TN samples only (n=10)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "figure_tn_only_deconvolution_heatmap.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Figure: volcano for best-powered subtype
    if de_results:
        best_st = max(de_results.keys(), key=lambda s: (
            (de_results[s]["p_adj"] < 0.05).sum()
        ))
        de = de_results[best_st]
        fig, ax = plt.subplots(figsize=(8, 7))
        sig = (de["p_adj"] < 0.05) & (de["abs_log2fc"] > 1)
        ax.scatter(de.loc[~sig, "log2fc"], -np.log10(de.loc[~sig, "p_adj"].clip(1e-300)), c="gray", s=12, alpha=0.5)
        up = sig & (de["log2fc"] > 0)
        down = sig & (de["log2fc"] < 0)
        ax.scatter(de.loc[up, "log2fc"], -np.log10(de.loc[up, "p_adj"].clip(1e-300)), c="red", s=20)
        ax.scatter(de.loc[down, "log2fc"], -np.log10(de.loc[down, "p_adj"].clip(1e-300)), c="blue", s=20)
        ax.axhline(-np.log10(0.05), ls="--", color="black", lw=0.8)
        ax.axvline(-1, ls="--", color="black", lw=0.8)
        ax.axvline(1, ls="--", color="black", lw=0.8)
        ax.set_xlabel("log2 fold change (TN subtype vs other TN)")
        ax.set_ylabel("-log10(FDR)")
        ax.set_title(f"Volcano plot — TN Subtype {best_st} vs other TN (n=10 discovery cohort)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "figure_tn_only_volcano_best_subtype.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def write_manuscript_draft(
    metadata: pd.DataFrame,
    k_metrics: pd.DataFrame,
    chosen_k: int,
    legacy_summary: pd.DataFrame,
    new_summary: pd.DataFrame,
    subtypes: pd.DataFrame,
    de_results: dict,
    output_dir: Path,
) -> None:
    counts = subtypes["subtype"].value_counts().sort_index()
    count_str = ", ".join([f"Subtype {i}: n={counts[i]}" for i in counts.index])
    sil_best = k_metrics.loc[k_metrics["k"] == chosen_k, "silhouette"].values
    sil_val = f"{sil_best[0]:.3f}" if len(sil_best) else "NA"

    legacy_lines = []
    for _, row in legacy_summary.iterrows():
        legacy_lines.append(
            f"- Legacy mixed cluster {int(row['legacy_cluster'])}: "
            f"{int(row['n_control'])} controls + {int(row['n_tn'])} TN "
            f"({row['pct_control']:.0f}% control)"
        )

    de_lines = []
    for st, de in sorted(de_results.items()):
        sig = de[(de["p_adj"] < 0.05) & (de["abs_log2fc"] > 1)]
        de_lines.append(f"- TN Subtype {st}: {len(sig)} genes (FDR<0.05, |log2FC|>1)")

    text = f"""# TN-Only Molecular Subtyping — Corrected Manuscript Language

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Correction:** Prior analysis incorrectly clustered all 20 GSE186505 samples (10 TN + 10 controls) and described them as 20 TN patients. Subtyping has been repeated on **TN samples only (n=10)**.

---

## Methods (corrected text)

### Study cohort and discovery dataset

Whole-blood RNA-seq data were obtained from GEO accession **GSE186505**, comprising **10 patients with trigeminal neuralgia (TN) and 10 pain-free healthy controls** (20 samples total). This cohort was used as the primary discovery dataset for blood-based molecular analyses.

### Molecular subtyping (TN-only)

Unsupervised molecular subtyping was performed **only on TN patient samples (n=10)**. Healthy control samples (n=10) were **excluded from clustering** and retained as an independent reference group for case–control differential expression. Expression values were log-transformed and standardized per gene. Principal component analysis (PCA) was applied prior to clustering, retaining up to min(n−1, 50) components. The optimal cluster number (k) was evaluated for k=2–4 using the elbow method (within-cluster sum of squares), silhouette coefficient, and Calinski–Harabasz index; **k={chosen_k}** was selected (silhouette={sil_val}). K-means clustering (random_state=42, n_init=20) was then applied in PCA space to assign TN samples to molecular subtypes.

For each TN subtype, differential expression was computed as **subtype versus remaining TN samples** (Welch's t-test), with Benjamini–Hochberg false discovery rate (FDR) correction. Separately, **TN versus control** differential expression was performed across all 10 TN and 10 control samples to characterize disease-associated transcriptional changes independent of subtype structure.

### Cell type deconvolution

Cell type proportions were estimated from bulk blood expression using reference-based deconvolution (11 immune cell types). Deconvolution was run on all 20 samples; subtype-stratified summaries were computed **using TN-only subtype labels**.

---

## Results (corrected text)

### Cohort composition

The GSE186505 discovery cohort included **10 TN patients and 10 healthy controls**. An initial exploratory analysis had incorrectly included controls in unsupervised subtype clustering, yielding three clusters of 10, 3, and 7 samples that **mixed TN and control individuals** (see Control Composition Audit). This did not represent TN molecular subtypes and was not used in the corrected analysis.

**Legacy mixed-cluster composition (superseded):**
{chr(10).join(legacy_lines)}

### TN-only molecular subtypes

After restricting clustering to TN samples only (n=10), we identified **{chosen_k} molecular subtypes**: {count_str}. Given the limited sample size, these subtypes should be interpreted as **hypothesis-generating** and require validation in independent cohorts.

**TN-only subtype assignments:**
{chr(10).join([f"- Subtype {int(r['tn_only_cluster'])}: {r['tn_sample_ids']}" for _, r in new_summary.iterrows()])}

### Subtype-specific differential expression (TN-only contrasts)

{chr(10).join(de_lines)}

Controls were not included in subtype definition or subtype-vs-subtype contrasts. Case–control differential expression (TN vs healthy controls, n=10 vs n=10) is reported separately in `tn_vs_control_de.csv`.

---

## Figure / table legend updates

- Replace "n=20 TN patients" with **"n=10 TN patients and n=10 healthy controls (GSE186505)"**.
- Subtype bar plots and heatmaps: **TN samples only (n=10)** unless explicitly showing the control reference panel.
- Volcano plots: specify contrast as **"TN subtype X vs other TN samples"**, not vs mixed cohort.

---

## Files updated

| File | Description |
|------|-------------|
| `results/molecular_subtypes.csv` | TN-only subtype assignments (primary) |
| `results/molecular_subtypes_mixed_cohort_legacy.csv` | Backup of incorrect mixed clustering |
| `results/gse186505_sample_metadata.csv` | All 20 samples with diagnosis labels |
| `results/tn_only_subtyping/control_composition_audit.csv` | Per-sample legacy vs TN-only mapping |
| `results/tn_only_subtyping/legacy_cluster_summary.csv` | Control/TN counts in legacy clusters |
| `results/subtype_characterization/de_subtype_*.csv` | TN-only DE (regenerated) |
| `results/tn_only_subtyping/tn_vs_control_de.csv` | Separate TN vs control DE |
"""
    (output_dir / "MANUSCRIPT_CORRECTED_LANGUAGE.md").write_text(text, encoding="utf-8")
    logger.info(f"Wrote manuscript draft to {output_dir / 'MANUSCRIPT_CORRECTED_LANGUAGE.md'}")


def main() -> None:
    logger.info("=" * 70)
    logger.info("TN-ONLY MOLECULAR SUBTYPING CORRECTION")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Backup legacy subtypes
    legacy_backup = project_root / "results" / "molecular_subtypes_mixed_cohort_legacy.csv"
    if LEGACY_SUBTYPES.exists() and not legacy_backup.exists():
        shutil.copy2(LEGACY_SUBTYPES, legacy_backup)
        logger.info(f"Backed up legacy subtypes to {legacy_backup}")

    legacy = load_legacy_subtypes()
    metadata = load_sample_metadata()
    metadata.to_csv(project_root / "results" / "gse186505_sample_metadata.csv", index=False)

    expr = pd.read_csv(EXPR_FILE, index_col=0)
    tn_samples = metadata[metadata["diagnosis"] == "TN"]["sample_id"].tolist()
    expr_tn = expr[[c for c in tn_samples if c in expr.columns]]
    logger.info(f"TN expression matrix: {expr_tn.shape[0]} genes x {expr_tn.shape[1]} samples")

    # k selection
    data = expr_tn.T
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    n_comp = max(1, min(data.shape[0] - 1, data.shape[1] - 1, 50))
    pca_tmp = PCA(n_components=n_comp, random_state=42)
    reduced_tmp = pca_tmp.fit_transform(scaled)
    k_range = range(2, min(5, len(tn_samples)))
    k_metrics = select_k(reduced_tmp, k_range)
    chosen_k = choose_k(k_metrics)
    k_metrics["selected"] = k_metrics["k"] == chosen_k
    k_metrics.to_csv(OUTPUT_DIR / "k_selection_metrics.csv", index=False)
    logger.info(f"Selected k={chosen_k}")
    logger.info(f"\n{k_metrics.to_string(index=False)}")

    # Cluster
    subtypes, pca, reduced = cluster_tn_only(expr_tn, chosen_k)

    # Audit tables
    full_audit, legacy_summary, new_summary = build_control_composition_table(metadata, legacy, subtypes)
    full_audit.to_csv(OUTPUT_DIR / "control_composition_audit.csv", index=False)
    legacy_summary.to_csv(OUTPUT_DIR / "legacy_cluster_summary.csv", index=False)
    new_summary.to_csv(OUTPUT_DIR / "tn_only_cluster_summary.csv", index=False)

    # Update primary subtype file (TN only)
    subtypes[["sample_id", "subtype", "subtype_name"]].to_csv(LEGACY_SUBTYPES, index=False)
    subtypes.to_csv(OUTPUT_DIR / "molecular_subtypes_tn_only.csv", index=False)

    # DE + pathways
    de_results = differential_expression_tn_only(expr_tn, subtypes, DE_DIR)
    pathway_enrichment(de_results, PATHWAY_DIR)
    case_control_de(expr, metadata, OUTPUT_DIR)

    # Plots
    plot_k_selection(k_metrics, chosen_k, OUTPUT_DIR)
    plot_pca_clusters(reduced, subtypes.reset_index(drop=True), pca, OUTPUT_DIR)
    plot_legacy_vs_tn_only(full_audit, OUTPUT_DIR)
    regenerate_subtype_figures(subtypes, de_results)

    write_manuscript_draft(
        metadata, k_metrics, chosen_k, legacy_summary, new_summary, subtypes, de_results, OUTPUT_DIR
    )

    logger.info("\n" + "=" * 70)
    logger.info("TN-ONLY RECLUSTERING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Subtypes: {subtypes['subtype'].value_counts().sort_index().to_dict()}")
    logger.info(f"Primary output: {LEGACY_SUBTYPES}")
    logger.info(f"Manuscript draft: {OUTPUT_DIR / 'MANUSCRIPT_CORRECTED_LANGUAGE.md'}")


if __name__ == "__main__":
    main()
