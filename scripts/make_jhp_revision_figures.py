"""
Figures for the JHP revision (main Figures 1-5, Supplementary Figures S1-S6).
Reads results/jhp_revision/*; writes results/jhp_revision/figures/*.png (600 dpi).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "jhp_revision"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ION = ROOT / "results" / "aim3" / "ion_cci" / "de_analysis"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "figure.dpi": 150,
})
C_S0, C_S1, C_CTRL = "#0072B2", "#D55E00", "#8C8C8C"
C_TG, C_SP = "#009E73", "#CC79A7"
GROUP_COL = {"Subtype 0": C_S0, "Subtype 1": C_S1, "Control": C_CTRL}
S = json.loads((R / "REVISION_SUMMARY.json").read_text())
ND = np.load(R / "null_draws.npz")


def panel(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "P < 0.0001"
    if p < 0.001:
        return f"P = {p:.4f}"
    return f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
def figure1():
    fig = plt.figure(figsize=(6.3, 5.2))
    gs = GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.55)
    pcs = pd.read_csv(R / "B7_TN_PC_scores.csv", index_col=0)
    proj = pd.read_csv(R / "B5_controls_projected_onto_TN_partition.csv")
    ve = S["B7_PC_variance_explained"]
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(proj.PC1, proj.PC2, s=26, facecolor="white", edgecolor=C_CTRL, lw=0.9, label="Controls (projected)", zorder=2)
    for st, col in [(0, C_S0), (1, C_S1)]:
        d = pcs[pcs.subtype == st]
        ax.scatter(d.PC1, d.PC2, s=30, color=col, edgecolor="black", lw=0.4, label=f"TN partition {st} (n={len(d)})", zorder=3)
        for sid, r in d.iterrows():
            ax.annotate(sid, (r.PC1, r.PC2), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({ve[0]*100:.1f}% of TN variance)")
    ax.set_ylabel(f"PC2 ({ve[1]*100:.1f}%)")
    ax.set_title("TN PCA (submitted pipeline), controls projected")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.2, columnspacing=0.8)
    panel(ax, "A", x=-0.24)

    ax = fig.add_subplot(gs[0, 1])
    nb = ND["sigclust_best_silhouette"]
    n2 = ND["sigclust_silhouette_k2"]
    bins = np.linspace(0, 0.6, 41)
    ax.hist(n2, bins=bins, color="#BBBBBB", edgecolor="white", lw=0.3, label="Null, k = 2")
    ax.hist(nb, bins=bins, histtype="step", color="black", lw=0.8, label="Null, best of k = 2-4")
    obs = 0.2237
    ax.axvline(obs, color=C_S1, lw=1.4)
    ax.text(0.98, 0.62, f"Observed k = 2 silhouette = {obs:.3f}\n"
            f"{fmt_p(S['B2_p_best_silhouette'])} vs best-k null\nSigClust cluster index {fmt_p(S['B2_sigclust_p_cluster_index'])}",
            transform=ax.transAxes, fontsize=5.8, va="top", ha="right", color=C_S1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xlabel("Average silhouette width")
    ax.set_ylabel("Null datasets (of 2,000)")
    ax.set_title("No-cluster (single Gaussian) null")
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "B", x=-0.24)

    ax = fig.add_subplot(gs[1, 0])
    gap = pd.read_csv(R / "B2_gap_statistic_k1_4.csv")
    gapl = pd.read_csv(R / "B2_gap_statistic_k1_4_log2.csv")
    ax.errorbar(gap.k - 0.05, gap.gap, yerr=gap.s_k, marker="o", ms=4, color="black", capsize=2, lw=0.8, label="z(FPKM) (submitted)")
    ax.errorbar(gapl.k + 0.05, gapl.gap, yerr=gapl.s_k, marker="s", ms=3.5, color="#666666", capsize=2, lw=0.8, ls="--", label="z(log2 FPKM+1)")
    ax.scatter([1], [gap.gap[0]], s=90, facecolor="none", edgecolor=C_S1, lw=1.2, zorder=5)
    ax.annotate("Tibshirani 1-SE rule\nselects k = 1", (1, gap.gap[0]), xytext=(1.7, gap.gap.max() + 0.05), fontsize=6,
                color=C_S1, arrowprops=dict(arrowstyle="-", color=C_S1, lw=0.6))
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Number of clusters, k")
    ax.set_ylabel("Gap statistic (± s$_k$)")
    ax.set_title("Gap statistic including k = 1")
    ax.legend(frameon=False, loc="lower left")
    panel(ax, "C", x=-0.24)

    ax = fig.add_subplot(gs[1, 1])
    C = pd.read_csv(R / "B3_consensus_matrix_subsampling.csv", index_col=0)
    order = pcs.sort_values(["subtype", "PC1"]).index.tolist()
    C = C.loc[order, order]
    im = ax.imshow(C.values, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(order, rotation=90, fontsize=5.5)
    ax.set_yticklabels(order, fontsize=5.5)
    for i, s in enumerate(order):
        col = C_S0 if pcs.loc[s, "subtype"] == 0 else C_S1
        ax.add_patch(plt.Rectangle((-1.6, i - 0.5), 0.6, 1, color=col, clip_on=False))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Co-clustering frequency", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    jac = S["B3_jaccard"]
    ax.set_title("Consensus over all 45 subsamples (8 of 10)")
    ax.text(0.5, -0.33, f"Leave-one-out {S['B3_LOO_n_identical']}/10 identical\nJaccard: subsampling "
            f"{jac[0]['subsample_8of10_mean_jaccard']:.2f}/{jac[1]['subsample_8of10_mean_jaccard']:.2f}, "
            f"bootstrap {jac[0]['bootstrap_mean_jaccard']:.2f}/{jac[1]['bootstrap_mean_jaccard']:.2f}",
            transform=ax.transAxes, fontsize=5.8, ha="center", va="top")
    panel(ax, "D", x=-0.36)
    save(fig, "Figure1_clusterability_stability")


# ---------------------------------------------------------------------------
def figure2():
    fig = plt.figure(figsize=(6.3, 5.3))
    gs = GridSpec(2, 2, figure=fig, wspace=0.85, hspace=0.62)
    g = pd.read_csv(R / "B10_globin_fraction_by_sample.csv", index_col=0)
    ax = fig.add_subplot(gs[0, :])
    cols = [GROUP_COL[x] for x in g.group]
    ax.bar(range(len(g)), g.globin_fraction, color=cols, edgecolor="black", lw=0.3)
    for i, (sid, r) in enumerate(g.iterrows()):
        if r.group == "Control":
            ax.text(i, r.globin_fraction + 0.01, "1" if "Subtype 1" in r.assigned_or_subtype else "0",
                    ha="center", fontsize=5.5, color=C_S1 if "Subtype 1" in r.assigned_or_subtype else C_S0)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=5.8)
    ax.set_ylabel("Haemoglobin transcripts\n(HBB+HBA1+HBA2+HBD) / total FPKM")
    ax.set_title("Globin share of the library in each sample (digits above control bars: projected partition)")
    ax.legend(handles=[Patch(color=C_S1, label="TN partition 1"), Patch(color=C_S0, label="TN partition 0"),
                       Patch(color=C_CTRL, label="Control")], frameon=False, ncol=3, loc="upper right")
    panel(ax, "A", x=-0.08)

    ax = fig.add_subplot(gs[1, 0])
    pcs = pd.read_csv(R / "B7_TN_PC_scores.csv", index_col=0)
    proj = pd.read_csv(R / "B5_controls_projected_onto_TN_partition.csv").set_index("sample")
    for st, col in [(0, C_S0), (1, C_S1)]:
        d = pcs[pcs.subtype == st]
        ax.scatter(g.loc[d.index, "globin_fraction"], d.PC1, color=col, s=26, edgecolor="black", lw=0.4, zorder=3)
    ax.scatter(g.loc[proj.index, "globin_fraction"], proj.PC1, facecolor="white", edgecolor=C_CTRL, s=24, lw=0.9, zorder=2)
    rt = stats.spearmanr(g.loc[pcs.index, "globin_fraction"], pcs.PC1)
    rc = stats.spearmanr(g.loc[proj.index, "globin_fraction"], proj.PC1)
    ax.text(0.97, 0.95, f"TN: Spearman ρ = {rt.correlation:.2f} ({fmt_p(rt.pvalue)})\n"
            f"Controls: ρ = {rc.correlation:.2f} ({fmt_p(rc.pvalue)})", transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    ax.set_xlabel("Globin fraction of total FPKM")
    ax.set_ylabel("PC1 score (submitted TN PCA)")
    ax.set_title("PC1 tracks library globin content")
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    v = pd.read_csv(R / "B10_partition_after_globin_adjustment.csv")
    labels = ["Submitted pipeline", "Globin genes removed,\nrenormalised, z-scored", "Globin genes removed,\nlog2, expressed genes",
              "Residualised on\nglobin fraction"]
    vals = [1.0] + v.ARI_vs_submitted.tolist()
    sizes = ["6/4"] + v.sizes.tolist()
    yy = np.arange(4)[::-1]
    ax.barh(yy, vals, color=["black", "#777777", "#777777", "#BBBBBB"], edgecolor="black", lw=0.3)
    for y_, b, s in zip(yy, vals, sizes):
        ax.text(b + 0.02, y_, f"{b:.2f} ({s})", va="center", fontsize=5.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("Agreement with submitted partition (ARI)")
    ax.set_title("Partition after globin adjustment")
    panel(ax, "C", x=-0.62)
    save(fig, "Figure2_globin_composition")


# ---------------------------------------------------------------------------
def figure3():
    fig = plt.figure(figsize=(6.3, 6.4))
    gs = GridSpec(3, 6, figure=fig, wspace=1.0, hspace=0.85)
    sub = pd.read_csv(R / "B4b_submitted_gene_sets_full_pipeline_null.csv")
    rnd = pd.read_csv(R / "B4c_submitted_gene_sets_matched_random_null.csv").set_index("module")
    names = {"peripheral_ecm_schwann": "Submitted 'peripheral ECM/Schwann'", "central_synaptic": "Submitted 'central synaptic'",
             "ntrk_signaling": "Submitted 'NTRK signalling'"}
    for i, (key, lab) in enumerate(names.items()):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        arr = ND[f"fullnull_submitted_exact_{key}"]
        arr = arr[~np.isnan(arr)]
        r = sub[sub.module == f"submitted_exact_{key}"].iloc[0]
        ax.hist(arr, bins=40, color="#BBBBBB", edgecolor="white", lw=0.3)
        ax.axvline(abs(r.submitted_cohens_d), color=C_S1, lw=1.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.set_title(f"{lab}\n({int(r.n_genes_as_submitted)} genes as scored)", fontsize=6.3)
        ax.set_xlabel("|Cohen's d| between k-means clusters")
        p_rand = rnd.loc[f"submitted_exact_{key}", "random_geneset_p"]
        ax.text(0.98, 0.97, f"observed |d| = {abs(r.submitted_cohens_d):.2f}\nfull-pipeline null: {fmt_p(r.full_pipeline_null_p)}\n"
                f"matched random sets: {fmt_p(p_rand)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))
        if i == 0:
            ax.set_ylabel("Null datasets")
            panel(ax, "A", x=-0.35)
    mres = pd.read_csv(R / "B4_module_effects_circularity_nulls.csv").set_index("module")
    for i, (key, lab) in enumerate(names.items()):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        name = f"legacy_{key}"
        arr = ND[f"randset_{name}"]
        r = mres.loc[name]
        ax.hist(arr, bins=40, color="#BBBBBB", edgecolor="white", lw=0.3)
        ax.axvline(abs(r.cohens_d_injury_oriented_log2), color=C_S1, lw=1.3)
        ax.set_title(f"{lab.replace('Submitted ', '')}\n({int(r.n_expressed_genes)} blood-expressed genes)", fontsize=6.3)
        ax.set_xlabel("|Cohen's d|, matched random sets")
        ax.text(0.97, 0.95, f"observed |d| = {abs(r.cohens_d_injury_oriented_log2):.2f}\n{fmt_p(r.random_geneset_p)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.6)
        if i == 0:
            ax.set_ylabel("Random gene sets\n(of 10,000)")
            panel(ax, "B", x=-0.35)
    for i, (f, lab, key) in enumerate([("B8_TN_vs_control_limma_trend_sex_adjusted.csv", "TN vs control (10 vs 10)", "B8_TN_vs_control_sexadj_FDR05"),
                                       ("B8_subtype1_vs_0_limma_trend_sex_adjusted.csv", "Partition 1 vs 0 (4 vs 6; circular)", "B8_subtype1_vs_0_sexadj_FDR05")]):
        ax = fig.add_subplot(gs[2, 3 * i:3 * i + 3])
        d = pd.read_csv(R / f)
        ax.hist(d.p_value, bins=40, color=C_CTRL if i == 0 else C_S1, edgecolor="white", lw=0.3)
        ax.axhline(len(d) / 40, color="black", ls="--", lw=0.7)
        ax.set_xlabel("Moderated-t P value (sex-adjusted)")
        ax.set_title(f"{lab}: {S[key]:,} genes at FDR < 0.05", fontsize=6.5)
        if i == 0:
            ax.set_ylabel("Genes (of 11,219 expressed)")
            panel(ax, "C", x=-0.3)
    save(fig, "Figure3_circularity_nulls")


# ---------------------------------------------------------------------------
def figure4():
    fig = plt.figure(figsize=(6.3, 8.4))
    gs = GridSpec(3, 4, figure=fig, wspace=1.2, hspace=0.75, height_ratios=[1.15, 0.9, 1.1])
    audit = pd.read_csv(R / "A1_ion_cci_sample_label_audit.csv", index_col=0)
    counts = pd.read_csv(ION / "ion_cci_counts.csv", index_col=0)
    lc = np.log2(counts.loc[(counts >= 10).sum(axis=1) >= 3] + 1)
    corr = lc.corr(method="spearman")
    order = corr.index[leaves_list(linkage(corr.values, "average"))]
    corr = corr.loc[order, order]
    ax = fig.add_subplot(gs[0, 0:2])
    im = ax.imshow(corr.values, cmap="viridis", vmin=corr.values.min(), vmax=1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([s.replace("SRR251585", "…") for s in order], rotation=90, fontsize=5.3)
    ax.set_yticks([])
    tc = {"tg": C_TG, "sp5c": C_SP}
    cc = {"injury": "black", "sham": "white"}
    for i, s in enumerate(order):
        a = audit.loc[s]
        for j, (col, key) in enumerate([(tc[a.tissue], "t"), (cc[a.condition], "c"), (tc[a.legacy_tissue], "lt"), (cc[a.legacy_condition], "lc")]):
            ax.add_patch(plt.Rectangle((-5.2 + j * 1.05, i - 0.5), 0.95, 1, facecolor=col, edgecolor="#444444", lw=0.3, clip_on=False))
        if a.tissue_mislabeled or a.condition_mislabeled:
            ax.text(-6.0, i, "×", color="red", fontsize=8, ha="center", va="center", clip_on=False, fontweight="bold")
    for j, lab in enumerate(["SRA tissue", "SRA group", "Legacy tissue", "Legacy group"]):
        ax.text(-5.2 + j * 1.05 + 0.45, -0.9, lab, rotation=90, fontsize=5, ha="center", va="bottom", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Spearman ρ (log2 counts)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5)
    ax.legend(handles=[Patch(facecolor=C_TG, label="TG"), Patch(facecolor=C_SP, label="Sp5C"),
                       Patch(facecolor="black", label="IoN-CCI"), Patch(facecolor="white", edgecolor="#444444", label="Sham")],
              frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    ax.set_title("PRJNA991739: samples cluster by SRA tissue;\n6/12 legacy labels incorrect (×)", fontsize=6.5)
    panel(ax, "A", x=-0.62)

    ax = fig.add_subplot(gs[0, 3])
    vals = [S["ion_cci_tg_DE_up_padj05"], S["ion_cci_tg_DE_down_padj05"], S["ion_cci_sp5c_DE_up_padj05"], S["ion_cci_sp5c_DE_down_padj05"]]
    ax.bar([0, 1, 2.5, 3.5], vals, color=[C_TG, C_TG, C_SP, C_SP], edgecolor="black", lw=0.3, hatch=["", "////", "", "////"])
    for xx, vv in zip([0, 1, 2.5, 3.5], vals):
        ax.text(xx, vv + 40, f"{vv:,}", ha="center", fontsize=5.4)
    ax.set_xticks([0, 1, 2.5, 3.5])
    ax.set_xticklabels(["TG up", "TG down", "Sp5C up", "Sp5C down"], rotation=45, ha="right")
    ax.set_ylabel("Genes, IoN-CCI vs sham\n(PyDESeq2, padj < 0.05; 3 vs 3)")
    ax.set_title("Corrected per-tissue DE", fontsize=6.5)
    panel(ax, "B", x=-0.75)

    leg = pd.read_csv(ION / "de_ion_cci_vs_sham_tg.csv").set_index("gene").dropna(subset=["log2FoldChange"])
    tis = pd.read_csv(R / "A2_ion_cci_deseq2_sham_TG_vs_Sp5C_tissue_identity.csv", index_col=0).dropna(subset=["log2FoldChange"])
    inj = pd.read_csv(R / "A2_ion_cci_deseq2_tg_injury_vs_sham.csv", index_col=0).dropna(subset=["log2FoldChange"])
    for k, (ref, lab, letter) in enumerate([(tis, "True tissue effect: TG vs Sp5C (sham), log2FC", "C"),
                                           (inj, "True injury effect: TG IoN-CCI vs sham, log2FC", None)]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        cm = leg.index.intersection(ref.index)
        x, yv = leg.loc[cm, "log2FoldChange"].clip(-12, 12), ref.loc[cm, "log2FoldChange"].clip(-12, 12)
        ax.hexbin(x, yv, gridsize=45, bins="log", cmap="Greys", mincnt=1, linewidths=0)
        rho = stats.spearmanr(x, yv).correlation
        ax.set_title(f"Legacy 'IoN-CCI' effect vs {'tissue identity' if k == 0 else 'true TG injury effect'}\n"
                     f"Spearman ρ = {rho:.2f} ({len(cm):,} genes)", fontsize=6.3)
        ax.set_xlabel("Legacy 'IoN-CCI vs sham' log2FC (mislabelled samples)")
        ax.set_ylabel(lab.split(": ")[1], fontsize=6)
        ax.axhline(0, color="#cccccc", lw=0.4)
        ax.axvline(0, color="#cccccc", lw=0.4)
        if letter:
            panel(ax, letter, x=-0.22)

    ax = fig.add_subplot(gs[2, 2:4])
    ora = pd.read_csv(R / "A3_ion_cci_reactome_ORA_by_tissue_direction.csv")
    rows = []
    for (t, d) in [("tg", "up"), ("tg", "down"), ("sp5c", "up"), ("sp5c", "down")]:
        sub_ = ora[(ora.tissue == t) & (ora.direction == d)].head(4)
        for _, r in sub_.iterrows():
            rows.append((f"{'TG' if t=='tg' else 'Sp5C'} {d}", r.term.split(" R-HSA")[0], -np.log10(r.p_adj), r.overlap, t))
    rows = rows[::-1]
    for i, (grp, term, q, ov, t) in enumerate(rows):
        ax.scatter(q, i, s=8 + ov * 0.6, color=C_TG if t == "tg" else C_SP, edgecolor="black", lw=0.3,
                   marker="^" if "up" in grp else "v")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{g}: {t[:58]}" for g, t, *_ in rows], fontsize=5.5)
    ax.set_xlabel("−log10 adjusted P (Reactome 2022 over-representation)")
    ax.set_title("Top pathways per tissue and direction (corrected labels)", fontsize=6.5)
    ax.spines["left"].set_visible(False)
    panel(ax, "D", x=-1.35)
    save(fig, "Figure4_ion_cci_corrected")


# ---------------------------------------------------------------------------
def _strip(ax, df, col, title, show_y=False):
    order = ["Control", "Subtype 0", "Subtype 1"]
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = df.loc[df.group == g, col].dropna()
        ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, medianprops=dict(color="black", lw=0.9),
                   boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=12, color=GROUP_COL[g], edgecolor="black", lw=0.3, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Ctrl", "TN-0", "TN-1"], fontsize=6)
    ctrl = df.loc[df.group == "Control", col]
    tn = df.loc[df.group != "Control", col]
    p = stats.mannwhitneyu(ctrl, tn, method="exact").pvalue
    ax.set_title(f"{title}\nTN vs control {fmt_p(p)}", fontsize=5.8)


def figure5():
    fig = plt.figure(figsize=(6.3, 6.3))
    gs = GridSpec(3, 4, figure=fig, wspace=0.65, hspace=0.95, height_ratios=[1, 0.8, 1])
    ms = pd.read_csv(R / "B5_module_scores_all20_log2z.csv", index_col=0)
    mods = [("revised_peripheral_ecm_schwann_signature", "TG injury-up signature\n(51 blood-expressed)"),
            ("revised_central_synaptic_pathway", "Sp5C injury-down synaptic\n(15 blood-expressed)"),
            ("revised_ntrk_signaling_pathway", "Sp5C injury-down NTRK\n(15 blood-expressed)"),
            ("revised_ntrk_signaling_TGdown_pathway", "TG injury-down NTRK\n(40 blood-expressed)")]
    for i, (col, lab) in enumerate(mods):
        ax = fig.add_subplot(gs[0, i])
        _strip(ax, ms, col, lab)
        if i == 0:
            ax.set_ylabel("Injury-oriented module score\n(mean z, log2 FPKM+1)")
            panel(ax, "A", x=-0.75, y=1.2)
    ax = fig.add_subplot(gs[1, :])
    cov = pd.read_csv(R / "B4_module_ortholog_blood_expression.csv")
    keep = ["legacy_peripheral_ecm_schwann", "legacy_central_synaptic", "legacy_ntrk_signaling",
            "revised_peripheral_ecm_schwann_pathway", "revised_central_synaptic_pathway", "revised_ntrk_signaling_pathway",
            "revised_central_synaptic_TGdown_pathway", "revised_ntrk_signaling_TGdown_pathway"]
    lab = {"legacy_peripheral_ecm_schwann": "Submitted peripheral ECM/Schwann", "legacy_central_synaptic": "Submitted central synaptic",
           "legacy_ntrk_signaling": "Submitted NTRK", "revised_peripheral_ecm_schwann_pathway": "Revised TG-up ECM/Schwann terms",
           "revised_central_synaptic_pathway": "Revised Sp5C-down synaptic terms", "revised_ntrk_signaling_pathway": "Revised Sp5C-down NTRK terms",
           "revised_central_synaptic_TGdown_pathway": "Revised TG-down synaptic terms", "revised_ntrk_signaling_TGdown_pathway": "Revised TG-down NTRK terms"}
    cov = cov.set_index("module").loc[keep]
    y = np.arange(len(cov))[::-1]
    ax.barh(y, cov.n_human_orthologs, color="#E5E5E5", edgecolor="black", lw=0.3, label="Human orthologs")
    ax.barh(y, cov.n_in_blood_matrix, color="#AAAAAA", edgecolor="black", lw=0.3, label="Present in blood matrix")
    ax.barh(y, cov.n_mean_FPKM_ge_1, color="black", edgecolor="black", lw=0.3, label="Mean FPKM ≥ 1 (scored)")
    for yy, (_, r) in zip(y, cov.iterrows()):
        ax.text(r.n_human_orthologs + 1.5, yy, f"{int(r.n_mean_FPKM_ge_1)}/{int(r.n_human_orthologs)}", va="center", fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels([lab[k] for k in cov.index], fontsize=5.8)
    ax.set_xlabel("Genes")
    ax.set_xlim(0, 150)
    ax.set_title("Module orthologs expressed in whole blood", fontsize=6.5)
    ax.legend(frameon=False, loc="upper right", fontsize=5.5)
    panel(ax, "B", x=-0.3)
    mk = pd.read_csv(R / "B6_marker_scores_all20.csv", index_col=0)
    tests = pd.read_csv(R / "B6_cell_composition_tests.csv").set_index("feature")
    cells = ["Erythroid", "Neutrophil", "Monocyte", "T_cell"]
    for i, c in enumerate(cells):
        ax = fig.add_subplot(gs[2, i])
        _strip(ax, mk, c, f"{c.replace('_', ' ')} markers\nTN-1 vs TN-0 q = {tests.loc[c, 'subtype1_vs_0_MWU_exact_q']:.3f}")
        if i == 0:
            ax.set_ylabel("Marker score\n(mean log2 FPKM+1)")
            panel(ax, "C", x=-0.75, y=1.2)
    save(fig, "Figure5_blood_modules_composition")


# ---------------------------------------------------------------------------
def supp_figures():
    # S1 k-selection metrics with null
    k = pd.read_csv(R / "B2_k_selection_metrics_with_null.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1))
    ci = json.loads((R / "REVISION_SUMMARY.json").read_text())["B2_obs_2means_cluster_index"]
    tss = float(k.loc[k.k == 2, "within_SS"].iloc[0]) / ci
    axs[0].plot([1] + k.k.tolist(), [tss / 1e5] + (k.within_SS / 1e5).tolist(), "o-", color="black", ms=3)
    axs[0].set_title("Within-cluster SS (×10⁵)")
    axs[1].plot(k.k, k.silhouette, "o-", color=C_S1, ms=3, label="Observed")
    axs[1].plot(k.k, k.null_silhouette_mean, "s--", color=C_CTRL, ms=3, label="Null mean")
    axs[1].fill_between(k.k, 0, k.null_silhouette_95pct, color="#DDDDDD", label="Null 95th pct")
    axs[1].set_title("Average silhouette")
    axs[1].legend(frameon=False)
    axs[2].plot(k.k, k.calinski_harabasz, "o-", color=C_S1, ms=3)
    axs[2].plot(k.k, k.null_CH_mean, "s--", color=C_CTRL, ms=3)
    axs[2].set_title("Calinski-Harabasz")
    for i, a in enumerate(axs):
        a.set_xticks([1, 2, 3, 4] if i == 0 else [2, 3, 4])
        a.set_xlabel("k")
    fig.tight_layout()
    save(fig, "FigureS1_k_selection_null")
    # S2 sensitivity
    sv = pd.read_csv(R / "B3_sensitivity_variants.csv")
    loo = pd.read_csv(R / "B3_leave_one_out.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    yy = np.arange(len(sv))[::-1]
    axs[0].barh(yy, sv.ARI_vs_original, color=["black" if a > 0.999 else C_S1 for a in sv.ARI_vs_original])
    axs[0].set_yticks(yy)
    axs[0].set_yticklabels(sv.variant, fontsize=5.2)
    axs[0].set_xlabel("ARI with submitted partition")
    axs[0].set_title("Analytical sensitivity")
    axs[1].bar(range(10), loo.ARI_vs_original, color="black")
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels(loo.left_out, rotation=90, fontsize=5.5)
    axs[1].set_ylabel("ARI (remaining 9)")
    axs[1].set_title("Leave-one-out")
    fig.tight_layout()
    save(fig, "FigureS2_stability_sensitivity")
    # S3 TN vs control volcano + GSEA
    d = pd.read_csv(R / "B8_TN_vs_control_limma_trend_sex_adjusted.csv")
    gsea = pd.read_csv(R / "B9_prerank_GSEA_TN_vs_control_moderated_t.csv")
    gsea["fdr"] = pd.to_numeric(gsea["FDR q-val"], errors="coerce")
    top = gsea.sort_values("fdr").head(12)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw=dict(width_ratios=[1, 1.5]))
    axs[0].scatter(d.log2FC, -np.log10(d.p_value), s=2, color=C_CTRL, alpha=0.6, lw=0)
    axs[0].set_xlabel("log2 fold change (TN vs control)")
    axs[0].set_ylabel("−log10 P (moderated t)")
    axs[0].set_title("TN vs control: 0 genes at FDR < 0.05")
    axs[1].barh(range(len(top)), top.NES, color=[C_S1 if n > 0 else C_S0 for n in top.NES])
    axs[1].set_yticks(range(len(top)))
    axs[1].set_yticklabels([f"{t.split(' R-HSA')[0].split(' (GO')[0][:48]} (q={q:.3f})" for t, q in zip(top.Term, top.fdr)], fontsize=5.2)
    axs[1].set_xlabel("Normalised enrichment score")
    axs[1].set_title("Preranked GSEA (moderated t)")
    fig.tight_layout()
    save(fig, "FigureS3_TN_vs_control")
    # S4 legacy DE artefacts
    leg = pd.read_csv(ROOT / "results" / "subtype_characterization" / "de_subtype_1.csv")
    mod = pd.read_csv(R / "B8_subtype1_vs_0_limma_trend_sex_adjusted.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
    sig = leg[(leg.p_adj < 0.05) & (leg.abs_log2fc > 1)]
    mean_expr = np.log10((sig.mean_subtype + sig.mean_other) / 2 + 1e-3)
    axs[0].scatter(mean_expr, sig.log2fc, s=5, color=C_S1)
    axs[0].axhline(1, ls="--", color="#999999", lw=0.5)
    axs[0].axhline(-1, ls="--", color="#999999", lw=0.5)
    axs[0].set_xlabel("log10 mean FPKM")
    axs[0].set_ylabel("Submitted log2FC (ratio of FPKM means)")
    axs[0].set_title(f"Submitted 247 genes: {int(((sig.mean_subtype<1)&(sig.mean_other<1)).sum())} with mean FPKM < 1")
    axs[1].scatter(mod.AveExpr, mod.log2FC, s=1.5, color=C_CTRL, alpha=0.5, lw=0)
    s2 = mod[mod.p_adj < 0.05]
    axs[1].scatter(s2.AveExpr, s2.log2FC, s=1.5, color=C_S1, alpha=0.5, lw=0)
    axs[1].set_xlabel("Mean log2(FPKM+1)")
    axs[1].set_ylabel("log2FC, partition 1 vs 0")
    axs[1].set_title(f"Moderated t: {len(s2):,} genes FDR < 0.05, mostly |log2FC| < 1")
    fig.tight_layout()
    save(fig, "FigureS4_subtype_DE_artefacts")
    # S5 ABIS
    ab = pd.read_csv(R / "B6_ABIS_NNLS_deconvolution_all20.csv", index_col=0)
    cells = [c for c in ab.columns if c not in ("group", "fit_pearson_r", "fit_nRMSE") and ab[c].mean() > 0.005]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    xpos = np.arange(len(cells))
    for j, g in enumerate(["Control", "Subtype 0", "Subtype 1"]):
        m = ab[ab.group == g][cells]
        axs[0].bar(xpos + (j - 1) * 0.27, m.mean(), 0.27, yerr=m.std(), color=GROUP_COL[g], edgecolor="black", lw=0.3,
                   error_kw=dict(lw=0.5, capsize=1.2), label=g.replace("Subtype", "TN partition"))
    axs[0].set_xticks(xpos)
    axs[0].set_xticklabels(cells, rotation=45, ha="right", fontsize=5.6)
    axs[0].set_ylabel("ABIS NNLS fraction (mean ± SD)")
    axs[0].legend(frameon=False)
    axs[0].set_title("Reference-based deconvolution (ABIS RNA-seq signature)")
    axs[1].scatter(range(20), ab.fit_pearson_r, c=[GROUP_COL[g] for g in ab.group], s=10)
    axs[1].set_ylim(0.7, 0.9)
    axs[1].set_xticks([])
    axs[1].set_ylabel("Fit Pearson r (signature genes)")
    axs[1].set_title("Goodness of fit")
    fig.tight_layout()
    save(fig, "FigureS5_ABIS_deconvolution")
    # S6 sex
    sx = pd.read_csv(R / "B7_sex_check.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    offsets = {"Control": -0.2, "Subtype 0": 0.0, "Subtype 1": 0.2}
    jit = np.random.default_rng(1)
    for g in ["Control", "Subtype 0", "Subtype 1"]:
        d = sx[sx.group == g]
        ax.scatter(np.where(d.sex_GEO == "M", 1, 0) + offsets[g] + jit.uniform(-0.05, 0.05, len(d)), d.Ychr_mean,
                   color=GROUP_COL[g], s=14, edgecolor="black", lw=0.3, label=g.replace("Subtype", "TN partition"))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GEO: female", "GEO: male"])
    ax.set_ylabel("Mean log2(FPKM+1), Y-linked genes")
    ax.set_title(f"Expression-based sex concordant in {S['B7_sex_concordant']}")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    save(fig, "FigureS6_sex_check")
    figure_s7()


def figure_s7():
    C = R / "C_salmon"
    if not (C / "C1_salmon_qc.csv").exists():
        print("skip FigureS7 (salmon results missing)")
        return
    qc = pd.read_csv(C / "C1_salmon_qc.csv", index_col=0)
    col = {"Control": C_CTRL, "Partition 0": C_S0, "Partition 1": C_S1}
    lab = {"Control": "Control", "Partition 0": "TN partition 0", "Partition 1": "TN partition 1"}
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for g in ["Control", "Partition 0", "Partition 1"]:
        d = qc[qc.group == g]
        axs[0].scatter(d.percent_mapped, d.spearman_log_TPM_vs_deposited_FPKM, color=col[g], s=14, edgecolor="black",
                       lw=0.3, label=lab[g])
        axs[1].scatter(d.globin_fraction_deposited_FPKM, d.globin_fraction_salmon_TPM, color=col[g], s=14,
                       edgecolor="black", lw=0.3)
    axs[0].set_xlabel("salmon mapping rate (%)")
    axs[0].set_ylabel("Spearman ρ, log TPM vs deposited log FPKM")
    axs[0].legend(frameon=False, fontsize=5.5)
    axs[0].set_title("Agreement with deposited values")
    lim = [0, max(qc.globin_fraction_deposited_FPKM.max(), qc.globin_fraction_salmon_TPM.max()) * 1.05]
    axs[1].plot(lim, lim, ls="--", lw=0.5, color="grey")
    rho = stats.spearmanr(qc.globin_fraction_deposited_FPKM, qc.globin_fraction_salmon_TPM).correlation
    axs[1].set_xlabel("Globin fraction, deposited FPKM")
    axs[1].set_ylabel("Globin fraction, salmon TPM")
    axs[1].set_title(f"Globin share (Spearman ρ = {rho:.2f})")
    de = pd.read_csv(C / "C2_TN_vs_control_DESeq2_sex_adjusted.csv", index_col=0).dropna(subset=["pvalue"])
    sig = de.padj < 0.05
    axs[2].scatter(de.log2FoldChange[~sig], -np.log10(de.pvalue[~sig]), s=1.5, color="#9E9E9E", rasterized=True)
    axs[2].scatter(de.log2FoldChange[sig], -np.log10(de.pvalue[sig]), s=3, color=C_S1, rasterized=True)
    axs[2].set_xlabel("log2 fold change, TN vs control")
    axs[2].set_ylabel("−log10 P")
    axs[2].set_title(f"TN vs control, PyDESeq2 (~ sex + condition)\n{int(sig.sum())} genes at adjusted P < 0.05")
    axs[0].set_title("Agreement with\ndeposited values")
    axs[1].set_title(f"Globin share\n(Spearman ρ = {rho:.2f})")
    for a, l, x in zip(axs, "ABC", [-0.45, -0.3, -0.25]):
        panel(a, l, x=x, y=1.12)
    fig.tight_layout(w_pad=2.0)
    save(fig, "FigureS7_salmon_requant")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "S"]
    fns = {"1": figure1, "2": figure2, "3": figure3, "4": figure4, "5": figure5, "S": supp_figures, "S7": figure_s7}
    for w in which:
        fns[w]()
