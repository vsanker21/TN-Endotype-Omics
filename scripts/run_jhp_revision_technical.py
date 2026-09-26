"""
JHP revision D: technical covariates available only from the raw reads of GSE186505.

D1  sequencing batch (instrument:run:flowcell:lane from the first read header of each R1 FASTQ, streamed from ENA;
    scripts/wsl/fastq_headers.sh), read depth and salmon mapping rate per sample
D2  association of these covariates with the TN partition, PC1, globin fraction and diagnosis
D3  TN vs control with sequencing batch as an additional covariate (limma-trend on deposited FPKM; PyDESeq2 on
    salmon counts)
Outputs: results/jhp_revision/D_technical/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jhp_revision_analyses import EXPR_FILE, GEO_SEX, OUT, SUBTYPES_FILE, limma_trend  # noqa: E402
from run_jhp_revision_salmon_requant import C_OUT, GLOBIN4, WSL, deseq  # noqa: E402

D_OUT = OUT / "D_technical"
D_OUT.mkdir(parents=True, exist_ok=True)
HEADERS = WSL / "tn_gse186505" / "fastq_headers.tsv"


def mwu(a, b) -> float:
    return float(stats.mannwhitneyu(a, b, method="exact").pvalue)


def main() -> None:
    qc = pd.read_csv(C_OUT / "C1_salmon_qc.csv", index_col=0)
    hd = pd.read_csv(HEADERS, sep="\t").set_index("run_accession")
    flow = hd["headers_first4"].str.strip().str.split("(").str[0]
    qc["sequencing_run"] = flow.reindex(qc["run"]).values
    qc["batch"] = qc["sequencing_run"].map({r: f"B{i + 1}" for i, r in
                                            enumerate(qc["sequencing_run"].value_counts().index)})
    pcs = pd.read_csv(OUT / "B7_TN_PC_scores.csv", index_col=0)
    qc["PC1_TN"] = pcs["PC1"].reindex(qc.index)
    qc.to_csv(D_OUT / "D1_technical_covariates_per_sample.csv")

    tn = qc[qc.group != "Control"]
    p1, p0, ctl = qc[qc.group == "Partition 1"], qc[qc.group == "Partition 0"], qc[qc.group == "Control"]
    S: dict = {"batches": qc.groupby("batch")["sequencing_run"].first().to_dict(),
               "batch_counts": qc.groupby(["batch", "group"]).size().unstack(fill_value=0).to_dict(orient="index")}
    ct = pd.crosstab(tn.group, tn.batch)
    S["batch_vs_partition_fisher_p"] = float(stats.fisher_exact(ct.values)[1]) if ct.shape == (2, 2) else 1.0
    ct2 = pd.crosstab(qc.group == "Control", qc.batch)
    S["batch_vs_diagnosis_fisher_p"] = float(stats.fisher_exact(ct2.values)[1])

    rows = []
    for v, lab in [("num_processed", "Read pairs processed"), ("percent_mapped", "salmon mapping rate (%)")]:
        rows.append({
            "covariate": lab,
            "partition0_median": p0[v].median(), "partition1_median": p1[v].median(), "control_median": ctl[v].median(),
            "partition1_vs_0_exact_MWU_p": mwu(p1[v], p0[v]),
            "TN_vs_control_MWU_p": mwu(tn[v], ctl[v]),
            "spearman_vs_PC1_TN": stats.spearmanr(tn[v], tn.PC1_TN).correlation,
            "spearman_vs_PC1_TN_p": stats.spearmanr(tn[v], tn.PC1_TN).pvalue,
            "spearman_vs_globin_all20": stats.spearmanr(qc[v], qc.globin_fraction_salmon_counts).correlation,
            "spearman_vs_globin_all20_p": stats.spearmanr(qc[v], qc.globin_fraction_salmon_counts).pvalue,
        })
    b2 = qc.batch == "B2"
    rows.append({"covariate": "Minority sequencing run (B2), n", "partition0_median": int(b2[p0.index].sum()),
                 "partition1_median": int(b2[p1.index].sum()), "control_median": int(b2[ctl.index].sum()),
                 "partition1_vs_0_exact_MWU_p": S["batch_vs_partition_fisher_p"],
                 "TN_vs_control_MWU_p": S["batch_vs_diagnosis_fisher_p"]})
    d2 = pd.DataFrame(rows)
    d2.to_csv(D_OUT / "D2_technical_covariates_tests.csv", index=False)
    S["tests"] = d2.round(4).to_dict(orient="records")

    # D3 TN vs control with batch
    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]
    L = np.log2(fpkm + 1)
    expressed = fpkm.index[(fpkm >= 1).sum(axis=1) >= 5]
    cols = L.columns
    cond = np.array([1.0 if s.startswith("TN") else 0.0 for s in cols])
    sex = np.array([1.0 if GEO_SEX[s] == "M" else 0.0 for s in cols])
    bat = np.array([1.0 if qc.loc[s, "batch"] == "B2" else 0.0 for s in cols])
    gf = (fpkm.reindex(GLOBIN4).sum() / fpkm.sum()).values
    for name, X in [("sex_batch", np.column_stack([np.ones(len(cols)), sex, bat, cond])),
                    ("sex_batch_globin", np.column_stack([np.ones(len(cols)), sex, bat, (gf - gf.mean()) / gf.std(), cond]))]:
        lt = limma_trend(L.loc[expressed], X, coef=X.shape[1] - 1)
        lt.to_csv(D_OUT / f"D3_TN_vs_control_limma_trend_{name}.csv", index=False)
        S[f"D3_limma_{name}_FDR05"] = int((lt.p_adj < 0.05).sum())
        S[f"D3_limma_{name}_FDR10"] = int((lt.p_adj < 0.10).sum())

    counts = pd.read_csv(C_OUT / "C0_salmon_gene_counts.csv", index_col=0)
    meta = pd.DataFrame({"condition": ["TN" if s.startswith("TN") else "Control" for s in counts.columns],
                         "sex": [GEO_SEX[s] for s in counts.columns],
                         "batch": [qc.loc[s, "batch"] for s in counts.columns]}, index=counts.columns)
    r = deseq(counts, meta, "~ sex + batch + condition")
    r.index.name = "gene"
    r.to_csv(D_OUT / "D3_TN_vs_control_DESeq2_sex_batch.csv")
    S["D3_deseq2_sex_batch_n_tested"] = int(r.padj.notna().sum())
    S["D3_deseq2_sex_batch_padj05"] = int((r.padj < 0.05).sum())
    S["D3_deseq2_sex_batch_padj10"] = int((r.padj < 0.10).sum())

    # D4 is TN more dispersed than controls along the partition axis? (non-circular: uses diagnosis labels only)
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(11)
    ms = pd.read_csv(OUT / "B5_module_scores_all20_log2z.csv", index_col=0)
    mod_cols = [c for c in ms.columns if c not in ("group", "sex_GEO")]
    distinct = ms[mod_cols].T.drop_duplicates().index.tolist()
    Lz = L.loc[expressed]
    Lz = Lz.sub(Lz.mean(axis=1), axis=0).div(Lz.std(axis=1).replace(0, np.nan), axis=0).dropna()
    jp = PCA(2, random_state=0).fit(Lz.T.values)
    jpc = pd.DataFrame(jp.transform(Lz.T.values), index=Lz.columns, columns=["joint_PC1", "joint_PC2"])
    feats = pd.DataFrame({"globin_fraction": pd.Series(gf, index=cols)}).join(jpc).join(ms[distinct])
    is_tn = np.array([s.startswith("TN") for s in feats.index])
    drows = []
    for c in feats.columns:
        v = feats[c].values
        bf = stats.levene(v[is_tn], v[~is_tn], center="median").pvalue
        obs = np.var(v[is_tn], ddof=1) / np.var(v[~is_tn], ddof=1)
        perm = np.empty(20000)
        for b in range(perm.size):
            p = rng.permutation(is_tn)
            perm[b] = np.var(v[p], ddof=1) / np.var(v[~p], ddof=1)
        drows.append({"feature": c, "SD_TN": np.std(v[is_tn], ddof=1), "SD_control": np.std(v[~is_tn], ddof=1),
                      "variance_ratio_TN_over_control": obs, "brown_forsythe_p": bf,
                      "permutation_p_variance_ratio_ge_observed": (1 + np.sum(perm >= obs)) / (perm.size + 1),
                      "TN_vs_control_MWU_p": mwu(v[is_tn], v[~is_tn])})
    d4 = pd.DataFrame(drows)
    d4["brown_forsythe_q"] = stats.false_discovery_control(d4.brown_forsythe_p.values)
    d4["permutation_q"] = stats.false_discovery_control(d4.permutation_p_variance_ratio_ge_observed.values)
    d4.to_csv(D_OUT / "D4_dispersion_TN_vs_control.csv", index=False)
    S["D4_joint_PC_variance_explained"] = [float(x) for x in jp.explained_variance_ratio_]
    S["D4_n_distinct_module_sets"] = len(distinct)
    S["D4"] = d4.round(4).to_dict(orient="records")

    # D5 module-score dispersion after regressing out globin fraction (ordinary least squares on all 20 samples)
    g = feats["globin_fraction"].values
    X = np.column_stack([np.ones_like(g), g])
    rrows = []
    for c in distinct:
        v = feats[c].values
        rho_all = stats.spearmanr(v, g)[0]
        rho_ctrl = stats.spearmanr(v[~is_tn], g[~is_tn])[0]
        res = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        bf = stats.levene(res[is_tn], res[~is_tn], center="median").pvalue
        obs = np.var(res[is_tn], ddof=1) / np.var(res[~is_tn], ddof=1)
        perm = np.empty(20000)
        for b in range(perm.size):
            p = rng.permutation(is_tn)
            perm[b] = np.var(res[p], ddof=1) / np.var(res[~p], ddof=1)
        rrows.append({"feature": c, "spearman_rho_globin_all20": rho_all, "spearman_rho_globin_controls": rho_ctrl,
                      "R2_globin_all20": 1 - np.var(res) / np.var(v),
                      "variance_ratio_TN_over_control_globin_adjusted": obs, "brown_forsythe_p_globin_adjusted": bf,
                      "permutation_p_globin_adjusted": (1 + np.sum(perm >= obs)) / (perm.size + 1)})
    d5 = pd.DataFrame(rrows)
    d5["brown_forsythe_q_globin_adjusted"] = stats.false_discovery_control(d5.brown_forsythe_p_globin_adjusted.values)
    d5["permutation_q_globin_adjusted"] = stats.false_discovery_control(d5.permutation_p_globin_adjusted.values)
    d5.to_csv(D_OUT / "D5_module_dispersion_globin_adjusted.csv", index=False)
    S["D5"] = d5.round(4).to_dict(orient="records")

    (D_OUT / "D_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
