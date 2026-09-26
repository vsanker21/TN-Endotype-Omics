"""
JHP revision C: analyses on GSE186505 re-quantified from raw reads (salmon 1.10.3, GENCODE v26 transcriptome).

C1  QC: mapping rate, reads, agreement between salmon TPM and the deposited RSEM FPKM
C2  TN vs control, PyDESeq2 on gene counts: ~ sex + condition, and ~ sex + globin + condition
C3  the submitted clustering pipeline and clusterability tests on salmon-derived expression; globin checks
C4  TN vs control, limma-trend on deposited log2(FPKM+1) with sex + globin fraction (sensitivity)
Outputs: results/jhp_revision/C_salmon/
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jhp_revision_analyses import (  # noqa: E402
    EXPR_FILE, GEO_SEX, OUT, SUBTYPES_FILE, gap_statistic, limma_trend, pipeline_labels,
)

# salmon output of scripts/wsl/gse186505_salmon.sh and the GENCODE v26 transcript-to-gene map
WSL = Path(os.environ.get("TN_SALMON_HOME", r"\\wsl.localhost\Ubuntu-24.04\home\vsanker"))
QUANT = WSL / "tn_gse186505" / "quant"
ENA = WSL / "tn_gse186505" / "ena.tsv"
TGMAP = Path(os.environ.get("TN_TGMAP", str(WSL / "meningioma_gtex_v8_requant" / "ref" / "gencode.v26.tgmap.tsv")))
C_OUT = OUT / "C_salmon"
C_OUT.mkdir(parents=True, exist_ok=True)
GLOBIN4 = ["HBB", "HBA1", "HBA2", "HBD"]
GLOBIN14 = ["HBB", "HBA1", "HBA2", "HBD", "HBM", "HBQ1", "HBG1", "HBG2", "HBZ", "HBE1", "ALAS2", "SLC4A1", "CA1", "AHSP"]
SUMMARY: dict = {}


def sample_titles() -> dict[str, str]:
    ena = pd.read_csv(ENA, sep="\t")
    txt = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186505&targ=gsm&form=text&view=brief",
                       timeout=120).text
    gsm2title, cur = {}, None
    for line in txt.splitlines():
        if line.startswith("^SAMPLE"):
            cur = line.split("=")[1].strip()
        elif line.startswith("!Sample_title") and cur:
            gsm2title[cur] = line.split("=", 1)[1].strip()
    return {r.run_accession: gsm2title[r.sample_alias] for r in ena.itertuples()}


def gene_symbols(ensg: list[str]) -> dict[str, str]:
    cache = C_OUT / "ensg_to_symbol.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)["symbol"].to_dict()
    import mygene
    mg = mygene.MyGeneInfo()
    res = mg.querymany(ensg, scopes="ensembl.gene", fields="symbol", species="human", as_dataframe=True,
                       returnall=False, verbose=False)
    res = res[~res.index.duplicated()]
    m = res["symbol"].dropna().to_dict()
    pd.Series(m, name="symbol").to_csv(cache)
    return m


def load_salmon() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run2title = sample_titles()
    tg = pd.read_csv(TGMAP, sep="\t", header=None, names=["tx", "gene"]).set_index("tx")["gene"]
    counts, tpm, qc = {}, {}, []
    for run, title in run2title.items():
        q = pd.read_csv(QUANT / run / "quant.sf", sep="\t", index_col=0)
        g = tg.reindex(q.index)
        counts[title] = q["NumReads"].groupby(g.values).sum()
        tpm[title] = q["TPM"].groupby(g.values).sum()
        meta = json.loads((QUANT / run / "aux_info" / "meta_info.json").read_text())
        lib = json.loads((QUANT / run / "lib_format_counts.json").read_text())
        qc.append({"sample": title, "run": run, "num_processed": meta["num_processed"],
                   "num_mapped": meta["num_mapped"], "percent_mapped": meta["percent_mapped"],
                   "library_type": lib.get("expected_format", "")})
    counts, tpm = pd.DataFrame(counts), pd.DataFrame(tpm)
    ensg = [i.split(".")[0] for i in counts.index]
    sym = gene_symbols(sorted(set(ensg)))
    symbols = pd.Series([sym.get(e, e) for e in ensg], index=counts.index)
    counts = counts.groupby(symbols.values).sum()
    tpm = tpm.groupby(symbols.values).sum()
    counts.index = counts.index.astype(str).str.upper()
    tpm.index = tpm.index.astype(str).str.upper()
    return counts.round().astype(int), tpm, pd.DataFrame(qc).set_index("sample")


def deseq(counts: pd.DataFrame, meta: pd.DataFrame, design: str) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats
    keep = (counts >= 10).sum(axis=1) >= 5
    inf = DefaultInference(n_cpus=4)
    dds = DeseqDataSet(counts=counts.loc[keep, meta.index].T, metadata=meta, design=design, refit_cooks=True,
                       inference=inf, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "TN", "Control"], inference=inf, quiet=True)
    ds.summary()
    return ds.results_df.sort_values("pvalue")


def clusterability(X: np.ndarray, y: np.ndarray, rng, B: int = 1000) -> dict:
    lab, Z = pipeline_labels(X)
    sil = silhouette_score(Z, lab)
    lamb = Z.var(axis=0, ddof=1)
    null = [silhouette_score(Zn, KMeans(2, random_state=42, n_init=20).fit_predict(Zn))
            for Zn in (rng.normal(size=Z.shape) * np.sqrt(lamb) for _ in range(B))]
    gap = gap_statistic(Z, 4, 300, seed=3)
    return {"ARI_vs_submitted": adjusted_rand_score(y, lab),
            "sizes": "/".join(map(str, sorted(np.bincount(lab), reverse=True))),
            "silhouette_k2": sil, "null_silhouette_mean": float(np.mean(null)),
            "null_silhouette_95pct": float(np.percentile(null, 95)),
            "silhouette_null_p": (1 + np.sum(np.array(null) >= sil)) / (B + 1),
            "gap_selected_k": int(gap.loc[gap.tibshirani_1SE_selected, "k"].iloc[0]),
            "PC1_scores": Z[:, 0]}


def main() -> None:
    rng = np.random.default_rng(20260928)
    counts, tpm, qc = load_salmon()
    counts.to_csv(C_OUT / "C0_salmon_gene_counts.csv")
    tpm.to_csv(C_OUT / "C0_salmon_gene_tpm.csv")
    samples = counts.columns.tolist()
    st = pd.read_csv(SUBTYPES_FILE).set_index("sample_id")
    tn = st.index.tolist()
    y = st.loc[tn, "subtype"].values
    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]

    # --- C1 QC ---------------------------------------------------------------
    common = tpm.index.intersection(fpkm.index)
    rho = {s: stats.spearmanr(np.log2(tpm.loc[common, s] + 1), np.log2(fpkm.loc[common, s] + 1)).correlation for s in samples}
    qc["spearman_log_TPM_vs_deposited_FPKM"] = pd.Series(rho)
    qc["globin_fraction_salmon_TPM"] = tpm.reindex(GLOBIN4).sum() / tpm.sum()
    qc["globin_fraction_salmon_counts"] = counts.reindex(GLOBIN4).sum() / counts.sum()
    qc["globin_fraction_deposited_FPKM"] = fpkm.reindex(GLOBIN4).sum() / fpkm.sum()
    qc["group"] = ["Control" if s.startswith("Control") else f"Partition {st.loc[s, 'subtype']}" for s in qc.index]
    qc["sex_GEO"] = [GEO_SEX[s] for s in qc.index]
    qc.to_csv(C_OUT / "C1_salmon_qc.csv")
    SUMMARY.update({
        "n_samples": len(samples), "median_reads_processed": float(qc.num_processed.median()),
        "median_percent_mapped": float(qc.percent_mapped.median()),
        "min_percent_mapped": float(qc.percent_mapped.min()),
        "n_genes_common_with_deposited": int(len(common)),
        "median_spearman_TPM_vs_FPKM": float(qc.spearman_log_TPM_vs_deposited_FPKM.median()),
        "globin_fraction_salmon_vs_FPKM_spearman": float(stats.spearmanr(qc.globin_fraction_salmon_TPM,
                                                                          qc.globin_fraction_deposited_FPKM).correlation),
        "globin_fraction_counts_median_partition1": float(qc.loc[tn][y == 1].globin_fraction_salmon_counts.median()),
        "globin_fraction_counts_median_partition0": float(qc.loc[tn][y == 0].globin_fraction_salmon_counts.median()),
        "globin_fraction_counts_median_controls": float(qc[qc.group == "Control"].globin_fraction_salmon_counts.median()),
        "globin_fraction_counts_partition_MWU_p": float(stats.mannwhitneyu(
            qc.loc[tn][y == 1].globin_fraction_salmon_counts, qc.loc[tn][y == 0].globin_fraction_salmon_counts,
            method="exact").pvalue),
    })

    # --- C2 DESeq2 TN vs control -------------------------------------------------
    meta = pd.DataFrame({"condition": ["TN" if s.startswith("TN") else "Control" for s in samples],
                         "sex": [GEO_SEX[s] for s in samples]}, index=samples)
    g = qc.loc[samples, "globin_fraction_salmon_counts"]
    meta["globin"] = (g - g.mean()) / g.std()
    counts_ng = counts.drop(index=[x for x in GLOBIN14 if x in counts.index])
    res = {}
    for name, design, cmat in [("sex_adjusted", "~ sex + condition", counts),
                               ("sex_globin_adjusted", "~ sex + globin + condition", counts),
                               ("sex_adjusted_globin_genes_removed", "~ sex + condition", counts_ng)]:
        r = deseq(cmat, meta, design)
        r.index.name = "gene"
        r.to_csv(C_OUT / f"C2_TN_vs_control_DESeq2_{name}.csv")
        res[name] = r
        SUMMARY[f"C2_{name}_n_tested"] = int(r.padj.notna().sum())
        SUMMARY[f"C2_{name}_padj05"] = int((r.padj < 0.05).sum())
        SUMMARY[f"C2_{name}_padj10"] = int((r.padj < 0.10).sum())
        SUMMARY[f"C2_{name}_padj05_absLFC1"] = int(((r.padj < 0.05) & (r.log2FoldChange.abs() > 1)).sum())
        SUMMARY[f"C2_{name}_top"] = r.head(10)[["log2FoldChange", "pvalue", "padj"]].round(4).reset_index().to_dict(orient="records")
        if "CSF2" in r.index:
            SUMMARY[f"C2_{name}_CSF2"] = r.loc["CSF2", ["baseMean", "log2FoldChange", "pvalue", "padj"]].round(4).to_dict()

    # --- C3 clustering on salmon-derived expression ---------------------------------
    rows = []
    t_tn = tpm[tn]
    nzv = t_tn.index[t_tn.std(axis=1) > 0]
    cpm = counts / counts.sum() * 1e6
    expr = cpm.index[(cpm >= 1).sum(axis=1) >= 5]
    tpm_ng = tpm.drop(index=[x for x in GLOBIN14 if x in tpm.index])
    tpm_ng = tpm_ng / tpm_ng.sum() * 1e6
    variants = {
        "salmon TPM, z-scored, all non-zero-variance genes (submitted pipeline)": t_tn.loc[nzv].T.values,
        "salmon log2(CPM+1), z-scored, expressed genes": np.log2(cpm.loc[expr, tn] + 1).T.values,
        "salmon TPM, globin/erythroid genes removed and renormalised, z-scored": tpm_ng.loc[tpm_ng[tn].std(axis=1) > 0, tn].T.values,
    }
    gfr = qc.loc[tn, "globin_fraction_salmon_counts"].values
    for name, X in variants.items():
        r = clusterability(X, y, rng)
        pc1 = r.pop("PC1_scores")
        r["PC1_vs_globin_spearman"] = float(stats.spearmanr(pc1, gfr).correlation)
        rows.append({"variant": name, **r})
    c3 = pd.DataFrame(rows)
    c3.to_csv(C_OUT / "C3_partition_on_salmon_counts.csv", index=False)
    SUMMARY["C3"] = c3.round(4).to_dict(orient="records")

    # --- C4 limma-trend on deposited FPKM with globin covariate ----------------------------
    L = np.log2(fpkm + 1)
    expressed = fpkm.index[(fpkm >= 1).sum(axis=1) >= 5]
    cond = np.array([1.0 if s.startswith("TN") else 0.0 for s in L.columns])
    sex = np.array([1.0 if GEO_SEX[s] == "M" else 0.0 for s in L.columns])
    gf = (fpkm.reindex(GLOBIN4).sum() / fpkm.sum()).values
    design = np.column_stack([np.ones(len(cond)), sex, (gf - gf.mean()) / gf.std(), cond])
    lt = limma_trend(L.loc[expressed], design, coef=3)
    lt.to_csv(C_OUT / "C4_TN_vs_control_limma_trend_sex_globin_adjusted.csv", index=False)
    SUMMARY["C4_limma_sex_globin_FDR05"] = int((lt.p_adj < 0.05).sum())
    SUMMARY["C4_limma_sex_globin_FDR10"] = int((lt.p_adj < 0.10).sum())

    summ = [
        ("Samples re-quantified", f"{len(samples)}"),
        ("Median read pairs processed", f"{qc.num_processed.median():,.0f}"),
        ("Median mapping rate (%)", f"{qc.percent_mapped.median():.1f} (min {qc.percent_mapped.min():.1f})"),
        ("Median per-sample Spearman, log TPM vs deposited log FPKM", f"{SUMMARY['median_spearman_TPM_vs_FPKM']:.3f}"),
        ("Globin fraction, salmon vs deposited (Spearman)", f"{SUMMARY['globin_fraction_salmon_vs_FPKM_spearman']:.3f}"),
        ("Globin fraction of counts, partition 1 vs 0 (median; exact P)",
         f"{SUMMARY['globin_fraction_counts_median_partition1']:.3f} vs {SUMMARY['globin_fraction_counts_median_partition0']:.3f}; "
         f"P = {SUMMARY['globin_fraction_counts_partition_MWU_p']:.4f}"),
    ]
    for name in res:
        summ.append((f"DESeq2 TN vs control ({name}): genes padj < 0.05 / < 0.10 (tested)",
                     f"{SUMMARY[f'C2_{name}_padj05']} / {SUMMARY[f'C2_{name}_padj10']} ({SUMMARY[f'C2_{name}_n_tested']:,})"))
    for r in rows:
        summ.append((f"Clustering: {r['variant']}",
                     f"ARI {r['ARI_vs_submitted']:.2f} ({r['sizes']}); silhouette {r['silhouette_k2']:.3f}, null P = {r['silhouette_null_p']:.2f}; "
                     f"gap k = {r['gap_selected_k']}; PC1 vs globin rho = {r['PC1_vs_globin_spearman']:.2f}"))
    summ.append(("limma-trend on deposited FPKM, ~ sex + globin + condition: genes FDR < 0.05 / < 0.10",
                 f"{SUMMARY['C4_limma_sex_globin_FDR05']} / {SUMMARY['C4_limma_sex_globin_FDR10']}"))
    pd.DataFrame(summ, columns=["metric", "value"]).to_csv(C_OUT / "C_summary_tables.csv", index=False)
    (C_OUT / "C_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
    print(pd.DataFrame(summ, columns=["metric", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
