"""
JHP major-revision analyses (reviewer-requested).

Part A  IoN-CCI re-analysis with SRA-verified sample labels (PRJNA991739), per-tissue
        PyDESeq2 contrasts, pathway over-representation, and rule-based module construction
        with MGI one-to-one mouse-human orthologs.
Part B  GSE186505 whole-blood analyses on log2(FPKM+1):
        B1 reproduction of the submitted k=2 partition
        B2 clusterability: k-selection metrics, gap statistic (k=1..4), SigClust-style Gaussian null
        B3 stability: leave-one-out, exhaustive 8/10 subsampling and bootstrap Jaccard, consensus/PAC,
           seed, number of PCs, gene filter, transform and algorithm sensitivity
        B4 circularity-aware nulls for module effects: full-pipeline Gaussian null and
           expression-matched random gene-set null; exact 210-labeling permutation
        B5 controls: module scores, projection onto the TN partition, joint clustering
        B6 reference-based deconvolution (ABIS RNA-seq signature, NNLS) and marker scores
        B7 sex check and technical/erythroid confounders
        B8 empirical-Bayes moderated (limma-trend) DE: TN vs control and subtype 1 vs 0 (+ sex)
        B9 preranked GSEA (moderated t) for both contrasts

All outputs: results/jhp_revision/
"""

from __future__ import annotations

import io
import itertools
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from scipy.special import digamma, polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "jhp_revision"
REF = OUT / "reference"
OUT.mkdir(parents=True, exist_ok=True)
REF.mkdir(parents=True, exist_ok=True)

EXPR_FILE = ROOT / "data" / "processed" / "GSE186505" / "human_blood_expr_processed.csv"
SUBTYPES_FILE = ROOT / "results" / "molecular_subtypes.csv"
ION = ROOT / "results" / "aim3" / "ion_cci" / "de_analysis"
SRA_META = ROOT / "data" / "external" / "ion_cci" / "sra_metadata.csv"

GEO_SEX = {
    "Control-1": "F", "Control-2": "F", "Control-3": "M", "Control-4": "F", "Control-5": "M",
    "Control-6": "F", "Control-7": "F", "Control-8": "F", "Control-9": "M", "Control-10": "M",
    "TN-2": "F", "TN-3": "F", "TN-4": "F", "TN-7": "F", "TN-9": "M",
    "TN-10": "F", "TN-11": "M", "TN-12": "F", "TN-13": "M", "TN-15": "M",
}

RNG_SEED = 20260926
SUMMARY: dict = {}
NULL_DRAWS: dict[str, np.ndarray] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        log(f"  downloading {url}")
        data = urllib.request.urlopen(url, timeout=120).read()
        dest.write_bytes(data)
    return dest


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / sp) if sp > 0 else np.nan


def load_gmt_enrichr(lib: str) -> dict[str, set[str]]:
    path = fetch(
        f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}",
        REF / f"{lib}.txt",
    )
    sets = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p for p in line.split("\t") if p]
        if len(parts) >= 3:
            sets[parts[0]] = {g.split(",")[0].upper() for g in parts[1:]}
    return sets


def load_mgi_orthologs() -> pd.DataFrame:
    """One-to-one mouse-human orthologs from the MGI/Alliance homology report."""
    path = fetch(
        "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
        REF / "HOM_MouseHumanSequence.rpt",
    )
    hom = pd.read_csv(path, sep="\t", dtype=str)
    key = "DB Class Key"
    mouse = hom[hom["NCBI Taxon ID"] == "10090"][[key, "Symbol"]].rename(columns={"Symbol": "mouse"})
    human = hom[hom["NCBI Taxon ID"] == "9606"][[key, "Symbol"]].rename(columns={"Symbol": "human"})
    m = mouse.merge(human, on=key)
    n_m = m.groupby(key)["mouse"].transform("nunique")
    n_h = m.groupby(key)["human"].transform("nunique")
    one2one = m[(n_m == 1) & (n_h == 1)].drop_duplicates(["mouse", "human"])
    return one2one[["mouse", "human"]]


# ---------------------------------------------------------------------------
# limma-style empirical Bayes moderated t (Smyth 2004), with optional mean trend
# ---------------------------------------------------------------------------
def trigamma_inverse(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, float))
    y = np.empty_like(x)
    big, small = x > 1e7, x < 1e-6
    y[big] = 1 / np.sqrt(x[big])
    y[small] = 1 / x[small]
    mid = ~(big | small)
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1 / xm
        for _ in range(50):
            tri = polygamma(1, ym)
            dif = tri * (1 - tri / xm) / polygamma(2, ym)
            ym = ym + dif
            if np.max(-dif / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def fit_f_dist(s2: np.ndarray, df: float, covariate: np.ndarray | None = None):
    s2 = np.maximum(s2, 1e-12)
    z = np.log(s2)
    e = z - digamma(df / 2) + np.log(df / 2)
    if covariate is None:
        emean = np.full_like(e, e.mean())
    else:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fitted = lowess(e, covariate, frac=0.4, return_sorted=False)
        emean = fitted
    evar = np.sum((e - emean) ** 2) / (len(e) - 1) - polygamma(1, df / 2)
    if evar > 0:
        d0 = float(2 * trigamma_inverse(np.array([evar]))[0])
        s0sq = np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0 = np.inf
        s0sq = np.exp(emean)
    return d0, s0sq


def limma_trend(Y: pd.DataFrame, design: np.ndarray, coef: int, trend: bool = True) -> pd.DataFrame:
    """Y: genes x samples (log scale). Returns moderated statistics for column `coef` of design."""
    X = np.asarray(design, float)
    Yv = Y.values
    n, p = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    B = Yv @ X @ XtXi
    resid = Yv - B @ X.T
    df = n - p
    s2 = (resid ** 2).sum(axis=1) / df
    amean = Yv.mean(axis=1)
    d0, s0sq = fit_f_dist(s2, df, amean if trend else None)
    if np.isinf(d0):
        s2post = s0sq
        dft = 1e6
    else:
        s2post = (d0 * s0sq + df * s2) / (d0 + df)
        dft = df + d0
    se = np.sqrt(s2post * XtXi[coef, coef])
    t = B[:, coef] / se
    pv = 2 * stats.t.sf(np.abs(t), dft)
    res = pd.DataFrame(
        {"gene": Y.index, "log2FC": B[:, coef], "AveExpr": amean, "t_moderated": t,
         "p_value": pv, "p_adj": bh(pv)}
    )
    res.attrs["d0"] = d0
    res.attrs["df_residual"] = df
    return res.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part A: IoN-CCI
# ---------------------------------------------------------------------------
REVISED_MODULES = {
    "peripheral_ecm_schwann": {
        "tissue": "tg", "direction": "up",
        "terms": ["Schwann Cell Myelination", "Extracellular Matrix Organization",
                  "Collagen Formation", "Collagen Chain Trimerization"],
    },
    "central_synaptic": {
        "tissue": "sp5c", "direction": "down",
        "terms": ["Neuronal System", "Transmission Across Chemical Synapses",
                  "Neurotransmitter Release Cycle", "GABA Synthesis"],
    },
    "ntrk_signaling": {
        "tissue": "sp5c", "direction": "down",
        "terms": ["Signaling By NTRKs", "Signaling By NTRK1"],
    },
}


def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame, factor: str, test: str, ref: str) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    keep = (counts >= 10).sum(axis=1) >= 3
    c = counts.loc[keep, meta.index].T.astype(int)
    inf = DefaultInference(n_cpus=4)
    dds = DeseqDataSet(counts=c, metadata=meta, design=f"~{factor}", refit_cooks=True, inference=inf, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=[factor, test, ref], inference=inf, quiet=True)
    ds.summary()
    return ds.results_df.copy()


def ora(genes: set[str], background: set[str], lib: dict[str, set[str]], min_size: int = 10) -> pd.DataFrame:
    genes = genes & background
    rows = []
    N, n = len(background), len(genes)
    for term, members in lib.items():
        m = members & background
        if len(m) < min_size:
            continue
        k = len(genes & m)
        if k == 0:
            continue
        p = stats.hypergeom.sf(k - 1, N, len(m), n)
        rows.append({"term": term, "overlap": k, "term_size": len(m), "n_query": n, "p_value": p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_adj"] = bh(df["p_value"].values)
    return df.sort_values("p_value").reset_index(drop=True)


def part_a_ion_cci(orth: pd.DataFrame, reactome: dict[str, set[str]]) -> dict[str, dict]:
    log("\n=== PART A: IoN-CCI re-analysis (PRJNA991739) ===")
    sra = pd.read_csv(SRA_META)
    meta = pd.DataFrame({
        "sample_id": sra["run_accession"],
        "tissue": np.where(sra["tissue"].str.contains("ganglia"), "tg", "sp5c"),
        "condition": np.where(sra["treatment"].str.upper() == "ION-CCI", "injury", "sham"),
    }).set_index("sample_id")
    legacy = pd.read_csv(ION / "ion_cci_metadata.csv").set_index("sample_id")
    audit = meta.join(legacy.rename(columns={"tissue": "legacy_tissue", "condition": "legacy_condition"}))
    audit["legacy_condition"] = audit["legacy_condition"].replace({"ion_cci": "injury"})
    audit["tissue_mislabeled"] = audit["tissue"] != audit["legacy_tissue"]
    audit["condition_mislabeled"] = audit["condition"] != audit["legacy_condition"]
    audit = audit.join(sra.set_index("run_accession")[["experiment_accession", "experiment_title"]])
    audit.to_csv(OUT / "A1_ion_cci_sample_label_audit.csv")
    n_bad = int((audit["tissue_mislabeled"] | audit["condition_mislabeled"]).sum())
    log(f"  mislabeled samples in legacy metadata: {n_bad}/12")

    counts = pd.read_csv(ION / "ion_cci_counts.csv", index_col=0)
    sym = pd.read_csv(ION / "ensembl_to_symbol_mapping.csv").drop_duplicates("ensembl_id").set_index("ensembl_id")["gene_symbol"]

    # expression-based verification of tissue identity
    lc = np.log2(counts.loc[(counts >= 10).sum(axis=1) >= 3] + 1)
    corr = lc.corr(method="spearman")
    km = KMeans(2, n_init=20, random_state=0).fit_predict(corr.values)
    ari_tissue = adjusted_rand_score(meta.loc[corr.index, "tissue"], km)
    ari_legacy = adjusted_rand_score(legacy.loc[corr.index, "tissue"], km)
    SUMMARY["ion_cci_expression_cluster_ARI_vs_SRA_tissue"] = ari_tissue
    SUMMARY["ion_cci_expression_cluster_ARI_vs_legacy_tissue"] = ari_legacy
    log(f"  expression clusters vs SRA tissue ARI={ari_tissue:.2f}; vs legacy tissue ARI={ari_legacy:.2f}")

    res = {}
    for tissue in ["tg", "sp5c"]:
        m = meta[meta["tissue"] == tissue]
        r = run_deseq(counts, m, "condition", "injury", "sham")
        r["gene_symbol"] = sym.reindex(r.index).values
        r.to_csv(OUT / f"A2_ion_cci_deseq2_{tissue}_injury_vs_sham.csv")
        res[tissue] = r
        n_up = int(((r.padj < 0.05) & (r.log2FoldChange > 0)).sum())
        n_dn = int(((r.padj < 0.05) & (r.log2FoldChange < 0)).sum())
        SUMMARY[f"ion_cci_{tissue}_DE_up_padj05"] = n_up
        SUMMARY[f"ion_cci_{tissue}_DE_down_padj05"] = n_dn
        log(f"  {tissue.upper()} injury vs sham (3 vs 3): up={n_up}, down={n_dn} (padj<0.05)")

    ms = meta[meta["condition"] == "sham"]
    rt = run_deseq(counts, ms, "tissue", "tg", "sp5c")
    rt["gene_symbol"] = sym.reindex(rt.index).values
    rt.to_csv(OUT / "A2_ion_cci_deseq2_sham_TG_vs_Sp5C_tissue_identity.csv")

    legacy_tg = pd.read_csv(ION / "de_ion_cci_vs_sham_tg.csv").set_index("gene") if (ION / "de_ion_cci_vs_sham_tg.csv").exists() else None
    if legacy_tg is not None and "log2FoldChange" in legacy_tg.columns:
        legacy_tg = legacy_tg.dropna(subset=["log2FoldChange"])
        common = legacy_tg.index.intersection(rt.index)
        common_inj = legacy_tg.index.intersection(res["tg"].index)
        r_tissue = stats.spearmanr(legacy_tg.loc[common, "log2FoldChange"], rt.loc[common, "log2FoldChange"], nan_policy="omit").correlation
        r_inj = stats.spearmanr(legacy_tg.loc[common_inj, "log2FoldChange"], res["tg"].loc[common_inj, "log2FoldChange"], nan_policy="omit").correlation
        sig_leg = legacy_tg[(legacy_tg.padj < 0.05)].index
        sig_tis = rt[(rt.padj < 0.05)].index
        SUMMARY["legacy_sig_genes_n"] = int(len(sig_leg))
        SUMMARY["legacy_sig_genes_also_TG_vs_Sp5C_tissue_sig_frac"] = float(np.mean(sig_leg.isin(sig_tis)))
        lu = legacy_tg.loc[legacy_tg.index.intersection(rt.index)]
        lu = lu[lu.padj < 0.05]
        SUMMARY["legacy_sig_same_sign_as_tissue_frac"] = float(np.mean(np.sign(lu.log2FoldChange) == np.sign(rt.loc[lu.index, "log2FoldChange"])))
        SUMMARY["legacy_ion_cci_LFC_vs_true_tissue_LFC_spearman"] = r_tissue
        SUMMARY["legacy_ion_cci_LFC_vs_true_TG_injury_LFC_spearman"] = r_inj
        log(f"  legacy 'IoN-CCI' LFC vs TG-vs-Sp5C tissue LFC rho={r_tissue:.2f}; vs true TG injury LFC rho={r_inj:.2f}")

    # ortholog mapping and ORA per tissue/direction
    o = orth.drop_duplicates("mouse").set_index("mouse")["human"]
    ora_rows = []
    modules = {}
    for tissue in ["tg", "sp5c"]:
        r = res[tissue].dropna(subset=["padj"])
        r = r.assign(human=o.reindex(r["gene_symbol"].astype(str)).values)
        bg = set(r["human"].dropna())
        for direction in ["up", "down"]:
            sel = r[(r.padj < 0.05) & ((r.log2FoldChange > 0) if direction == "up" else (r.log2FoldChange < 0))]
            q = set(sel["human"].dropna())
            e = ora(q, bg, reactome)
            if not e.empty:
                e.insert(0, "direction", direction)
                e.insert(0, "tissue", tissue)
                ora_rows.append(e)
            SUMMARY[f"ion_cci_{tissue}_{direction}_n_human_1to1"] = len(q)
    ora_df = pd.concat(ora_rows, ignore_index=True) if ora_rows else pd.DataFrame()
    ora_df.to_csv(OUT / "A3_ion_cci_reactome_ORA_by_tissue_direction.csv", index=False)

    for name, spec in REVISED_MODULES.items():
        r = res[spec["tissue"]].dropna(subset=["padj"])
        r = r.assign(human=o.reindex(r["gene_symbol"].astype(str)).values)
        up = spec["direction"] == "up"
        sel = r[(r.padj < 0.05) & ((r.log2FoldChange > 0) if up else (r.log2FoldChange < 0))]
        term_genes = set()
        matched_terms = []
        for term, genes in reactome.items():
            if any(t.lower() in term.lower() for t in spec["terms"]):
                term_genes |= genes
                matched_terms.append(term)
        pathway_module = sorted(set(sel["human"].dropna()) & term_genes)
        ranked = sel.dropna(subset=["human"]).sort_values("padj")
        signature_module = ranked["human"].drop_duplicates().head(100).tolist()
        tsub = ora_df[(ora_df.tissue == spec["tissue"]) & (ora_df.direction == spec["direction"])] if not ora_df.empty else pd.DataFrame()
        term_stats = tsub[tsub["term"].isin(matched_terms)][["term", "overlap", "term_size", "p_value", "p_adj"]] if not tsub.empty else pd.DataFrame()
        modules[name] = {
            "tissue": spec["tissue"], "direction": spec["direction"], "terms": matched_terms,
            "pathway_genes": pathway_module, "signature_genes": signature_module,
            "term_enrichment": term_stats.to_dict(orient="records"),
        }
        log(f"  revised {name}: {spec['tissue'].upper()} {spec['direction']} AND Reactome terms -> {len(pathway_module)} genes; "
            f"top-100 injury signature -> {len(signature_module)} genes")
    # sensitivity: central modules from TG-down (legacy intent 'TG/Sp5C')
    for name in ["central_synaptic", "ntrk_signaling"]:
        spec = REVISED_MODULES[name]
        r = res["tg"].dropna(subset=["padj"])
        r = r.assign(human=o.reindex(r["gene_symbol"].astype(str)).values)
        sel = r[(r.padj < 0.05) & (r.log2FoldChange < 0)]
        tg_genes = set()
        for term, genes in reactome.items():
            if any(t.lower() in term.lower() for t in spec["terms"]):
                tg_genes |= genes
        modules[f"{name}_TGdown"] = {
            "tissue": "tg", "direction": "down", "terms": spec["terms"],
            "pathway_genes": sorted(set(sel["human"].dropna()) & tg_genes),
            "signature_genes": sel.dropna(subset=["human"]).sort_values("padj")["human"].drop_duplicates().head(100).tolist(),
            "term_enrichment": [],
        }
    rows = []
    for name, m in modules.items():
        for kind in ["pathway_genes", "signature_genes"]:
            for g in m[kind]:
                rows.append({"module": name, "definition": kind, "tissue": m["tissue"], "direction": m["direction"], "human_gene": g})
    pd.DataFrame(rows).to_csv(OUT / "A4_revised_module_gene_sets.csv", index=False)
    with open(OUT / "A4_revised_module_definitions.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in modules.items()}, fh, indent=2, default=str)
    return modules


# ---------------------------------------------------------------------------
# Part B helpers
# ---------------------------------------------------------------------------
def pipeline_labels(Xs_samples: np.ndarray, k: int = 2, n_pcs: int | None = None, seed: int = 42,
                    n_init: int = 20, algo: str = "kmeans", standardize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    X = StandardScaler().fit_transform(Xs_samples) if standardize else Xs_samples - Xs_samples.mean(0)
    nmax = min(X.shape[0] - 1, X.shape[1] - 1, 50)
    nc = nmax if n_pcs is None else min(n_pcs, nmax)
    Z = PCA(n_components=nc, random_state=42).fit_transform(X)
    if algo == "kmeans":
        lab = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit_predict(Z)
    elif algo == "ward":
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    elif algo == "average_correlation":
        C = 1 - np.corrcoef(X)
        lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(C)
    else:
        raise ValueError(algo)
    return lab, Z


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def gap_statistic(Z: np.ndarray, kmax: int = 4, B: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def wk(X, k):
        if k == 1:
            return ((X - X.mean(0)) ** 2).sum()
        return KMeans(k, n_init=20, random_state=0).fit(X).inertia_

    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    Zp = Zc @ Vt.T
    lo, hi = Zp.min(0), Zp.max(0)
    rows = []
    for k in range(1, kmax + 1):
        logw = np.log(wk(Z, k))
        ref = np.array([np.log(wk(rng.uniform(lo, hi, size=Zp.shape) @ Vt, k)) for _ in range(B)])
        rows.append({"k": k, "log_W": logw, "E_log_W_ref": ref.mean(), "gap": ref.mean() - logw,
                     "s_k": ref.std(ddof=0) * np.sqrt(1 + 1 / B)})
    df = pd.DataFrame(rows)
    choose = None
    for i in range(len(df) - 1):
        if df.loc[i, "gap"] >= df.loc[i + 1, "gap"] - df.loc[i + 1, "s_k"]:
            choose = int(df.loc[i, "k"])
            break
    df["tibshirani_1SE_selected"] = df["k"] == (choose if choose else int(df.loc[df.gap.idxmax(), "k"]))
    return df


def module_scores(zmat: pd.DataFrame, genes: list[str], direction: str) -> pd.Series:
    g = [x for x in genes if x in zmat.index]
    if not g:
        return pd.Series(np.nan, index=zmat.columns)
    s = zmat.loc[g].mean(axis=0)
    return -s if direction == "down" else s


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    return df.sub(df.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)


MARKERS = {
    "Erythroid": ["HBB", "HBA1", "HBA2", "HBD", "ALAS2", "CA1", "SLC4A1", "AHSP"],
    "Platelet": ["PPBP", "PF4", "ITGA2B", "GP9", "TUBB1", "GP1BB", "CLEC1B"],
    "Neutrophil": ["FCGR3B", "CXCR2", "CSF3R", "FPR1", "CMTM2", "MME", "ALPL"],
    "Monocyte": ["CD14", "LYZ", "CSF1R", "FCN1", "VCAN"],
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2", "CD6"],
    "CD8_T": ["CD8A", "CD8B"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "CD22"],
    "NK": ["KLRF1", "NCR1", "SH2D1B", "KLRD1"],
    "Eosinophil": ["IL5RA", "SIGLEC8", "CCR3", "PRG2", "CLC"],
}
SEX_GENES = {"female": ["XIST", "TSIX"], "male": ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y"]}


def mwu_exact(a, b) -> float:
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
    except Exception:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def part_b(revised: dict[str, dict], reactome: dict[str, set[str]], gobp: dict[str, set[str]]) -> None:
    log("\n=== PART B: GSE186505 whole blood ===")
    rng = np.random.default_rng(RNG_SEED)
    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]
    st = pd.read_csv(SUBTYPES_FILE).set_index("sample_id")
    tn = st.index.tolist()
    ctrl = [c for c in fpkm.columns if c.startswith("Control")]
    y = st.loc[tn, "subtype"].values
    L = np.log2(fpkm + 1)
    SUMMARY["n_genes_matrix"] = int(fpkm.shape[0])
    expressed = fpkm.index[(fpkm >= 1).sum(axis=1) >= 5]
    SUMMARY["n_genes_expressed_FPKM1_in5"] = int(len(expressed))

    # --- B1 reproduction -------------------------------------------------
    raw_tn = fpkm[tn].T.values
    lab0, Z0 = pipeline_labels(raw_tn)
    ari_repro = adjusted_rand_score(y, lab0)
    SUMMARY["B1_reproduction_ARI"] = ari_repro
    log(f"  B1 reproduction of submitted partition (raw FPKM pipeline): ARI={ari_repro:.2f}")
    nz = (fpkm[tn].std(axis=1) > 0).sum()
    SUMMARY["n_genes_nonzero_variance_TN"] = int(nz)

    # --- B2 clusterability -----------------------------------------------
    krows = []
    for k in range(2, 5):
        lab = KMeans(k, random_state=42, n_init=20).fit_predict(Z0)
        krows.append({"k": k, "within_SS": KMeans(k, random_state=42, n_init=20).fit(Z0).inertia_,
                      "silhouette": silhouette_score(Z0, lab), "calinski_harabasz": calinski_harabasz_score(Z0, lab),
                      "sizes": "/".join(map(str, sorted(np.bincount(lab), reverse=True)))})
    kdf = pd.DataFrame(krows)
    lamb = Z0.var(axis=0, ddof=1)
    # SigClust-style: single multivariate Gaussian with the observed PC variances
    B = 2000
    null_sil = {k: [] for k in range(2, 5)}
    null_ch = {k: [] for k in range(2, 5)}
    null_ci, null_best = [], []
    for _ in range(B):
        Zn = rng.normal(size=Z0.shape) * np.sqrt(lamb)
        best = -1
        for k in range(2, 5):
            km = KMeans(k, random_state=42, n_init=20).fit(Zn)
            s = silhouette_score(Zn, km.labels_)
            null_sil[k].append(s)
            null_ch[k].append(calinski_harabasz_score(Zn, km.labels_))
            best = max(best, s)
            if k == 2:
                null_ci.append(km.inertia_ / ((Zn - Zn.mean(0)) ** 2).sum())
        null_best.append(best)
    obs_ci = KMeans(2, random_state=42, n_init=20).fit(Z0).inertia_ / ((Z0 - Z0.mean(0)) ** 2).sum()
    for i, k in enumerate(range(2, 5)):
        kdf.loc[i, "null_silhouette_mean"] = np.mean(null_sil[k])
        kdf.loc[i, "null_silhouette_95pct"] = np.percentile(null_sil[k], 95)
        kdf.loc[i, "silhouette_null_p"] = (1 + np.sum(np.array(null_sil[k]) >= kdf.loc[i, "silhouette"])) / (B + 1)
        kdf.loc[i, "null_CH_mean"] = np.mean(null_ch[k])
        kdf.loc[i, "CH_null_p"] = (1 + np.sum(np.array(null_ch[k]) >= kdf.loc[i, "calinski_harabasz"])) / (B + 1)
    kdf.to_csv(OUT / "B2_k_selection_metrics_with_null.csv", index=False)
    NULL_DRAWS["sigclust_best_silhouette"] = np.array(null_best)
    NULL_DRAWS["sigclust_silhouette_k2"] = np.array(null_sil[2])
    NULL_DRAWS["sigclust_cluster_index_k2"] = np.array(null_ci)
    p_ci = (1 + np.sum(np.array(null_ci) <= obs_ci)) / (B + 1)
    p_best = (1 + np.sum(np.array(null_best) >= kdf["silhouette"].max())) / (B + 1)
    SUMMARY.update({"B2_obs_2means_cluster_index": obs_ci, "B2_sigclust_p_cluster_index": p_ci,
                    "B2_null_best_silhouette_mean": float(np.mean(null_best)),
                    "B2_null_best_silhouette_95pct": float(np.percentile(null_best, 95)),
                    "B2_p_best_silhouette": p_best})
    gap = gap_statistic(Z0, 4, 500)
    gap.to_csv(OUT / "B2_gap_statistic_k1_4.csv", index=False)
    SUMMARY["B2_gap_selected_k"] = int(gap.loc[gap.tibshirani_1SE_selected, "k"].iloc[0])
    log(f"  B2 k=2 silhouette={kdf.silhouette[0]:.3f}; null best-silhouette mean={np.mean(null_best):.3f}, p={p_best:.3f}; "
        f"SigClust CI p={p_ci:.3f}; gap-selected k={SUMMARY['B2_gap_selected_k']}")
    # same on log scale
    Zlog = pipeline_labels(L[tn].T.values)[1]
    lamb_l = Zlog.var(axis=0, ddof=1)
    obs_sil_log = silhouette_score(Zlog, KMeans(2, random_state=42, n_init=20).fit_predict(Zlog))
    nl = [silhouette_score(Zn, KMeans(2, random_state=42, n_init=20).fit_predict(Zn))
          for Zn in (rng.normal(size=Zlog.shape) * np.sqrt(lamb_l) for _ in range(1000))]
    SUMMARY["B2_log_k2_silhouette"] = obs_sil_log
    SUMMARY["B2_log_k2_silhouette_null_p"] = (1 + np.sum(np.array(nl) >= obs_sil_log)) / 1001
    gap_l = gap_statistic(Zlog, 4, 500, seed=1)
    gap_l.to_csv(OUT / "B2_gap_statistic_k1_4_log2.csv", index=False)
    SUMMARY["B2_gap_selected_k_log2"] = int(gap_l.loc[gap_l.tibshirani_1SE_selected, "k"].iloc[0])

    # --- B3 stability ----------------------------------------------------
    stab = []
    # LOO
    loo_rows = []
    for i, s in enumerate(tn):
        keep = [j for j in range(10) if j != i]
        lab, Zs = pipeline_labels(raw_tn[keep])
        sils = {k: silhouette_score(Zs, KMeans(k, random_state=42, n_init=20).fit_predict(Zs)) for k in range(2, 5)}
        loo_rows.append({"left_out": s, "left_out_subtype": int(y[i]), "ARI_vs_original": adjusted_rand_score(y[keep], lab),
                         "silhouette_k2": sils[2], "k_selected_by_silhouette": max(sils, key=sils.get)})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT / "B3_leave_one_out.csv", index=False)
    SUMMARY["B3_LOO_ARI_mean"] = float(loo.ARI_vs_original.mean())
    SUMMARY["B3_LOO_n_identical"] = int((loo.ARI_vs_original > 0.999).sum())
    SUMMARY["B3_LOO_n_k2_selected"] = int((loo.k_selected_by_silhouette == 2).sum())
    # exhaustive 8/10 subsampling + bootstrap: Hennig cluster-wise Jaccard
    orig = {c: set(np.where(y == c)[0]) for c in (0, 1)}

    def cw_jaccard(idx, lab):
        out = {}
        for c in (0, 1):
            oc = orig[c] & set(idx)
            best = 0
            for cc in np.unique(lab):
                members = {idx[j] for j in range(len(idx)) if lab[j] == cc}
                best = max(best, jaccard(oc, members))
            out[c] = best
        return out

    sub_j = []
    cons = np.zeros((10, 10))
    cnt = np.zeros((10, 10))
    for idx in itertools.combinations(range(10), 8):
        idx = list(idx)
        lab, _ = pipeline_labels(raw_tn[idx])
        sub_j.append(cw_jaccard(idx, lab))
        for a, b in itertools.combinations(range(8), 2):
            ia, ib = idx[a], idx[b]
            cnt[ia, ib] += 1
            cnt[ib, ia] += 1
            if lab[a] == lab[b]:
                cons[ia, ib] += 1
                cons[ib, ia] += 1
    boot_j = []
    for _ in range(1000):
        idx = rng.choice(10, 10, replace=True)
        if len(np.unique(idx)) < 4:
            continue
        lab, _ = pipeline_labels(raw_tn[idx])
        uniq = {}
        for j, ii in enumerate(idx):
            uniq.setdefault(ii, lab[j])
        u_idx = list(uniq.keys())
        boot_j.append(cw_jaccard(u_idx, np.array([uniq[i] for i in u_idx])))
    sj, bj = pd.DataFrame(sub_j), pd.DataFrame(boot_j)
    C = np.divide(cons, cnt, out=np.ones_like(cons), where=cnt > 0)
    np.fill_diagonal(C, 1)
    pd.DataFrame(C, index=tn, columns=tn).to_csv(OUT / "B3_consensus_matrix_subsampling.csv")
    iu = np.triu_indices(10, 1)
    pac = float(np.mean((C[iu] > 0.1) & (C[iu] < 0.9)))
    jac = pd.DataFrame({
        "cluster": ["Subtype 0 (n=6)", "Subtype 1 (n=4)"],
        "subsample_8of10_mean_jaccard": [sj[0].mean(), sj[1].mean()],
        "subsample_8of10_frac_ge_0.75": [(sj[0] >= 0.75).mean(), (sj[1] >= 0.75).mean()],
        "bootstrap_mean_jaccard": [bj[0].mean(), bj[1].mean()],
        "bootstrap_frac_dissolved_lt_0.5": [(bj[0] < 0.5).mean(), (bj[1] < 0.5).mean()],
    })
    jac.to_csv(OUT / "B3_clusterwise_jaccard.csv", index=False)
    SUMMARY["B3_PAC_0.1_0.9"] = pac
    SUMMARY["B3_jaccard"] = jac.to_dict(orient="records")
    log(f"  B3 LOO: {SUMMARY['B3_LOO_n_identical']}/10 identical, mean ARI={SUMMARY['B3_LOO_ARI_mean']:.2f}; "
        f"Jaccard sub0={sj[0].mean():.2f}, sub1={sj[1].mean():.2f}; bootstrap {bj[0].mean():.2f}/{bj[1].mean():.2f}; PAC={pac:.2f}")
    # seed
    seed_same = [adjusted_rand_score(y, pipeline_labels(raw_tn, seed=s, n_init=1)[0]) for s in range(200)]
    seed_same20 = [adjusted_rand_score(y, pipeline_labels(raw_tn, seed=s, n_init=20)[0]) for s in range(50)]
    stab.append({"variant": "random seed (200 seeds, n_init=1)", "ARI_vs_original": np.mean(seed_same),
                 "frac_identical": float(np.mean(np.array(seed_same) > 0.999))})
    stab.append({"variant": "random seed (50 seeds, n_init=20)", "ARI_vs_original": np.mean(seed_same20),
                 "frac_identical": float(np.mean(np.array(seed_same20) > 0.999))})
    # PCs / filters / transform / algorithm
    var_log = L[tn].var(axis=1).sort_values(ascending=False)
    variants = [
        ("original: z(FPKM), all genes, 9 PCs, k-means", raw_tn, dict()),
        ("z(FPKM), 2 PCs", raw_tn, dict(n_pcs=2)),
        ("z(FPKM), 3 PCs", raw_tn, dict(n_pcs=3)),
        ("z(FPKM), 5 PCs", raw_tn, dict(n_pcs=5)),
        ("z(FPKM), Ward", raw_tn, dict(algo="ward")),
        ("z(FPKM), average-linkage correlation", raw_tn, dict(algo="average_correlation")),
        ("z(log2 FPKM+1), all genes", L[tn].T.values, dict()),
        ("z(log2 FPKM+1), expressed genes (FPKM>=1 in >=5 samples)", L.loc[expressed, tn].T.values, dict()),
        ("log2 FPKM+1 unscaled, expressed genes", L.loc[expressed, tn].T.values, dict(standardize=False)),
    ]
    for n in (500, 1000, 2000, 5000):
        g = var_log.index[:n]
        variants.append((f"z(log2 FPKM+1), top {n} variable genes", L.loc[g, tn].T.values, dict()))
        variants.append((f"log2 FPKM+1 unscaled, top {n} variable genes", L.loc[g, tn].T.values, dict(standardize=False)))
    for name, X, kw in variants:
        lab, Z = pipeline_labels(X, **kw)
        stab.append({"variant": name, "ARI_vs_original": adjusted_rand_score(y, lab),
                     "frac_identical": float(adjusted_rand_score(y, lab) > 0.999),
                     "sizes": "/".join(map(str, sorted(np.bincount(lab), reverse=True))),
                     "silhouette": silhouette_score(Z, lab) if len(np.unique(lab)) > 1 else np.nan})
    sdf = pd.DataFrame(stab)
    sdf.to_csv(OUT / "B3_sensitivity_variants.csv", index=False)
    SUMMARY["B3_n_variants_identical"] = f"{int((sdf.frac_identical == 1).sum() - 2 * 0)}/{len(sdf)}"

    # --- module definitions for testing ---------------------------------
    legacy_map = pd.read_csv(ROOT / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")
    legacy_dir = {"peripheral_ecm_schwann": "up", "central_synaptic": "down", "ntrk_signaling": "down"}
    mod_sets = {}
    for mname, d in legacy_dir.items():
        genes = legacy_map[legacy_map.module == mname]["human_symbol"].str.upper().unique().tolist()
        mod_sets[f"legacy_{mname}"] = (genes, d)
    for mname, m in revised.items():
        mod_sets[f"revised_{mname}_pathway"] = (m["pathway_genes"], m["direction"])
        mod_sets[f"revised_{mname}_signature"] = (m["signature_genes"], m["direction"])

    # reproduce legacy statistic exactly (z over 20 samples of raw FPKM, all genes present)
    z_all_raw = zscore_rows(fpkm)
    rep = []
    for mname in legacy_dir:
        genes, d = mod_sets[f"legacy_{mname}"]
        s = module_scores(z_all_raw, genes, d)[tn]
        rep.append({"module": mname, "d_reproduced": cohens_d(s[y == 0], s[y == 1])})
    pd.DataFrame(rep).to_csv(OUT / "B4_legacy_effect_reproduction.csv", index=False)

    # coverage / expression of module genes in blood
    cov = []
    mean_f = fpkm.mean(axis=1)
    for name, (genes, d) in mod_sets.items():
        g = [x for x in genes if x in fpkm.index]
        cov.append({"module": name, "direction_in_injury": d, "n_human_orthologs": len(genes),
                    "n_in_blood_matrix": len(g),
                    "n_mean_FPKM_ge_0.5": int((mean_f.reindex(g) >= 0.5).sum()),
                    "n_mean_FPKM_ge_1": int((mean_f.reindex(g) >= 1).sum()),
                    "n_mean_FPKM_ge_5": int((mean_f.reindex(g) >= 5).sum()),
                    "median_FPKM_module_genes": float(mean_f.reindex(g).median()) if g else np.nan,
                    "median_FPKM_all_genes": float(mean_f.median()),
                    "genes_mean_FPKM_ge_1": "; ".join(sorted([x for x in g if mean_f[x] >= 1]))})
    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(OUT / "B4_module_ortholog_blood_expression.csv", index=False)

    # --- B4 circularity-aware nulls ---------------------------------------
    # observed statistics use expressed module genes (mean FPKM>=1), z over TN only, log2 scale
    Ltn = L[tn]
    z_tn = zscore_rows(Ltn)
    z_tn_raw = zscore_rows(fpkm[tn])
    Xs = StandardScaler().fit_transform(raw_tn)
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    r = int(np.sum(S > 1e-8 * S[0]))
    S, Vt = S[:r], Vt[:r]
    gene_index = fpkm.index
    all_labels = [np.array(c) for c in itertools.combinations(range(10), 4)]

    def expressed_genes(genes):
        return [g for g in genes if g in mean_f.index and mean_f[g] >= 1]

    res_rows = []
    B_full = 1000
    null_d_store = {}
    # full-pipeline null (standardized raw-FPKM gene space, as in the submitted pipeline)
    test_sets = {n: expressed_genes(g) for n, (g, d) in mod_sets.items()}
    test_idx = {n: np.array([gene_index.get_loc(g) for g in gs]) for n, gs in test_sets.items() if len(gs) >= 3}
    submitted_sets = {}
    for mname, d in legacy_dir.items():
        gs = [g for g in mod_sets[f"legacy_{mname}"][0] if g in gene_index]
        submitted_sets[f"submitted_exact_{mname}"] = (gs, d)
        test_idx[f"submitted_exact_{mname}"] = np.array([gene_index.get_loc(g) for g in gs])
    null_abs_d = {n: [] for n in test_idx}
    null_sil2 = []
    for b in range(B_full):
        Zr = rng.normal(size=(10, r))
        Xn = Zr @ np.diag(S / np.sqrt(10)) @ Vt
        Xn = StandardScaler().fit_transform(Xn)
        Zp = PCA(n_components=min(9, r), random_state=42).fit_transform(Xn)
        lab = KMeans(2, random_state=42, n_init=20).fit_predict(Zp)
        null_sil2.append(silhouette_score(Zp, lab))
        for n, ix in test_idx.items():
            sc = Xn[:, ix].mean(axis=1)
            null_abs_d[n].append(abs(cohens_d(sc[lab == 0], sc[lab == 1])))
    for name, (genes, d) in mod_sets.items():
        g_expr = test_sets[name]
        if len(g_expr) < 3:
            res_rows.append({"module": name, "n_expressed_genes": len(g_expr)})
            continue
        s_raw = module_scores(z_tn_raw, g_expr, d)
        s_log = module_scores(z_tn, g_expr, d)
        raw_mean = z_tn.loc[g_expr].mean(axis=0)
        d_raw = cohens_d(s_raw[y == 0], s_raw[y == 1])
        d_log = cohens_d(s_log[y == 0], s_log[y == 1])
        # exact permutation over all 210 labelings (log scale)
        obs_delta = abs(s_log[y == 1].mean() - s_log[y == 0].mean())
        perm = []
        for idx1 in all_labels:
            lab = np.zeros(10, int)
            lab[idx1] = 1
            perm.append(abs(s_log[lab == 1].mean() - s_log[lab == 0].mean()))
        p_exact = float(np.mean(np.array(perm) >= obs_delta - 1e-12))
        # full pipeline null (compare |d| in standardized raw space)
        nd = np.array(null_abs_d[name])
        nd = nd[~np.isnan(nd)]  # singleton clusters leave Cohen's d undefined
        p_full = (1 + np.sum(nd >= abs(d_raw))) / (len(nd) + 1)
        # expression-matched random gene sets (log scale, observed labels)
        amean = Ltn.mean(axis=1)
        pool = amean[amean.index.isin(mean_f.index[mean_f >= 1])]
        bins = pd.qcut(pool, 10, labels=False, duplicates="drop")
        need = bins.reindex(g_expr).value_counts()
        rs_d = []
        zv = z_tn
        by_bin = {bn: pool.index[bins == bn].values for bn in need.index}
        for _ in range(10000):
            gs = np.concatenate([rng.choice(by_bin[bn], int(c), replace=False) for bn, c in need.items()])
            sc = zv.loc[gs].mean(axis=0).values
            rs_d.append(abs(cohens_d(sc[y == 0], sc[y == 1])))
        rs_d = np.array(rs_d)
        NULL_DRAWS[f"randset_{name}"] = rs_d
        p_rs = (1 + np.sum(rs_d >= abs(d_log))) / 10001
        res_rows.append({
            "module": name, "direction_in_injury": d, "n_expressed_genes": len(g_expr),
            "raw_mean_z_log_subtype0": float(raw_mean[y == 0].mean()),
            "raw_mean_z_log_subtype1": float(raw_mean[y == 1].mean()),
            "cohens_d_injury_oriented_zFPKM": d_raw, "cohens_d_injury_oriented_log2": d_log,
            "exact_perm_p_210_labelings": p_exact, "perm_p_floor": 1 / 210,
            "full_pipeline_null_n_valid_splits": int(len(nd)),
            "full_pipeline_null_median_abs_d": float(np.median(nd)),
            "full_pipeline_null_95pct_abs_d": float(np.percentile(nd, 95)),
            "full_pipeline_null_p": p_full,
            "random_geneset_median_abs_d": float(np.median(rs_d)),
            "random_geneset_95pct_abs_d": float(np.percentile(rs_d, 95)),
            "random_geneset_p": p_rs,
        })
    sub_rows = []
    for name, (gs, d) in submitted_sets.items():
        s = module_scores(z_all_raw, gs, d)[tn]
        d_obs = cohens_d(s[y == 0], s[y == 1])
        nd = np.array(null_abs_d[name])
        nd = nd[~np.isnan(nd)]
        sub_rows.append({"module": name, "n_genes_as_submitted": len(gs), "submitted_cohens_d": d_obs,
                         "null_n_valid_splits": int(len(nd)), "null_median_abs_d": float(np.median(nd)),
                         "null_95pct_abs_d": float(np.percentile(nd, 95)), "null_99pct_abs_d": float(np.percentile(nd, 99)),
                         "full_pipeline_null_p": (1 + np.sum(nd >= abs(d_obs))) / (len(nd) + 1)})
    pd.DataFrame(sub_rows).to_csv(OUT / "B4b_submitted_gene_sets_full_pipeline_null.csv", index=False)
    SUMMARY["B4b_submitted_modules_full_pipeline_null"] = sub_rows
    NULL_DRAWS.update({f"fullnull_{n}": np.array(v) for n, v in null_abs_d.items()})
    NULL_DRAWS["fullnull_silhouette_k2"] = np.array(null_sil2)
    mres = pd.DataFrame(res_rows)
    mres.to_csv(OUT / "B4_module_effects_circularity_nulls.csv", index=False)
    SUMMARY["B4_full_pipeline_null_silhouette_mean"] = float(np.mean(null_sil2))
    log("  B4 module nulls:")
    log(mres[[c for c in ["module", "n_expressed_genes", "cohens_d_injury_oriented_zFPKM", "full_pipeline_null_p",
                          "random_geneset_p", "exact_perm_p_210_labelings"] if c in mres.columns]].to_string(index=False))

    # --- B5 controls ------------------------------------------------------
    z20 = zscore_rows(L)
    groups = pd.Series("Control", index=L.columns)
    groups[tn] = ["Subtype 0" if v == 0 else "Subtype 1" for v in y]
    ctl_rows = []
    score_tab = pd.DataFrame(index=L.columns)
    for name, (genes, d) in mod_sets.items():
        g_expr = test_sets[name]
        if len(g_expr) < 3:
            continue
        s = module_scores(z20, g_expr, d)
        score_tab[name] = s
        a, b0, b1 = s[groups == "Control"], s[groups == "Subtype 0"], s[groups == "Subtype 1"]
        ctl_rows.append({"module": name, "control_mean": a.mean(), "subtype0_mean": b0.mean(), "subtype1_mean": b1.mean(),
                         "kruskal_p": stats.kruskal(a, b0, b1).pvalue,
                         "TN_vs_control_MWU_p": mwu_exact(a, pd.concat([b0, b1])),
                         "control_vs_subtype0_MWU_p": mwu_exact(a, b0), "control_vs_subtype1_MWU_p": mwu_exact(a, b1),
                         "control_range": f"{a.min():.2f} to {a.max():.2f}",
                         "subtype0_range": f"{b0.min():.2f} to {b0.max():.2f}",
                         "subtype1_range": f"{b1.min():.2f} to {b1.max():.2f}"})
    score_tab.insert(0, "group", groups)
    score_tab.insert(1, "sex_GEO", [GEO_SEX[s] for s in score_tab.index])
    score_tab.to_csv(OUT / "B5_module_scores_all20_log2z.csv")
    pd.DataFrame(ctl_rows).to_csv(OUT / "B5_module_scores_controls_vs_subtypes.csv", index=False)
    # projection of controls onto TN partition
    sc = StandardScaler().fit(raw_tn)
    pca = PCA(n_components=9, random_state=42).fit(sc.transform(raw_tn))
    km = KMeans(2, random_state=42, n_init=20).fit(pca.transform(sc.transform(raw_tn)))
    # align km labels with subtype labels
    flip = adjusted_rand_score(y, km.labels_) > 0.999 and np.all(km.labels_ == y)
    pc_ctrl = pca.transform(sc.transform(fpkm[ctrl].T.values))
    assign = km.predict(pc_ctrl)
    if not flip:
        assign = 1 - assign if np.all(km.labels_ == 1 - y) else assign
    proj = pd.DataFrame({"sample": ctrl, "assigned_subtype": assign, "PC1": pc_ctrl[:, 0], "PC2": pc_ctrl[:, 1],
                         "sex_GEO": [GEO_SEX[c] for c in ctrl]})
    proj.to_csv(OUT / "B5_controls_projected_onto_TN_partition.csv", index=False)
    SUMMARY["B5_controls_assigned_subtype0"] = int((assign == 0).sum())
    SUMMARY["B5_controls_assigned_subtype1"] = int((assign == 1).sum())
    # log-scale projection as sensitivity
    Ltn_T, Lc_T = L.loc[expressed, tn].T.values, L.loc[expressed, ctrl].T.values
    scl = StandardScaler().fit(Ltn_T)
    pcl = PCA(n_components=9, random_state=42).fit(scl.transform(Ltn_T))
    Ztn_l = pcl.transform(scl.transform(Ltn_T))
    cent = np.vstack([Ztn_l[y == c].mean(0) for c in (0, 1)])
    Zc_l = pcl.transform(scl.transform(Lc_T))
    a_l = np.argmin(((Zc_l[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    SUMMARY["B5_controls_assigned_subtype1_log2_expressed"] = int((a_l == 1).sum())
    # joint clustering of all 20
    lab20, _ = pipeline_labels(L.loc[expressed].T.values)
    diag = np.array([0 if c.startswith("Control") else 1 for c in L.columns])
    SUMMARY["B5_joint_k2_ARI_vs_diagnosis"] = adjusted_rand_score(diag, lab20)
    log(f"  B5 controls projected: {SUMMARY['B5_controls_assigned_subtype0']} -> Subtype 0, "
        f"{SUMMARY['B5_controls_assigned_subtype1']} -> Subtype 1; joint k=2 ARI vs diagnosis={SUMMARY['B5_joint_k2_ARI_vs_diagnosis']:.2f}")

    # --- B6 deconvolution ------------------------------------------------
    sig = pd.read_csv(fetch("https://raw.githubusercontent.com/giannimonaco/ABIS/master/data/sigmatrixRNAseq.txt",
                            REF / "ABIS_sigmatrixRNAseq.txt"), sep="\t", index_col=0)
    sig.index = sig.index.astype(str).str.upper().str.strip('"')
    sig.columns = [c.strip('"') for c in sig.columns]
    tpm = fpkm / fpkm.sum(axis=0) * 1e6
    common = sig.index.intersection(tpm.index)
    A = sig.loc[common].values
    dec_rows = []
    for s in tpm.columns:
        bvec = tpm.loc[common, s].values
        coef, _ = nnls(A, bvec)
        fit = A @ coef
        r_fit = stats.pearsonr(fit, bvec)[0]
        rmse = np.sqrt(np.mean((fit - bvec) ** 2)) / np.mean(bvec)
        frac = coef / coef.sum() if coef.sum() > 0 else coef
        dec_rows.append(dict(zip(sig.columns, frac)) | {"sample": s, "fit_pearson_r": r_fit, "fit_nRMSE": rmse})
    dec = pd.DataFrame(dec_rows).set_index("sample")
    dec.insert(0, "group", groups.reindex(dec.index))
    dec.to_csv(OUT / "B6_ABIS_NNLS_deconvolution_all20.csv")
    SUMMARY["B6_ABIS_signature_genes_used"] = int(len(common))
    SUMMARY["B6_fit_r_median"] = float(dec["fit_pearson_r"].median())
    # marker scores (mean log2 FPKM+1 of markers)
    mk = pd.DataFrame(index=L.columns)
    for cell, genes in MARKERS.items():
        g = [x for x in genes if x in L.index]
        mk[cell] = L.loc[g].mean(axis=0)
    mk["globin_fraction_of_total_FPKM"] = fpkm.reindex(["HBB", "HBA1", "HBA2", "HBD"]).sum() / fpkm.sum()
    # tests
    cell_cols = list(sig.columns)
    trows = []
    features = [(c, "ABIS_NNLS_fraction", dec[c]) for c in cell_cols]
    features += [(c, "marker_score", mk[c]) for c in list(MARKERS) + ["globin_fraction_of_total_FPKM"]]
    for c, src, v in features:
        v0, v1, vc = v[groups == "Subtype 0"], v[groups == "Subtype 1"], v[groups == "Control"]
        trows.append({"feature": c, "source": src,
                      "control_mean": vc.mean(), "subtype0_mean": v0.mean(), "subtype1_mean": v1.mean(),
                      "subtype1_vs_0_MWU_exact_p": mwu_exact(v0, v1),
                      "TN_vs_control_MWU_p": mwu_exact(vc, pd.concat([v0, v1]))})
    tdf = pd.DataFrame(trows)
    for col in ["subtype1_vs_0_MWU_exact_p", "TN_vs_control_MWU_p"]:
        for src in tdf.source.unique():
            m = tdf.source == src
            tdf.loc[m, col.replace("_p", "_q")] = bh(tdf.loc[m, col].values)
    tdf.to_csv(OUT / "B6_cell_composition_tests.csv", index=False)
    mk.insert(0, "group", groups)
    mk.to_csv(OUT / "B6_marker_scores_all20.csv")
    log(f"  B6 ABIS NNLS: {len(common)} signature genes, median fit r={SUMMARY['B6_fit_r_median']:.2f}; "
        f"min subtype q={tdf['subtype1_vs_0_MWU_exact_q'].min():.3f}")

    # --- B7 sex and confounders -------------------------------------------
    sx = pd.DataFrame(index=L.columns)
    sx["XIST"] = L.loc["XIST"] if "XIST" in L.index else np.nan
    ygenes = [g for g in SEX_GENES["male"] if g in L.index]
    sx["Ychr_mean"] = L.loc[ygenes].mean(axis=0)
    # XIST is absent from the deposited GEO matrix; call sex from Y-linked genes (log2 FPKM+1 > 1)
    sx["sex_predicted"] = np.where(sx["Ychr_mean"] > 1, "M", "F")
    sx["sex_GEO"] = [GEO_SEX[s] for s in sx.index]
    sx["concordant"] = sx["sex_predicted"] == sx["sex_GEO"]
    sx["group"] = groups
    sx.to_csv(OUT / "B7_sex_check.csv")
    SUMMARY["B7_sex_concordant"] = f"{int(sx.concordant.sum())}/20"
    t_sex = pd.crosstab(sx.loc[tn, "group"], sx.loc[tn, "sex_GEO"])
    SUMMARY["B7_sex_by_subtype"] = t_sex.to_dict()
    SUMMARY["B7_sex_by_subtype_fisher_p"] = float(stats.fisher_exact(t_sex.values)[1]) if t_sex.shape == (2, 2) else np.nan
    t_sex_cc = pd.crosstab(np.where(sx.group == "Control", "Control", "TN"), sx["sex_GEO"])
    SUMMARY["B7_sex_TN_vs_control_fisher_p"] = float(stats.fisher_exact(t_sex_cc.values)[1])
    tech = pd.DataFrame(index=L.columns)
    tech["genes_detected_FPKM_gt_0.1"] = (fpkm > 0.1).sum(axis=0)
    tech["total_FPKM"] = fpkm.sum(axis=0)
    mt = [g for g in fpkm.index if g.startswith("MT-")]
    tech["mito_fraction"] = fpkm.loc[mt].sum() / fpkm.sum() if mt else np.nan
    tech["top100_gene_fraction"] = fpkm.apply(lambda c: c.nlargest(100).sum() / c.sum())
    tech = tech.join(mk[["Erythroid", "Platelet", "Neutrophil", "globin_fraction_of_total_FPKM"]])
    tech["group"] = groups
    pcs = pd.DataFrame(Z0[:, :3], index=tn, columns=["PC1", "PC2", "PC3"])
    crow = []
    for c in [x for x in tech.columns if x != "group"]:
        v = tech.loc[tn, c]
        rho1 = stats.spearmanr(v, pcs["PC1"])
        rho2 = stats.spearmanr(v, pcs["PC2"])
        crow.append({"feature": c, "subtype0_median": v[y == 0].median(), "subtype1_median": v[y == 1].median(),
                     "subtype_MWU_exact_p": mwu_exact(v[y == 0], v[y == 1]),
                     "spearman_PC1": rho1.correlation, "p_PC1": rho1.pvalue,
                     "spearman_PC2": rho2.correlation, "p_PC2": rho2.pvalue,
                     "control_median": tech.loc[ctrl, c].median(),
                     "TN_vs_control_MWU_p": mwu_exact(tech.loc[ctrl, c], v)})
    sexnum = (sx.loc[tn, "sex_GEO"] == "M").astype(int)
    crow.append({"feature": "sex (male=1)", "subtype0_median": sexnum[y == 0].mean(), "subtype1_median": sexnum[y == 1].mean(),
                 "subtype_MWU_exact_p": SUMMARY["B7_sex_by_subtype_fisher_p"],
                 "spearman_PC1": stats.spearmanr(sexnum, pcs["PC1"]).correlation, "p_PC1": stats.spearmanr(sexnum, pcs["PC1"]).pvalue,
                 "spearman_PC2": stats.spearmanr(sexnum, pcs["PC2"]).correlation, "p_PC2": stats.spearmanr(sexnum, pcs["PC2"]).pvalue})
    tech.to_csv(OUT / "B7_technical_covariates_all20.csv")
    pd.DataFrame(crow).to_csv(OUT / "B7_confounder_tests_TN.csv", index=False)
    pcs.join(st[["subtype"]]).to_csv(OUT / "B7_TN_PC_scores.csv")
    SUMMARY["B7_PC_variance_explained"] = list(np.round(Z0.var(0, ddof=1)[:5] / Z0.var(0, ddof=1).sum(), 3))
    # PC1 loadings
    pca_full = PCA(n_components=9, random_state=42).fit(StandardScaler().fit_transform(raw_tn))
    load = pd.Series(pca_full.components_[0], index=fpkm.index)
    pd.DataFrame({"gene": load.abs().sort_values(ascending=False).index[:200],
                  "PC1_loading": load[load.abs().sort_values(ascending=False).index[:200]].values}).to_csv(
        OUT / "B7_PC1_top200_loadings.csv", index=False)

    # --- B8 moderated DE ---------------------------------------------------
    Lx = L.loc[expressed]
    diag_vec = np.array([0 if c.startswith("Control") else 1 for c in Lx.columns])
    sex_vec = np.array([1 if GEO_SEX[c] == "M" else 0 for c in Lx.columns])
    de_cc = limma_trend(Lx, np.column_stack([np.ones(20), diag_vec, sex_vec]), coef=1)
    de_cc.to_csv(OUT / "B8_TN_vs_control_limma_trend_sex_adjusted.csv", index=False)
    de_cc_u = limma_trend(Lx, np.column_stack([np.ones(20), diag_vec]), coef=1)
    de_cc_u.to_csv(OUT / "B8_TN_vs_control_limma_trend_unadjusted.csv", index=False)
    Lt = Lx[tn]
    sex_t = np.array([1 if GEO_SEX[c] == "M" else 0 for c in tn])
    de_st = limma_trend(Lt, np.column_stack([np.ones(10), y, sex_t]), coef=1)
    de_st.to_csv(OUT / "B8_subtype1_vs_0_limma_trend_sex_adjusted.csv", index=False)
    for nm, df in [("TN_vs_control_sexadj", de_cc), ("TN_vs_control_unadj", de_cc_u), ("subtype1_vs_0_sexadj", de_st)]:
        SUMMARY[f"B8_{nm}_FDR05"] = int((df.p_adj < 0.05).sum())
        SUMMARY[f"B8_{nm}_FDR05_absLFC1"] = int(((df.p_adj < 0.05) & (df.log2FC.abs() > 1)).sum())
        SUMMARY[f"B8_{nm}_FDR10"] = int((df.p_adj < 0.10).sum())
        SUMMARY[f"B8_{nm}_prior_df_d0"] = float(df.attrs.get("d0", np.nan))
    SUMMARY["B8_TN_vs_control_top10"] = de_cc.head(10)[["gene", "log2FC", "p_adj"]].round(4).to_dict(orient="records")
    SUMMARY["B8_subtype_top10"] = de_st.head(10)[["gene", "log2FC", "p_adj"]].round(4).to_dict(orient="records")
    if "CSF2" in de_cc.gene.values:
        SUMMARY["B8_CSF2_TN_vs_control"] = de_cc[de_cc.gene == "CSF2"][["log2FC", "p_value", "p_adj"]].round(4).to_dict(orient="records")
    legacy_de = pd.read_csv(ROOT / "results" / "subtype_characterization" / "de_subtype_1.csv")
    leg_sig = legacy_de[(legacy_de.p_adj < 0.05) & (legacy_de.abs_log2fc > 1)]
    SUMMARY["B8_legacy_247_n_absLFC_gt10"] = int((leg_sig.abs_log2fc > 10).sum())
    SUMMARY["B8_legacy_247_n_mean_FPKM_lt1_both"] = int(((leg_sig.mean_subtype < 1) & (leg_sig.mean_other < 1)).sum())
    ov = de_st.set_index("gene").reindex(leg_sig.gene)
    SUMMARY["B8_legacy_247_retained_FDR05_moderated"] = int((ov.p_adj < 0.05).sum())
    SUMMARY["B8_legacy_247_in_expressed_set"] = int(ov.p_adj.notna().sum())
    hl = ["AMZ2", "LEMD2", "REPIN1", "NUDT17", "DMPK"]
    SUMMARY["B8_highlighted_genes_legacy_log2fc"] = legacy_de.set_index("gene").loc[hl, "log2fc"].round(3).to_dict()
    log(f"  B8 TN vs control (sex-adj): FDR<0.05 n={SUMMARY['B8_TN_vs_control_sexadj_FDR05']}; "
        f"subtype 1 vs 0 (sex-adj): FDR<0.05 n={SUMMARY['B8_subtype1_vs_0_sexadj_FDR05']}; "
        f"legacy 247 retained={SUMMARY['B8_legacy_247_retained_FDR05_moderated']}")

    # --- B9 preranked GSEA ------------------------------------------------
    try:
        import gseapy as gp

        libs = {"Reactome_2022": reactome, "GO_Biological_Process_2023": gobp}
        for nm, df in [("TN_vs_control", de_cc), ("subtype1_vs_0", de_st)]:
            rnk = df[["gene", "t_moderated"]].dropna().drop_duplicates("gene").set_index("gene")["t_moderated"].sort_values(ascending=False)
            outs = []
            for lname, lib in libs.items():
                pre = gp.prerank(rnk=rnk, gene_sets={k: list(v) for k, v in lib.items()}, min_size=15, max_size=500,
                                 permutation_num=1000, seed=42, threads=4, outdir=None, verbose=False, no_plot=True)
                r2 = pre.res2d.copy()
                r2.insert(0, "library", lname)
                outs.append(r2)
            g = pd.concat(outs, ignore_index=True)
            g.to_csv(OUT / f"B9_prerank_GSEA_{nm}_moderated_t.csv", index=False)
            fdr = pd.to_numeric(g["FDR q-val"], errors="coerce")
            SUMMARY[f"B9_{nm}_n_terms_FDR05"] = int((fdr < 0.05).sum())
            SUMMARY[f"B9_{nm}_n_terms_FDR25"] = int((fdr < 0.25).sum())
            SUMMARY[f"B9_{nm}_top_terms"] = g.assign(fdr=fdr).sort_values("fdr").head(8)[["library", "Term", "NES", "FDR q-val"]].to_dict(orient="records")
        log("  B9 preranked GSEA done")
    except Exception as e:
        log(f"  B9 GSEA failed: {e}")


def main() -> None:
    log("Loading references ...")
    reactome = load_gmt_enrichr("Reactome_2022")
    gobp = load_gmt_enrichr("GO_Biological_Process_2023")
    orth = load_mgi_orthologs()
    SUMMARY["mgi_one_to_one_orthologs"] = int(len(orth))
    revised = part_a_ion_cci(orth, reactome)
    part_b(revised, reactome, gobp)
    np.savez_compressed(OUT / "null_draws.npz", **NULL_DRAWS)
    with open(OUT / "REVISION_SUMMARY.json", "w") as fh:
        json.dump(SUMMARY, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("\nSUMMARY written to results/jhp_revision/REVISION_SUMMARY.json")


if __name__ == "__main__":
    main()
