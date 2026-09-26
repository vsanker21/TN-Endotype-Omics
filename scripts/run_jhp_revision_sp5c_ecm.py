"""
JHP revision E: the Sp5C-down extracellular-matrix/Schwann programme scored in blood.

With corrected IoN-CCI labels the dominant Sp5C change is a decrease in extracellular-matrix genes, which none of
the pre-specified modules captures (the Sp5C-down top-100 signature contains two ECM genes). This script defines the
module with the same rule as the other pathway modules (Sp5C genes down at padj < 0.05 that belong to the ECM/Schwann
Reactome terms, one-to-one human orthologs) and runs it through the same tests: blood expression, partition effect
with exact permutation, full-pipeline null (same generative model as B4, independent draws), expression-matched random
gene sets, controls, TN-versus-control mean and dispersion (with and without globin adjustment), and globin depletion.
It is kept separate from the 11 pre-specified gene sets so that their multiple-testing families are unchanged.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jhp_revision_analyses import (  # noqa: E402
    EXPR_FILE, OUT, REVISED_MODULES, SUBTYPES_FILE, cohens_d, load_gmt_enrichr, load_mgi_orthologs, module_scores,
    mwu_exact, zscore_rows,
)
from run_jhp_revision_globin_check import GLOBIN  # noqa: E402

E_OUT = OUT / "E_sp5c_ecm"
E_OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261001


def main() -> None:
    rng = np.random.default_rng(SEED)
    reactome = load_gmt_enrichr("Reactome_2022")
    orth = load_mgi_orthologs()
    o = orth.drop_duplicates("mouse").set_index("mouse")["human"]

    r = pd.read_csv(OUT / "A2_ion_cci_deseq2_sp5c_injury_vs_sham.csv").dropna(subset=["padj"])
    r = r.assign(human=o.reindex(r["gene_symbol"].astype(str)).values)
    sel = r[(r.padj < 0.05) & (r.log2FoldChange < 0)]
    terms = REVISED_MODULES["peripheral_ecm_schwann"]["terms"]
    matched = [t for t in reactome if any(x.lower() in t.lower() for x in terms)]
    term_genes = set().union(*(reactome[t] for t in matched))
    genes = sorted(set(sel["human"].dropna()) & term_genes)

    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]
    st = pd.read_csv(SUBTYPES_FILE).set_index("sample_id")
    tn = st.index.tolist()
    y = st.loc[tn, "subtype"].values
    L = np.log2(fpkm + 1)
    mean_f = fpkm.mean(axis=1)
    in_mat = [g for g in genes if g in fpkm.index]
    g_expr = [g for g in in_mat if mean_f[g] >= 1]
    S = {"module": "revised_ecm_schwann_Sp5Cdown_pathway", "reactome_terms": matched,
         "n_human_orthologs": len(genes), "n_in_blood_matrix": len(in_mat), "n_mean_FPKM_ge_1": len(g_expr),
         "median_FPKM_module_genes": float(mean_f.reindex(in_mat).median()),
         "median_FPKM_all_genes": float(mean_f.median()), "genes_mean_FPKM_ge_1": "; ".join(g_expr)}
    pd.DataFrame({"human_gene": genes, "in_blood_matrix": [g in fpkm.index for g in genes],
                  "mean_FPKM_blood": [float(mean_f.get(g, np.nan)) for g in genes]}).to_csv(
        E_OUT / "E1_sp5c_ecm_module_genes.csv", index=False)

    # partition effect: submitted-space (z FPKM) and log2 scale, exact permutation
    z_tn_raw = zscore_rows(fpkm[tn])
    z_tn = zscore_rows(L[tn])
    s_raw = module_scores(z_tn_raw, g_expr, "down")
    s_log = module_scores(z_tn, g_expr, "down")
    d_raw = cohens_d(s_raw[y == 0], s_raw[y == 1])
    d_log = cohens_d(s_log[y == 0], s_log[y == 1])
    obs = abs(s_log[y == 1].mean() - s_log[y == 0].mean())
    perm = []
    for idx1 in itertools.combinations(range(10), 4):
        lab = np.zeros(10, int)
        lab[list(idx1)] = 1
        perm.append(abs(s_log[lab == 1].mean() - s_log[lab == 0].mean()))
    S.update(cohens_d_injury_oriented_zFPKM=d_raw, cohens_d_injury_oriented_log2=d_log,
             exact_perm_p_210_labelings=float(np.mean(np.array(perm) >= obs - 1e-12)))

    # full-pipeline null (as B4: rank-r Gaussian with observed gene covariance -> standardise -> PCA -> 2-means)
    Xs = StandardScaler().fit_transform(fpkm[tn].T.values)
    _, sv, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    rk = int(np.sum(sv > 1e-8 * sv[0]))
    sv, Vt = sv[:rk], Vt[:rk]
    ix = np.array([fpkm.index.get_loc(g) for g in g_expr])
    nd = []
    for _ in range(1000):
        Xn = StandardScaler().fit_transform(rng.normal(size=(10, rk)) @ np.diag(sv / np.sqrt(10)) @ Vt)
        Zp = PCA(n_components=min(9, rk), random_state=42).fit_transform(Xn)
        lab = KMeans(2, random_state=42, n_init=20).fit_predict(Zp)
        sc = Xn[:, ix].mean(axis=1)
        nd.append(abs(cohens_d(sc[lab == 0], sc[lab == 1])))
    nd = np.array(nd)
    nd = nd[~np.isnan(nd)]
    S.update(full_pipeline_null_n_valid_splits=int(len(nd)), full_pipeline_null_median_abs_d=float(np.median(nd)),
             full_pipeline_null_p=float((1 + np.sum(nd >= abs(d_raw))) / (len(nd) + 1)))

    # expression-matched random gene sets (log2 scale, observed labels)
    amean = L[tn].mean(axis=1)
    pool = amean[amean.index.isin(mean_f.index[mean_f >= 1])]
    bins = pd.qcut(pool, 10, labels=False, duplicates="drop")
    need = bins.reindex(g_expr).value_counts()
    by_bin = {b: pool.index[bins == b].values for b in need.index}
    rs = []
    for _ in range(10000):
        gs = np.concatenate([rng.choice(by_bin[b], int(c), replace=False) for b, c in need.items()])
        sc = z_tn.loc[gs].mean(axis=0).values
        rs.append(abs(cohens_d(sc[y == 0], sc[y == 1])))
    rs = np.array(rs)
    S.update(random_geneset_median_abs_d=float(np.median(rs)),
             random_geneset_p=float((1 + np.sum(rs >= abs(d_log))) / 10001))

    # controls, TN versus control mean and dispersion
    z20 = zscore_rows(L)
    s20 = module_scores(z20, g_expr, "down")
    is_tn = s20.index.isin(tn)
    part = pd.Series(np.nan, index=s20.index)
    part[tn] = y
    c, p0, p1 = s20[~is_tn], s20[part == 0], s20[part == 1]
    gf = fpkm.reindex(["HBB", "HBA1", "HBA2", "HBD"]).sum() / fpkm.sum()
    g = gf.reindex(s20.index).values
    v = s20.values
    X = np.column_stack([np.ones_like(g), g])
    res = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    def perm_vr(x):
        obs_vr = np.var(x[is_tn], ddof=1) / np.var(x[~is_tn], ddof=1)
        pv = np.empty(20000)
        for b in range(pv.size):
            p = rng.permutation(is_tn)
            pv[b] = np.var(x[p], ddof=1) / np.var(x[~p], ddof=1)
        return obs_vr, float((1 + np.sum(pv >= obs_vr)) / (pv.size + 1))

    vr, vr_p = perm_vr(v)
    vra, vra_p = perm_vr(res)
    S.update(control_mean=float(c.mean()), partition0_mean=float(p0.mean()), partition1_mean=float(p1.mean()),
             control_range=[float(c.min()), float(c.max())],
             TN_vs_control_MWU_p=mwu_exact(c, s20[is_tn]),
             spearman_rho_globin_all20=float(stats.spearmanr(v, g)[0]),
             variance_ratio_TN_over_control=float(vr), permutation_p_variance_ratio=vr_p,
             brown_forsythe_p=float(stats.levene(v[is_tn], v[~is_tn], center="median").pvalue),
             variance_ratio_globin_adjusted=float(vra), permutation_p_globin_adjusted=vra_p,
             brown_forsythe_p_globin_adjusted=float(stats.levene(res[is_tn], res[~is_tn], center="median").pvalue))

    # globin depletion (as B10)
    dep = fpkm.loc[[x for x in fpkm.index if x not in GLOBIN]]
    dep = dep / dep.sum() * 1e6
    s_dep = module_scores(zscore_rows(np.log2(dep[tn] + 1)), g_expr, "down")
    S["d_globin_depleted"] = cohens_d(s_dep[y == 0], s_dep[y == 1])
    S["module_raw_expression_vs_globin_spearman_TN"] = float(
        stats.spearmanr(z_tn.loc[g_expr].mean(), gf[tn].values).correlation)

    pd.DataFrame({"sample": s20.index, "group": ["Control" if not t else f"Partition {int(part[s])}"
                                                 for s, t in zip(s20.index, is_tn)],
                  "score_injury_oriented": v, "globin_fraction": g, "score_globin_residual": res}).to_csv(
        E_OUT / "E2_sp5c_ecm_scores_all20.csv", index=False)
    pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in S.items()}]).T.rename(
        columns={0: "value"}).to_csv(E_OUT / "E3_sp5c_ecm_tests.csv")
    (E_OUT / "E_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps({k: v for k, v in S.items() if k != "genes_mean_FPKM_ge_1"}, indent=2, default=float))


if __name__ == "__main__":
    main()
