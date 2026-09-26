"""
JHP revision B10: is the k=2 TN partition driven by globin/erythroid library composition?

1. Rank of globin fraction vs subtype, in TN and projected controls.
2. Globin-depleted renormalisation: drop haemoglobin/erythroid-dominant genes, rescale each sample
   to a common total (TPM-like), log2(x+1), then rerun the submitted pipeline and module tests.
3. Covariate regression: regress log2 expression on globin fraction, recluster residuals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jhp_revision_analyses import (  # noqa: E402
    EXPR_FILE, OUT, SUBTYPES_FILE, cohens_d, module_scores, pipeline_labels, zscore_rows,
)

GLOBIN = ["HBB", "HBA1", "HBA2", "HBD", "HBM", "HBQ1", "HBG1", "HBG2", "HBZ", "HBE1", "ALAS2", "SLC4A1", "CA1", "AHSP"]


def main() -> None:
    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]
    st = pd.read_csv(SUBTYPES_FILE).set_index("sample_id")
    tn = st.index.tolist()
    y = st.loc[tn, "subtype"].values
    ctrl = [c for c in fpkm.columns if c.startswith("Control")]
    out = {}

    gfrac = fpkm.reindex(["HBB", "HBA1", "HBA2", "HBD"]).sum() / fpkm.sum()
    tab = pd.DataFrame({"globin_fraction": gfrac, "group": ["Control" if c in ctrl else f"Subtype {st.loc[c,'subtype']}" for c in fpkm.columns]})
    proj = pd.read_csv(OUT / "B5_controls_projected_onto_TN_partition.csv").set_index("sample")
    tab["assigned_or_subtype"] = [f"Subtype {st.loc[c,'subtype']}" if c in tn else f"Subtype {proj.loc[c,'assigned_subtype']} (projected)" for c in tab.index]
    tab = tab.sort_values("globin_fraction", ascending=False)
    tab.to_csv(OUT / "B10_globin_fraction_by_sample.csv")
    t = tab.loc[tn].sort_values("globin_fraction", ascending=False)
    out["TN_top4_globin_are_subtype1"] = bool((t.head(4)["group"] == "Subtype 1").all())
    pc_c = proj.join(tab["globin_fraction"])
    out["controls_globin_projected_s1_vs_s0_MWU_p"] = float(stats.mannwhitneyu(
        pc_c[pc_c.assigned_subtype == 1].globin_fraction, pc_c[pc_c.assigned_subtype == 0].globin_fraction,
        method="exact").pvalue)
    out["controls_globin_vs_PC1_spearman"] = float(stats.spearmanr(pc_c.globin_fraction, pc_c.PC1).correlation)

    # globin-depleted renormalisation
    keep = [g for g in fpkm.index if g not in GLOBIN]
    dep = fpkm.loc[keep]
    dep = dep / dep.sum() * 1e6
    Ld = np.log2(dep + 1)
    expressed = Ld.index[(dep >= 1).sum(axis=1) >= 5]
    variants = {
        "globin-depleted TPM, z-scored, all genes (submitted pipeline otherwise)": dep[tn].T.values,
        "globin-depleted log2 TPM+1, expressed genes": Ld.loc[expressed, tn].T.values,
    }
    Lraw = np.log2(fpkm + 1)
    X = Lraw.loc[expressed.intersection(Lraw.index), tn]
    g = gfrac[tn].values
    D = np.column_stack([np.ones(10), g])
    beta = np.linalg.lstsq(D, X.T.values, rcond=None)[0]
    resid = X.T.values - D @ beta
    variants["log2 FPKM+1 residualised on globin fraction, expressed genes"] = resid
    rows = []
    for name, M in variants.items():
        lab, Z = pipeline_labels(M)
        rows.append({"variant": name, "ARI_vs_submitted": adjusted_rand_score(y, lab),
                     "sizes": "/".join(map(str, sorted(np.bincount(lab), reverse=True))),
                     "silhouette": silhouette_score(Z, lab),
                     "PC1_var_explained": float(Z.var(0)[0] / Z.var(0).sum()),
                     "PC1_vs_globin_spearman": float(stats.spearmanr(Z[:, 0], g).correlation)})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(OUT / "B10_partition_after_globin_adjustment.csv", index=False)

    # module effects under globin-depleted normalisation (submitted labels)
    genes_tab = pd.read_csv(OUT / "A4_revised_module_gene_sets.csv")
    legacy = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")
    sets = {f"legacy_{m}": (legacy[legacy.module == m].human_symbol.str.upper().unique().tolist(), d)
            for m, d in [("peripheral_ecm_schwann", "up"), ("central_synaptic", "down"), ("ntrk_signaling", "down")]}
    for (m, dfn), grp in genes_tab.groupby(["module", "definition"]):
        sets[f"revised_{m}_{dfn.replace('_genes','')}"] = (grp.human_gene.tolist(), grp.direction.iloc[0])
    mean_raw = fpkm.mean(axis=1)
    z_raw = zscore_rows(np.log2(fpkm[tn] + 1))
    z_dep = zscore_rows(Ld[tn])
    mrows = []
    for name, (genes, d) in sets.items():
        ge = [x for x in genes if x in mean_raw.index and mean_raw[x] >= 1 and x in Ld.index]
        if len(ge) < 3:
            continue
        s0 = module_scores(z_raw, ge, d)
        s1 = module_scores(z_dep, ge, d)
        raw_expr = z_raw.loc[ge].mean()
        mrows.append({"module": name, "n_expressed": len(ge),
                      "d_log2FPKM": cohens_d(s0[y == 0], s0[y == 1]),
                      "d_globin_depleted": cohens_d(s1[y == 0], s1[y == 1]),
                      "module_raw_expression_vs_globin_spearman": float(stats.spearmanr(raw_expr, g).correlation)})
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(OUT / "B10_module_effects_after_globin_depletion.csv", index=False)

    # how many subtype DE genes are explained by globin fraction
    pd.Series(out).to_json(OUT / "B10_summary.json", indent=2)
    pd.set_option("display.width", 220)
    print(pd.Series(out).to_string())
    print(tab.round(3).to_string())
    print(vdf.round(3).to_string())
    print(mdf.round(3).to_string())


if __name__ == "__main__":
    main()
