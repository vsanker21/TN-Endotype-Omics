"""
JHP revision B4c: expression-matched random gene-set null for the submitted module gene sets, scored exactly as
in the submitted analysis (row z-scores of raw FPKM over all 20 samples, all module genes present in the matrix,
including non-expressed genes). Random sets are matched gene-for-gene on mean-FPKM decile over all genes with
non-zero variance, so the null reproduces the submitted sets' mixture of expressed and near-background genes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jhp_revision_analyses import (  # noqa: E402
    EXPR_FILE, OUT, ROOT, SUBTYPES_FILE, cohens_d, module_scores, zscore_rows,
)

N_DRAWS = 10000
LEGACY_DIR = {"peripheral_ecm_schwann": "up", "central_synaptic": "down", "ntrk_signaling": "down"}


def main() -> None:
    rng = np.random.default_rng(20260927)
    fpkm = pd.read_csv(EXPR_FILE, index_col=0)
    fpkm.index = fpkm.index.astype(str).str.upper()
    fpkm = fpkm[~fpkm.index.duplicated()]
    fpkm = fpkm[fpkm.var(axis=1) > 0]
    st = pd.read_csv(SUBTYPES_FILE).set_index("sample_id")
    tn = st.index.tolist()
    y = st.loc[tn, "subtype"].values
    z = zscore_rows(fpkm)
    ztn = z[tn].values
    mean_f = fpkm.mean(axis=1)
    bins = pd.qcut(mean_f.rank(method="first"), 10, labels=False)
    by_bin = {b: np.flatnonzero(bins.values == b) for b in range(10)}
    legacy = pd.read_csv(ROOT / "results" / "high_impact_integration" / "ion_cci_human_ortholog_mapping.csv")
    rows = []
    for mname, d in LEGACY_DIR.items():
        genes = [g for g in legacy[legacy.module == mname]["human_symbol"].str.upper().unique() if g in fpkm.index]
        s = module_scores(z, genes, d)[tn]
        d_obs = cohens_d(s[y == 0].values, s[y == 1].values)
        need = bins.reindex(genes).value_counts()
        null = np.empty(N_DRAWS)
        for i in range(N_DRAWS):
            ix = np.concatenate([rng.choice(by_bin[b], int(c), replace=False) for b, c in need.items()])
            sc = ztn[ix].mean(axis=0)
            null[i] = abs(cohens_d(sc[y == 0], sc[y == 1]))
        rows.append({"module": f"submitted_exact_{mname}", "n_genes_as_submitted": len(genes),
                     "n_mean_FPKM_ge_1": int((mean_f.reindex(genes) >= 1).sum()),
                     "submitted_abs_cohens_d": abs(d_obs),
                     "random_matched_median_abs_d": float(np.median(null)),
                     "random_matched_95pct_abs_d": float(np.percentile(null, 95)),
                     "random_matched_frac_ge_observed": float(np.mean(null >= abs(d_obs))),
                     "random_geneset_p": (1 + np.sum(null >= abs(d_obs))) / (N_DRAWS + 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B4c_submitted_gene_sets_matched_random_null.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
