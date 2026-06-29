"""
Nature Communications extension analyses (items 2–5 + figure package).

1. Fix GSE124272/GSE150408 probe→gene mapping (GPL21185)
2. Meta-analytic module scoring across neuropathic pain cohorts
3. Imaging endotype ↔ nerve-region ROI concordance (OpenNeuro ds005713)
4. Drug repurposing + Open Targets connectivity + TG ligand–receptor
5. Stanford validation SAP + Nat Comm figure panels

Scientific framing: convergent validity of peripheral axis in NP blood;
central axis reported as context-specific; no TN subtype replication claims.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import urllib.request
import warnings
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.extract_nerve_region_imaging import extract_cohort_nerve_metrics
from scripts.run_endotype_validation_tiers import (
    FIG as VAL_FIG,
    MODULES,
    OUT as VAL_OUT,
    load_frozen_human_module_genes,
    load_gse177034_counts,
    parse_geo_series_matrix_metadata,
    score_modules,
)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

OUT = project_root / "results" / "natcomm_package"
FIG = OUT / "figures"
GEO = project_root / "data" / "external" / "geo"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Curated ligand–receptor pairs relevant to IoN-CCI modules (CellChatDB / literature)
LR_PAIRS = [
    ("NGF", "NTRK1", "ntrk_signaling", "Neurons"),
    ("BDNF", "NTRK2", "ntrk_signaling", "Neurons"),
    ("NTF3", "NTRK3", "ntrk_signaling", "Neurons"),
    ("NRG1", "ERBB2", "peripheral_ecm_schwann", "Schwann_cells"),
    ("NRG1", "ERBB3", "peripheral_ecm_schwann", "Schwann_cells"),
    ("TGFB1", "TGFBR1", "peripheral_ecm_schwann", "Schwann_cells"),
    ("TGFB1", "TGFBR2", "peripheral_ecm_schwann", "Schwann_cells"),
    ("GDNF", "RET", "central_synaptic", "Neurons"),
    ("GDNF", "GFRA1", "central_synaptic", "Neurons"),
    ("SLIT2", "ROBO1", "peripheral_ecm_schwann", "Schwann_cells"),
    ("SEMA3A", "NRP1", "peripheral_ecm_schwann", "Neurons"),
    ("GABA", "GABRA1", "central_synaptic", "Neurons"),
    ("GABA", "GABBR1", "central_synaptic", "Neurons"),
    ("GLUTAMATE", "GRIN1", "central_synaptic", "Neurons"),
    ("GLUTAMATE", "GRIN2B", "central_synaptic", "Neurons"),
    ("CNTF", "CNTFR", "peripheral_ecm_schwann", "Schwann_cells"),
    ("LIF", "LIFR", "peripheral_ecm_schwann", "Satellite_glial_cells"),
    ("IL6", "IL6R", "peripheral_ecm_schwann", "Immune_cells"),
    ("CCL2", "CCR2", "peripheral_ecm_schwann", "Immune_cells"),
    ("COL1A1", "ITGA1", "peripheral_ecm_schwann", "Fibroblasts"),
]


def parse_geo_phenotypes(path: Path) -> pd.DataFrame:
    """Parse GEO series matrix metadata including multi-line characteristics."""
    opener = gzip.open if str(path).endswith(".gz") else open
    sample_keys: list[str] = []
    fields: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("!Sample_"):
                continue
            parts = line.strip().split("\t")
            field = parts[0].replace("!Sample_", "")
            vals = [p.strip('"') for p in parts[1:]]
            if field == "geo_accession":
                sample_keys = vals
            elif field == "title" and sample_keys:
                for sid, val in zip(sample_keys, vals):
                    titles[sid] = val
            elif field.startswith("characteristics_ch1") and sample_keys:
                for sid, val in zip(sample_keys, vals):
                    if ":" in val:
                        key, v = val.split(":", 1)
                        fields.setdefault(sid, {})[key.strip()] = v.strip()
            else:
                for sid, val in zip(sample_keys, vals):
                    fields.setdefault(sid, {})[field] = val

    rows = []
    for sid in sample_keys:
        rec = {"geo_accession": sid, "sample_id": sid, **fields.get(sid, {})}
        if sid in titles:
            rec["title"] = titles[sid]
        rows.append(rec)
    return pd.DataFrame(rows)


def assign_np_group(meta: pd.DataFrame) -> pd.Series:
    """Assign NP_pain vs control from merged GEO phenotype fields."""
    diagnosis = meta.get("diagnosis", pd.Series("", index=meta.index)).astype(str)
    title = meta.get("title", pd.Series("", index=meta.index)).astype(str)
    combined = (diagnosis + " " + title).str.lower()
    is_control = combined.str.contains(
        r"healthy|control|volunteer|normal", case=False, regex=True
    ) & ~combined.str.contains(r"patient", case=False)
    is_case = combined.str.contains(
        r"patient|sciatica|neuropathic|idd|degeneration|pain", case=False, regex=True
    ) & ~is_control
    group = pd.Series("unknown", index=meta.index)
    group[is_control] = "control"
    group[is_case] = "NP_pain"
    return group


def download_geo_series(accession: str) -> Path:
    """Download GEO series matrix if missing."""
    out_dir = GEO / accession
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / f"{accession}_series_matrix.txt.gz"
    txt_path = out_dir / f"{accession}_series_matrix.txt"
    if txt_path.exists() or gz_path.exists():
        return gz_path if gz_path.exists() else txt_path
    series_num = accession.replace("GSE", "")
    prefix = series_num[:-3] + "nnn"
    url = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE{prefix}/{accession}/matrix/{accession}_series_matrix.txt.gz"
    logger.info(f"Downloading {accession} from GEO...")
    urllib.request.urlretrieve(url, gz_path)
    return gz_path


def download_gpl21185_table() -> Path:
    """Stream-parse GPL21185 probe table from family.soft.gz."""
    cache = GEO / "platforms" / "GPL21185_probe_gene_map.csv"
    if cache.exists() and cache.stat().st_size > 1000:
        return cache

    soft_gz = GEO / "platforms" / "GPL21185_family.soft.gz"
    if not soft_gz.exists():
        soft_gz.parent.mkdir(parents=True, exist_ok=True)
        url = "ftp://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL21nnn/GPL21185/soft/GPL21185_family.soft.gz"
        logger.info("Downloading GPL21185_family.soft.gz (~2.4 GB)...")
        urllib.request.urlretrieve(url, soft_gz)

    rows = []
    in_table = False
    header = None
    with gzip.open(soft_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                if header is None:
                    header = line.strip().split("\t")
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts)))

    df = pd.DataFrame(rows)
    df = df.rename(columns={"ID": "probe_id", "GENE_SYMBOL": "gene_symbol"})
    df = df[["probe_id", "gene_symbol"]].copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip().str.upper()
    df = df[(df["gene_symbol"] != "") & (df["gene_symbol"] != "NAN") & (~df["gene_symbol"].str.startswith("---"))]
    df.to_csv(cache, index=False)
    logger.info(f"GPL21185 probe map: {len(df)} probes → {df['gene_symbol'].nunique()} genes")
    return cache


def load_agilent_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Agilent series matrix; return gene-level log2 expression + phenotype."""
    probe_map = pd.read_csv(download_gpl21185_table())
    probe_to_gene = dict(zip(probe_map["probe_id"], probe_map["gene_symbol"]))

    meta = parse_geo_phenotypes(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    end = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    table = pd.read_csv(StringIO("".join(lines[start + 1 : end])), sep="\t")
    table = table.set_index(table.columns[0])
    table.index = table.index.astype(str)
    table = table.apply(pd.to_numeric, errors="coerce")

    # Map probes → genes (max probe per gene)
    gene_rows = {}
    for probe, row in table.iterrows():
        gene = probe_to_gene.get(probe)
        if not gene:
            continue
        if gene not in gene_rows:
            gene_rows[gene] = row.copy()
        else:
            gene_rows[gene] = np.maximum(gene_rows[gene], row)

    expr = pd.DataFrame(gene_rows).T
    expr.index = expr.index.astype(str).str.upper()
    expr = np.log2(expr + 1)

    # Align columns to sample titles
    if "title" in meta.columns:
        col_map = dict(zip(meta["geo_accession"], meta["title"]))
        expr = expr.rename(columns={c: col_map.get(c, c) for c in expr.columns})

    return expr, meta


def cohort_test(
    scores: pd.DataFrame,
    group_col: str,
    case_label: str,
    control_label: str,
    cohort: str,
    contrast: str,
    timepoint: str = "",
) -> list[dict]:
    """Mann–Whitney module tests for one cohort contrast."""
    merged = scores.copy()
    rows = []
    for mod in MODULES + ["central_axis"]:
        case = merged[merged[group_col] == case_label][mod].dropna()
        ctrl = merged[merged[group_col] == control_label][mod].dropna()
        if len(case) < 3 or len(ctrl) < 3:
            continue
        u, p = stats.mannwhitneyu(case, ctrl, alternative="two-sided")
        pooled_sd = np.sqrt((case.var() + ctrl.var()) / 2) if (case.var() + ctrl.var()) > 0 else np.nan
        d = (case.mean() - ctrl.mean()) / pooled_sd if pooled_sd and pooled_sd > 0 else 0.0
        rows.append({
            "cohort": cohort,
            "contrast": contrast,
            "timepoint": timepoint,
            "module": mod,
            "case_label": case_label,
            "control_label": control_label,
            "mean_case": case.mean(),
            "mean_control": ctrl.mean(),
            "delta_case_minus_control": case.mean() - ctrl.mean(),
            "cohens_d": d,
            "mannwhitney_p": p,
            "n_case": len(case),
            "n_control": len(ctrl),
        })
    return rows


def fisher_combine_pvalues(p_values: list[float]) -> float:
    """Fisher's method for combining p-values."""
    p_values = [p for p in p_values if 0 < p <= 1]
    if not p_values:
        return np.nan
    stat = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    return float(stats.chi2.sf(stat, df))


def stouffer_combine(p_values: list[float], weights: list[float] | None = None) -> float:
    """Weighted Stouffer Z combination."""
    p_values = [p for p in p_values if 0 < p <= 1]
    if not p_values:
        return np.nan
    if weights is None:
        weights = [1.0] * len(p_values)
    z = [stats.norm.ppf(1 - p / 2) * np.sign(0.5 - p) for p in p_values]
    w = np.array(weights[: len(z)])
    combined_z = np.sum(w * z) / np.sqrt(np.sum(w**2))
    return float(2 * stats.norm.sf(abs(combined_z)))


def run_np_meta_analysis(module_genes: dict) -> pd.DataFrame:
    """Meta-analytic module scoring across GSE177034 + GSE124272 + GSE150408."""
    logger.info("NP meta-analysis: frozen IoN-CCI modules across external cohorts")
    all_tests = []

    # GSE177034 — Persistent vs Resolved
    logcpm, pheno177 = load_gse177034_counts()
    col_map = {c: re.sub(r"\.bam$", "", c) for c in logcpm.columns}
    logcpm = logcpm.rename(columns=col_map)
    scores177 = score_modules(logcpm, module_genes)
    m177 = scores177.merge(pheno177, on="sample_id", how="inner")
    m177.to_csv(OUT / "gse177034_module_scores.csv", index=False)

    for tp_suffix, tp_label in [(".t0", "t0_baseline"), (".t1", "t1_followup")]:
        sub = m177[m177["sample_id"].str.endswith(tp_suffix)].copy()
        sub["group"] = sub["paingroup"]
        all_tests.extend(
            cohort_test(
                sub, "group", "Persistent", "Resolved",
                "GSE177034", "persistent_vs_resolved", tp_label,
            )
        )

    # GSE124272 — IDD NP vs healthy
    path124 = download_geo_series("GSE124272")
    expr124, meta124 = load_agilent_series_matrix(path124)
    pheno124 = meta124.copy()
    pheno124["group"] = assign_np_group(pheno124)
    pheno124 = pheno124[pheno124["group"].isin(["NP_pain", "control"])]
    scores124 = score_modules(expr124, module_genes)
    m124 = scores124.merge(pheno124[["sample_id", "group"]], on="sample_id", how="inner")
    m124.to_csv(OUT / "gse124272_module_scores.csv", index=False)
    all_tests.extend(
        cohort_test(m124, "group", "NP_pain", "control", "GSE124272", "np_vs_healthy", "")
    )

    # GSE150408 — sciatica NP vs healthy
    expr150 = None
    try:
        path150 = download_geo_series("GSE150408")
        expr150, meta150 = load_agilent_series_matrix(path150)
        pheno150 = meta150.copy()
        pheno150["group"] = assign_np_group(pheno150)
        pheno150 = pheno150[pheno150["group"].isin(["NP_pain", "control"])]
        scores150 = score_modules(expr150, module_genes)
        m150 = scores150.merge(pheno150[["sample_id", "group"]], on="sample_id", how="inner")
        m150.to_csv(OUT / "gse150408_module_scores.csv", index=False)
        all_tests.extend(
            cohort_test(m150, "group", "NP_pain", "control", "GSE150408", "np_vs_healthy", "")
        )
    except Exception as exc:
        logger.warning(f"GSE150408 skipped: {exc}")

    per_cohort = pd.DataFrame(all_tests)
    per_cohort.to_csv(OUT / "np_meta_per_cohort_tests.csv", index=False)

    # Meta-combine: primary peripheral at chronic/persistent endpoints
    meta_rows = []
    peripheral_primary = per_cohort[
        (per_cohort["module"] == "peripheral_ecm_schwann")
        & (
            (per_cohort["contrast"] == "persistent_vs_resolved")
            & (per_cohort["timepoint"] == "t1_followup")
            | (per_cohort["contrast"] == "np_vs_healthy")
        )
    ]
    for mod in MODULES + ["central_axis"]:
        sub = per_cohort[
            (per_cohort["module"] == mod)
            & (
                ((per_cohort["contrast"] == "persistent_vs_resolved") & (per_cohort["timepoint"] == "t1_followup"))
                | (per_cohort["contrast"] == "np_vs_healthy")
            )
        ]
        if len(sub) == 0:
            continue
        ps = sub["mannwhitney_p"].tolist()
        ns = (sub["n_case"] + sub["n_control"]).tolist()
        meta_rows.append({
            "module": mod,
            "n_cohorts": len(sub),
            "cohorts_included": ";".join(sub["cohort"].unique()),
            "mean_cohens_d": sub["cohens_d"].mean(),
            "cohort_direction_consistency": float(np.mean(np.sign(sub["cohens_d"]) == np.sign(sub["cohens_d"].iloc[0]))),
            "fisher_combined_p": fisher_combine_pvalues(ps),
            "stouffer_combined_p": stouffer_combine(ps, weights=[np.sqrt(n) for n in ns]),
            "min_cohort_p": sub["mannwhitney_p"].min(),
            "max_cohort_p": sub["mannwhitney_p"].max(),
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUT / "np_meta_combined_tests.csv", index=False)

    # Coverage report
    cov_rows = []
    for name, expr in [("GSE177034", logcpm), ("GSE124272", expr124), ("GSE150408", expr150)]:
        if expr is None:
            continue
        for mod, genes in module_genes.items():
            avail = [g for g in genes if g in expr.index]
            cov_rows.append({
                "cohort": name,
                "module": mod,
                "n_genes_total": len(genes),
                "n_genes_expressed": len(avail),
                "coverage": len(avail) / max(len(genes), 1),
            })
    pd.DataFrame(cov_rows).to_csv(OUT / "np_meta_module_coverage.csv", index=False)

    return meta_df


def run_imaging_nerve_concordance() -> pd.DataFrame:
    """DTI-first imaging endotype vs atlas nerve-adjacent ROI metrics (OpenNeuro)."""
    logger.info("Imaging ↔ DTI nerve-adjacent ROI concordance (ds005713)")
    mvd_path = project_root / "results" / "endotype_framework" / "mvd_cohort_with_endotypes.csv"
    dti_path = project_root / "results" / "dti_nerve_adjacent" / "dti_nerve_adjacent_mvd_merged.csv"

    if dti_path.exists():
        merged = pd.read_csv(dti_path)
        nerve_df = merged
    elif mvd_path.exists():
        mvd = pd.read_csv(mvd_path)
        openeuro = project_root / "data" / "external" / "openneuro" / "ds005713"
        subject_ids = mvd["subject_id"].dropna().astype(str).tolist()
        nerve_df = extract_cohort_nerve_metrics(
            subject_ids, openeuro, OUT / "nerve_region_imaging_refined.csv",
            compute_dti=True, roi_method="atlas", include_t1_t2=False,
        )
        merged = mvd.merge(nerve_df, on="subject_id", how="left", suffixes=("", "_nerve"))
    else:
        logger.warning("MVD endotype file missing; skipping imaging analysis")
        return pd.DataFrame()

    merged.to_csv(OUT / "imaging_endotype_nerve_roi_merged.csv", index=False)

    # DTI composite indices (primary)
    if "fa_rez_bilateral_mean" in merged.columns and "md_rez_bilateral_mean" in merged.columns:
        merged["dti_rez_fa_md_ratio"] = merged["fa_rez_bilateral_mean"] / (merged["md_rez_bilateral_mean"] + 1e-12)
    if "fa_cisternal_bilateral_mean" in merged.columns and "fa_pons_nuclei_proxy_mean" in merged.columns:
        merged["dti_cisternal_pons_fa_ratio"] = (
            merged["fa_cisternal_bilateral_mean"] / (merged["fa_pons_nuclei_proxy_mean"] + 1e-9)
        )

    # Primary: atlas DTI metrics; exclude legacy T1/T2 unless explicitly tagged
    roi_cols = [
        c for c in merged.columns
        if (c.startswith(("fa_", "md_", "rd_", "ad_")) and c.endswith("_mean") and "legacy" not in c)
        or c.endswith("_ratio")
    ]
    roi_cols += [c for c in merged.columns if c.startswith("dti_") and c.endswith("_ratio")]
    roi_cols = list(dict.fromkeys(roi_cols))

    test_rows = []
    for roi in roi_cols:
        if merged[roi].notna().sum() < 8:
            continue
        for label in merged["imaging_endotype_label"].dropna().unique():
            pass
        # Central vs intermediate/peripheral imaging endotypes
        central = merged[merged["imaging_endotype_label"] == "central"][roi].dropna()
        non_central = merged[merged["imaging_endotype_label"] != "central"][roi].dropna()
        if len(central) >= 3 and len(non_central) >= 3:
            u, p = stats.mannwhitneyu(central, non_central, alternative="two-sided")
            test_rows.append({
                "roi_metric": roi,
                "comparison": "central_vs_noncentral_imaging_endotype",
                "mean_central": central.mean(),
                "mean_non_central": non_central.mean(),
                "mannwhitney_p": p,
                "n_central": len(central),
                "n_non_central": len(non_central),
                "metric_tier": "primary_dti_atlas",
            })
        # ROI vs MVD outcome
        if "outcome_binary" in merged.columns:
            pos = merged[merged["outcome_binary"] == 1][roi].dropna()
            neg = merged[merged["outcome_binary"] == 0][roi].dropna()
            if len(pos) >= 2 and len(neg) >= 2:
                u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
                test_rows.append({
                    "roi_metric": roi,
                    "comparison": "mvd_pos_vs_neg",
                    "mean_pos": pos.mean(),
                    "mean_neg": neg.mean(),
                    "mannwhitney_p": p,
                    "n_pos": len(pos),
                    "n_neg": len(neg),
                    "metric_tier": "primary_dti_atlas",
                })

    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "imaging_nerve_roi_tests.csv", index=False)

    # MVD outcome by imaging endotype (Fisher)
    if "imaging_endotype_label" in merged.columns and "outcome_binary" in merged.columns:
        outcome_rows = []
        for label in merged["imaging_endotype_label"].dropna().unique():
            sub = merged[merged["imaging_endotype_label"] == label]
            n_pos = int((sub["outcome_binary"] == 1).sum())
            n_neg = int((sub["outcome_binary"] == 0).sum())
            outcome_rows.append({
                "imaging_endotype": label,
                "n": len(sub),
                "mvd_response_rate": n_pos / len(sub) if len(sub) else np.nan,
                "n_responder": n_pos,
                "n_non_responder": n_neg,
            })
        outcome_df = pd.DataFrame(outcome_rows)
        outcome_df.to_csv(OUT / "mvd_outcome_by_imaging_endotype.csv", index=False)

        ct = pd.crosstab(merged["imaging_endotype_label"], merged["outcome_binary"])
        if ct.shape == (2, 2) or ct.size >= 4:
            try:
                _, fisher_p = stats.fisher_exact(ct.values[:2, :2] if ct.shape[0] >= 2 and ct.shape[1] >= 2 else ct.values)
                pd.DataFrame([{"test": "fisher_imaging_endotype_x_mvd", "p_value": fisher_p}]).to_csv(
                    OUT / "mvd_imaging_endotype_fisher.csv", index=False
                )
            except Exception:
                pass

    return tests


def document_patient_linkage() -> None:
    """Explicit audit: no blood–imaging patient overlap available."""
    blood_samples = pd.read_csv(project_root / "results" / "molecular_subtypes.csv")["sample_id"].tolist()
    mvd = pd.read_csv(project_root / "results" / "endotype_framework" / "mvd_cohort_with_endotypes.csv")
    audit = {
        "gse186505_tn_samples": blood_samples,
        "openneuro_mvd_subjects": mvd["subject_id"].tolist(),
        "n_blood_tn": len(blood_samples),
        "n_openneuro_mvd": len(mvd),
        "matched_patients": [],
        "n_matched": 0,
        "conclusion": (
            "No programmatic patient linkage between GSE186505 blood RNA and OpenNeuro ds005713 imaging. "
            "Blood–imaging–outcome triangulation requires Stanford/collaborator matched cohort."
        ),
    }
    (OUT / "patient_linkage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def query_open_targets(genes: list[str], max_genes: int = 25) -> pd.DataFrame:
    """Query Open Targets Platform GraphQL for known drugs targeting module genes."""
    import urllib.error

    endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
    rows = []
    for gene in genes[:max_genes]:
        query = """
        query knownDrugs($symbol: String!) {
          target(ensemblIds: []) { id }
          search(queryString: $symbol, entityNames: ["target"]) {
            hits { id name entity
              object { ... on Target { id approvedSymbol knownDrugs { count rows { drug { id name maximumClinicalTrialPhase } mechanismOfAction disease { id name } } } } }
            }
          }
        }
        """
        # Simpler search-based query
        q2 = """
        query($q: String!) {
          search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
            hits {
              name
              object {
                ... on Target {
                  approvedSymbol
                  knownDrugs {
                    rows {
                      drug { name maximumClinicalTrialPhase }
                      mechanismOfAction
                    }
                  }
                }
              }
            }
          }
        }
        """
        payload = json.dumps({"query": q2, "variables": {"q": gene}}).encode()
        req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                continue
            obj = hits[0].get("object") or {}
            for row in (obj.get("knownDrugs") or {}).get("rows", [])[:5]:
                drug = row.get("drug") or {}
                rows.append({
                    "target_gene": gene,
                    "approved_symbol": obj.get("approvedSymbol", gene),
                    "drug_name": drug.get("name"),
                    "max_clinical_phase": drug.get("maximumClinicalTrialPhase"),
                    "mechanism": row.get("mechanismOfAction"),
                    "source": "OpenTargets_v4",
                })
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.debug(f"OpenTargets query failed for {gene}: {exc}")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "open_targets_module_drugs.csv", index=False)
    return df


def run_ligand_receptor_analysis(module_genes: dict) -> pd.DataFrame:
    """Score curated LR pairs using human TG reference signatures (GSE197289)."""
    logger.info("TG ligand–receptor analysis for IoN-CCI modules")
    ref = pd.read_csv(
        project_root / "results" / "aim1_enhanced" / "human_tg_integration" / "human_tg_reference_signatures.csv",
        index_col=0,
    )
    ref.index = ref.index.astype(str).str.upper()

    cell_map = {
        "Neurons": ["Neurons", "Unknown_3", "Unknown_14"],
        "Schwann_cells": ["Schwann_cells", "Unknown_8"],
        "Satellite_glial_cells": ["Unknown_11"],
        "Immune_cells": ["Unknown_7"],
        "Fibroblasts": ["Fibroblasts", "Unknown_12", "Unknown_2"],
    }
    canon = {}
    for ct, cols in cell_map.items():
        present = [c for c in cols if c in ref.columns]
        if present:
            canon[ct] = ref[present].max(axis=1)

    rows = []
    for ligand, receptor, module, receiver_ct in LR_PAIRS:
        l_expr = ref.loc[ligand].max() if ligand in ref.index else np.nan
        r_expr = canon.get(receiver_ct, pd.Series(dtype=float))
        r_val = float(r_expr.get(receptor, np.nan)) if receptor in r_expr.index else np.nan
        lig_in_module = ligand in module_genes.get(module, []) or receptor in module_genes.get(module, [])
        score = np.sqrt((l_expr if pd.notna(l_expr) else 0) * (r_val if pd.notna(r_val) else 0))
        rows.append({
            "ligand": ligand,
            "receptor": receptor,
            "module": module,
            "receiver_celltype": receiver_ct,
            "ligand_expr_tg_max": l_expr,
            "receptor_expr_celltype": r_val,
            "lr_score_sqrt_product": score,
            "module_gene_overlap": lig_in_module,
        })

    lr_df = pd.DataFrame(rows).sort_values("lr_score_sqrt_product", ascending=False)
    lr_df.to_csv(OUT / "tg_ligand_receptor_module_pairs.csv", index=False)

    # Top pairs per module
    summary = (
        lr_df.groupby("module")
        .apply(lambda x: x.nlargest(5, "lr_score_sqrt_product")[["ligand", "receptor", "lr_score_sqrt_product"]].to_dict("records"))
        .to_dict()
    )
    (OUT / "tg_lr_top_pairs_by_module.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return lr_df


def enhance_drug_shortlist(module_genes: dict) -> pd.DataFrame:
    """Merge curated shortlist with Open Targets hits for module genes."""
    curated = pd.read_csv(project_root / "results" / "high_impact_integration" / "drug_repurposing_ntrk_synaptic_shortlist.csv")
    top_genes = []
    for mod in MODULES:
        top_genes.extend(module_genes[mod][:15])
    top_genes = list(dict.fromkeys(top_genes))[:25]

    ot = query_open_targets(top_genes)
    curated["source"] = "curated_module_DE"
    if len(ot):
        merged = pd.concat([curated, ot], ignore_index=True, sort=False)
    else:
        merged = curated.copy()
    merged.to_csv(OUT / "drug_repurposing_enhanced.csv", index=False)
    return merged


def write_stanford_sap() -> None:
    """Locked validation analysis plan for future Stanford cohort."""
    sap = """# Stanford External Validation — Statistical Analysis Plan (SAP)

**Version:** 1.0 (locked prior to data access)  
**Discovery cohort:** GSE186505 (n=10 TN blood; hypothesis-generating)  
**Validation cohort:** Stanford TN blood (target n≥25)

## Primary endpoint
**Peripheral module convergent validity:** `peripheral_ecm_schwann` module score differs across validation groups defined by pre-specified clinical stratification (peripheral-dominant vs central-dominant clinical/imaging features, or continuous axis replication).

- Test: Spearman correlation between discovery-derived `central_axis` score and validation cohort axis (permutation n=9,999)
- Success criterion: r ≥ 0.30, permutation p < 0.05, direction concordant with discovery

## Secondary endpoints
1. Module score directionality: peripheral_ecm_schwann, central_synaptic, ntrk_signaling (Welch t-test / Mann–Whitney with BH-FDR)
2. Portable 25-gene panel classification AUC (if labels available)
3. Matched blood–imaging–MVD outcome (if n≥10 matched): logistic regression MVD ~ central_axis + nerve ROI

## Excluded / non-claims
- Independent k=2 TN subtype replication is **not** a primary endpoint (underpowered in discovery)
- GSE177034/GSE124272/GSE150408 are convergent NP cohorts, not TN validation

## Frozen assets (do not modify after lock)
- `results/endotype_framework/portable_endotype_signatures.json`
- `results/high_impact_integration/ion_cci_human_ortholog_mapping.csv`
- Module direction signs: peripheral=up; central/ntrk=down

## Analysis code
`scripts/run_endotype_validation_tiers.py` + `scripts/run_natcomm_extension_analyses.py`
"""
    (OUT / "STANFORD_VALIDATION_SAP.md").write_text(sap, encoding="utf-8")


def make_figures(meta_df: pd.DataFrame, per_cohort: pd.DataFrame) -> None:
    """Generate Nat Comm figure panels."""
    logger.info("Generating Nat Comm figure panels")

    # Fig 3A: Forest plot — peripheral module across cohorts
    forest = per_cohort[
        (per_cohort["module"] == "peripheral_ecm_schwann")
        & (
            ((per_cohort["contrast"] == "persistent_vs_resolved") & (per_cohort["timepoint"] == "t1_followup"))
            | (per_cohort["contrast"] == "np_vs_healthy")
        )
    ].copy()
    if len(forest):
        forest["label"] = forest["cohort"] + "\n" + forest.get("timepoint", "").fillna("") + forest["contrast"]
        fig, ax = plt.subplots(figsize=(8, max(3, len(forest) * 0.8)))
        y = np.arange(len(forest))
        ax.errorbar(
            forest["cohens_d"], y,
            xerr=1.96 / np.sqrt(forest["n_case"] + forest["n_control"]),
            fmt="o", capsize=4, color="#2E86AB",
        )
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(forest["cohort"] + " (" + forest["contrast"] + ")")
        ax.set_xlabel("Cohen's d (case − control)")
        ax.set_title("Supplementary Figure S18. External NP cohort meta-analysis — peripheral ECM/Schwann module")
        plt.tight_layout()
        plt.savefig(FIG / "fig3_np_meta_forest_peripheral.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 3B: Module heatmap of p-values across cohorts
    pivot = per_cohort.pivot_table(
        index="cohort", columns="module", values="mannwhitney_p", aggfunc="min"
    )
    if pivot.shape[0] >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(-np.log10(pivot.clip(lower=1e-6)), annot=pivot.round(3), fmt="", cmap="YlOrRd", ax=ax)
        ax.set_title("Supplementary Figure S19. Module score tests across external NP blood cohorts")
        plt.tight_layout()
        plt.savefig(FIG / "fig3b_np_meta_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 4: TG cell type enrichment (from tier3 if available)
    tg_path = VAL_OUT / "tier3_module_tg_celltype_enrichment.csv"
    if tg_path.exists():
        tg = pd.read_csv(tg_path)
        ct_cols = [c for c in tg.columns if c.startswith("mean_")]
        if ct_cols:
            mat = tg.set_index("module")[ct_cols]
            mat.columns = [c.replace("mean_", "") for c in mat.columns]
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(mat, annot=True, fmt=".3f", cmap="Blues", ax=ax)
            ax.set_title("Supplementary Figure S20. Human TG cell-type enrichment of IoN-CCI modules")
            plt.tight_layout()
            plt.savefig(FIG / "fig4_tg_celltype_enrichment.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Fig 5: DTI atlas nerve-adjacent ROI tests (primary imaging)
    roi_tests = OUT / "imaging_nerve_roi_tests.csv"
    if roi_tests.exists():
        rt = pd.read_csv(roi_tests)
        if "metric_tier" in rt.columns:
            rt = rt[rt["metric_tier"] == "primary_dti_atlas"]
        top = rt.nsmallest(8, "mannwhitney_p")
        if len(top):
            fig, ax = plt.subplots(figsize=(9, 4))
            colors = ["#E94F37" if "central" in c else "#44AF69" for c in top["comparison"]]
            ax.barh(top["roi_metric"], -np.log10(top["mannwhitney_p"].clip(lower=1e-6)), color=colors)
            ax.set_xlabel("−log10(p)")
            ax.set_title("Supplementary Figure S21. DTI atlas nerve-adjacent ROI tests (primary imaging)")
            plt.tight_layout()
            plt.savefig(FIG / "fig5_imaging_nerve_roi.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Fig 6: Drug + LR summary
    lr_path = OUT / "tg_ligand_receptor_module_pairs.csv"
    drug_path = OUT / "drug_repurposing_enhanced.csv"
    if lr_path.exists() and drug_path.exists():
        lr = pd.read_csv(lr_path).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(lr["ligand"] + "→" + lr["receptor"], lr["lr_score_sqrt_product"], color="#6A4C93")
        ax.set_xlabel("LR score (√ ligand × receptor expr)")
        ax.set_title("Supplementary Figure S22. Drug repurposing and TG ligand–receptor translation hypotheses")
        plt.tight_layout()
        plt.savefig(FIG / "fig6_translation_lr_pairs.png", dpi=300, bbox_inches="tight")
        plt.close()


def write_summary(meta_df: pd.DataFrame) -> None:
    """Nat Comm package summary markdown."""
    lines = [
        "# Nature Communications Analysis Package",
        "",
        "**Script:** `scripts/run_natcomm_extension_analyses.py`",
        f"**Output:** `{OUT.relative_to(project_root)}/`",
        "",
        "## Claim framework (locked — see `results/CLAIM_TABLE_LOCKED.md`)",
        "",
        "| Supported | Not supported / future work |",
        "|-----------|-------------------------------|",
        "| Frozen IoN-CCI modules scoreable in 3 external NP blood cohorts | Independent TN k=2 subtype replication |",
        "| Mechanistic TG cell-type + LR anchoring (Tier 3) | Blood–imaging patient matching (n=0) |",
        "| DTI atlas nerve-adjacent ROI ↔ imaging endotype (exploratory) | CN V-specific segmentation (SEVB-Net → **future work**) |",
        "| Manual CN V pilot (n=10) infrastructure | Manual CN V primary claims (masks/QC pending) |",
        "| Drug/LR translation hypotheses | Clinical efficacy claims |",
        "",
        "**Imaging hierarchy:** Tier 1 = DTI in atlas ROIs; Tier 3 = legacy T1/T2 intensity (supplementary only).",
        "",
        "## Item 2: NP meta-analysis",
        "",
    ]
    if len(meta_df):
        for _, r in meta_df.iterrows():
            lines.append(
                f"- **{r['module']}**: {int(r['n_cohorts'])} cohorts; "
                f"Fisher p={r['fisher_combined_p']:.4g}; Stouffer p={r['stouffer_combined_p']:.4g}; "
                f"mean d={r['mean_cohens_d']:.2f}"
            )
    lines.extend([
        "",
        "## Item 3: Patient-linked outcome",
        "",
        "- **Matched blood–imaging–MVD:** none (see `patient_linkage_audit.json`)",
        "- **Imaging-only MVD:** see `mvd_outcome_by_imaging_endotype.csv`",
        "",
        "## Item 4: Imaging nerve-adjacent specificity (DTI-first)",
        "",
        "- Atlas ROIs: rez_bilateral, cisternal_bilateral, pons_nuclei_proxy",
        "- Primary metrics: FA/MD/RD/AD (not T1/T2 intensity)",
        "- SEVB-Net → future work; manual n=10 pilot → supplementary if QC complete",
        "- See `imaging_nerve_roi_tests.csv`, `../dti_nerve_adjacent/`",
        "",
        "## Item 5: Translation",
        "",
        "- `drug_repurposing_enhanced.csv` (curated + Open Targets)",
        "- `tg_ligand_receptor_module_pairs.csv`",
        "",
        "## Stanford SAP",
        "",
        "- `STANFORD_VALIDATION_SAP.md` (locked validation plan)",
        "",
        "## Figures",
        "",
        "- `figures/fig3_np_meta_forest_peripheral.png`",
        "- `figures/fig3b_np_meta_heatmap.png`",
        "- `figures/fig4_tg_celltype_enrichment.png`",
        "- `figures/fig5_imaging_nerve_roi.png`",
        "- `figures/fig6_translation_lr_pairs.png`",
    ])
    (OUT / "NATCOMM_PACKAGE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    logger.info("=" * 70)
    logger.info("NAT COMM EXTENSION ANALYSES")
    logger.info("=" * 70)

    module_genes = load_frozen_human_module_genes()
    meta_df = run_np_meta_analysis(module_genes)
    per_cohort = pd.read_csv(OUT / "np_meta_per_cohort_tests.csv")

    document_patient_linkage()
    run_imaging_nerve_concordance()
    run_ligand_receptor_analysis(module_genes)
    enhance_drug_shortlist(module_genes)
    write_stanford_sap()
    make_figures(meta_df, per_cohort)
    write_summary(meta_df)

    logger.info(f"Complete → {OUT}")


if __name__ == "__main__":
    main()
