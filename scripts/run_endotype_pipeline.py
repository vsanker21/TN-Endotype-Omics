"""
TN Endotype Analysis Pipeline (results only — no manuscript / publication figure generation).

Runs the locked-claims analysis chain from results/CLAIM_TABLE_LOCKED.md:

  1. TN-only k=2 blood subtyping + DE + pathways
  2. Blood deconvolution (corrected proportions)
  3. IoN-CCI module integration on TN blood
  4. Cross-dataset integration (modules, MVD, pre/post)
  5. Three-tier endotype validation (NP meta + internal + TG hub)
  6. Endotype framework (portable signatures, population bridge)
  7. DTI nerve-adjacent cohort metrics
  8. Imaging QC audit
  9. Manual CN V pilot export (infrastructure)
 10. NatComm extension analyses + Stanford SAP
 11. k=3 sensitivity (supplementary tables)

Usage:
    python scripts/run_endotype_pipeline.py
    python scripts/run_endotype_pipeline.py --from-step 4
    python scripts/run_endotype_pipeline.py --only high_impact,endotype_framework
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

STEPS = [
    ("subtyping", "recluster_tn_only_subtypes.py", "TN-only k=2 subtyping, DE, pathways"),
    ("deconvolution", "fix_deconvolution_and_regenerate_figures.py", "Corrected blood deconvolution"),
    ("ion_cci_modules", "integrate_ion_cci_pathways_with_subtypes.py", "IoN-CCI pathway × subtype integration"),
    ("high_impact", "run_high_impact_cross_dataset_analyses.py", "Module scores, MVD, mechanism–imaging"),
    ("validation_tiers", "run_endotype_validation_tiers.py", "External NP + internal + TG validation"),
    ("endotype_framework", "run_endotype_framework_analyses.py", "Portable signatures + population bridge"),
    ("dti_cohort", "run_dti_nerve_adjacent_cohort.py", "DTI atlas ROI cohort metrics"),
    ("imaging_qc", "run_imaging_qc_audit.py", "Imaging QC vs blood endotypes"),
    ("cnv_pilot", "run_manual_cnv_pilot_cohort.py", "Manual CN V pilot (--export-only)"),
    ("natcomm", "run_natcomm_extension_analyses.py", "NP meta, TG enrichment, DTI tests, SAP"),
    ("k3_sensitivity", "run_k3_sensitivity_analysis.py", "k=3 sensitivity tables"),
]


def run_step(script: str, extra_args: list[str] | None = None) -> bool:
    path = project_root / "scripts" / script
    if not path.exists():
        print(f"  SKIP (missing): {script}")
        return False
    cmd = [sys.executable, str(path)] + (extra_args or [])
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TN endotype analysis pipeline (results only)")
    parser.add_argument("--from-step", type=int, default=1, help="1-based step index to start from")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated step ids (e.g. high_impact,endotype_framework)",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()

    print("=" * 70)
    print("TN ENDOTYPE ANALYSIS PIPELINE")
    print(f"Project root: {project_root}")
    print("=" * 70)

    ok = True
    for i, (step_id, script, desc) in enumerate(STEPS, 1):
        if i < args.from_step:
            continue
        if only and step_id not in only:
            continue
        print(f"\n--- Step {i}/{len(STEPS)}: {step_id} — {desc} ---")
        extra: list[str] = []
        if script == "run_manual_cnv_pilot_cohort.py":
            extra = ["--export-only"]
        if not run_step(script, extra):
            print(f"  FAILED: {script}")
            ok = False
            break

    print("\n" + "=" * 70)
    if ok:
        print("Pipeline complete. Key outputs:")
        print("  results/molecular_subtypes.csv")
        print("  results/high_impact_integration/")
        print("  results/endotype_framework/")
        print("  results/endotype_validation/")
        print("  results/natcomm_package/")
        print("  results/dti_nerve_adjacent/")
        print("  results/tn_only_subtyping/k3_sensitivity/")
    else:
        print("Pipeline stopped with errors.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
