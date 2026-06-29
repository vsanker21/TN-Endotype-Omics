"""
Run differential expression analysis on IoN-CCI data.

This script:
1. Collects gene counts from STAR alignment outputs
2. Creates count matrix
3. Runs DESeq2 differential expression analysis
4. Compares IoN-CCI vs Sham for TG and Sp5C tissues
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import subprocess
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

def collect_star_counts(processed_dir):
    """Collect gene count files from STAR alignments or featureCounts."""
    logger.info("Collecting gene counts...")
    
    count_files = {}
    for sample_dir in processed_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        
        sample_id = sample_dir.name
        count_file = None
        
        # Try STAR ReadsPerGene files first
        count_file = sample_dir / f"{sample_id}_ReadsPerGene.out.tab"
        if not count_file.exists():
            count_file = sample_dir / "ReadsPerGene.out.tab"
        
        # Try featureCounts output
        if not count_file.exists():
            count_file = sample_dir / f"{sample_id}_featureCounts.txt"
        
        if count_file.exists():
            count_files[sample_id] = count_file
            logger.info(f"  Found counts for {sample_id}: {count_file.name}")
        else:
            logger.warning(f"  No counts found for {sample_id}")
    
    logger.info(f"Found {len(count_files)} samples with counts")
    return count_files

def create_count_matrix(count_files):
    """Create count matrix from STAR ReadsPerGene or featureCounts files."""
    logger.info("Creating count matrix...")
    
    all_genes = set()
    count_data = {}
    
    # First pass: collect all gene IDs
    for sample_id, count_file in count_files.items():
        try:
            # Check if it's STAR or featureCounts format
            if 'featureCounts' in count_file.name:
                # featureCounts format: tab-separated, first column is Geneid
                df = pd.read_csv(count_file, sep='\t', skiprows=1)  # Skip header comment
                all_genes.update(df['Geneid'].tolist())
            else:
                # STAR ReadsPerGene format
                # Read all rows, then drop summary rows starting with 'N_'
                df = pd.read_csv(count_file, sep='\t', header=None)
                df = df[~df[0].astype(str).str.startswith('N_')]
                all_genes.update(df[0].tolist())
        except Exception as e:
            logger.warning(f"  Error reading {sample_id}: {e}")
    
    all_genes = sorted(list(all_genes))
    logger.info(f"  Found {len(all_genes)} genes")
    
    # Second pass: collect counts
    for sample_id, count_file in count_files.items():
        try:
            if 'featureCounts' in count_file.name:
                # featureCounts format
                df = pd.read_csv(count_file, sep='\t', skiprows=1)
                # Last column is the count column (sample name)
                count_col = df.columns[-1]
                df = df.set_index('Geneid')
                counts = pd.Series(0, index=all_genes)
                counts.loc[df.index] = df[count_col]
                count_data[sample_id] = counts
            else:
                # STAR ReadsPerGene format
                # Read all rows, then drop summary rows starting with 'N_'
                df = pd.read_csv(count_file, sep='\t', header=None)
                df = df[~df[0].astype(str).str.startswith('N_')]
                df.columns = ['gene_id', 'unstranded', 'forward', 'reverse']
                df = df.set_index('gene_id')
                
                # Create series with all genes (fill missing with 0)
                counts = pd.Series(0, index=all_genes)
                counts.loc[df.index] = df['unstranded']
                count_data[sample_id] = counts
            
        except Exception as e:
            logger.warning(f"  Error processing {sample_id}: {e}")
            # Fill with zeros if failed
            count_data[sample_id] = pd.Series(0, index=all_genes)
    
    # Create count matrix
    count_matrix = pd.DataFrame(count_data, index=all_genes)
    count_matrix = count_matrix.astype(int)
    
    logger.info(f"  Count matrix shape: {count_matrix.shape}")
    logger.info(f"  Total counts: {count_matrix.sum().sum():,}")
    
    return count_matrix

def create_sample_metadata(count_files):
    """Create sample metadata from sample IDs."""
    logger.info("Creating sample metadata...")
    
    metadata = []
    for sample_id in count_files.keys():
        # Parse sample ID to extract condition and tissue
        # Format: SRR25158547 (Sham TG), SRR25158553 (IoN-CCI TG), etc.
        # Based on known sample IDs from ION_CCI_PROCESSING_STATUS.md
        sample_info = {
            'sample_id': sample_id,
            'condition': 'unknown',
            'tissue': 'unknown'
        }
        
        # Known sample mapping (from processing status)
        sample_mapping = {
            'SRR25158547': ('sham', 'tg'),
            'SRR25158548': ('sham', 'tg'),
            'SRR25158549': ('sham', 'tg'),
            'SRR25158550': ('sham', 'tg'),
            'SRR25158551': ('sham', 'sp5c'),
            'SRR25158552': ('sham', 'sp5c'),
            'SRR25158553': ('ion_cci', 'tg'),
            'SRR25158554': ('ion_cci', 'tg'),
            'SRR25158555': ('sham', 'sp5c'),
            'SRR25158556': ('sham', 'sp5c'),
            'SRR25158557': ('ion_cci', 'sp5c'),
            'SRR25158558': ('ion_cci', 'sp5c'),
        }
        
        if sample_id in sample_mapping:
            sample_info['condition'], sample_info['tissue'] = sample_mapping[sample_id]
        else:
            logger.warning(f"  Unknown sample ID: {sample_id}")
        
        metadata.append(sample_info)
    
    metadata_df = pd.DataFrame(metadata)
    logger.info(f"  Created metadata for {len(metadata_df)} samples")
    logger.info(f"  Conditions: {metadata_df['condition'].value_counts().to_dict()}")
    logger.info(f"  Tissues: {metadata_df['tissue'].value_counts().to_dict()}")
    
    return metadata_df

def run_deseq2_analysis(count_matrix, metadata, output_dir):
    """Run DESeq2 analysis using R script."""
    logger.info("Running DESeq2 analysis...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save count matrix and metadata
    count_file = output_dir / "ion_cci_counts.csv"
    meta_file = output_dir / "ion_cci_metadata.csv"
    
    count_matrix.to_csv(count_file)
    metadata.to_csv(meta_file, index=False)
    
    logger.info(f"  Saved count matrix: {count_file}")
    logger.info(f"  Saved metadata: {meta_file}")

    # Locate Rscript executable (robust lookup)
    try:
        import shutil, glob
        rscript_path = shutil.which("Rscript")
        if not rscript_path:
            # Try common Windows R locations
            r_paths = glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe")
            if r_paths:
                rscript_path = r_paths[0]
        if not rscript_path:
            logger.error("Rscript not found. Please install R and ensure Rscript is on PATH, or install R under 'C:/Program Files/R/'.")
            return False

        # Check Rscript works
        result = subprocess.run([rscript_path, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error(f"Rscript not available or failed to run: {result.stderr}")
            return False
        logger.info(f"  Using Rscript at: {rscript_path}")
        logger.info(f"  Rscript version: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Error locating Rscript: {e}")
        return False
    
    # Run DESeq2 R script
    r_script = project_root / "scripts" / "run_ion_cci_deseq2.R"
    if not r_script.exists():
        logger.error(f"DESeq2 R script not found: {r_script}")
        logger.info("Creating DESeq2 R script...")
        create_deseq2_script(r_script, output_dir)
    
    logger.info("  Running DESeq2...")
    # Pass output directory as argument so the R script knows where to read/write files
    result = subprocess.run(
        [rscript_path, str(r_script), str(output_dir)],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=1800  # 30 minutes
    )
    
    if result.returncode == 0:
        logger.info("  ✅ DESeq2 analysis complete")
        logger.info(result.stdout[-500:])
        return True
    else:
        logger.error("  ❌ DESeq2 analysis failed")
        logger.error(result.stderr[-1000:])
        return False

def create_deseq2_script(r_script_path, output_dir):
    """Create DESeq2 R script if it doesn't exist."""
    r_script_content = f"""
# DESeq2 Differential Expression Analysis for IoN-CCI
library(DESeq2)
library(dplyr)

# Load data
count_matrix <- read.csv("{output_dir}/ion_cci_counts.csv", row.names=1)
metadata <- read.csv("{output_dir}/ion_cci_metadata.csv")

# Ensure sample order matches
count_matrix <- count_matrix[, metadata$sample_id]

# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = metadata,
    design = ~ condition + tissue + condition:tissue
)

# Filter low count genes
keep <- rowSums(counts(dds)) >= 10
dds <- dds[keep,]

# Run DESeq2
dds <- DESeq(dds)

# Results: IoN-CCI vs Sham for TG
results_tg <- results(dds, contrast=c("condition", "ion_cci", "sham"), 
                     filter=list(tissue="tg"))
results_tg_df <- as.data.frame(results_tg)
results_tg_df$gene <- rownames(results_tg_df)
write.csv(results_tg_df, "{output_dir}/de_ion_cci_vs_sham_tg.csv", row.names=FALSE)

# Results: IoN-CCI vs Sham for Sp5C
results_sp5c <- results(dds, contrast=c("condition", "ion_cci", "sham"),
                       filter=list(tissue="sp5c"))
results_sp5c_df <- as.data.frame(results_sp5c)
results_sp5c_df$gene <- rownames(results_sp5c_df)
write.csv(results_sp5c_df, "{output_dir}/de_ion_cci_vs_sham_sp5c.csv", row.names=FALSE)

# Results: Combined (all tissues)
results_combined <- results(dds, contrast=c("condition", "ion_cci", "sham"))
results_combined_df <- as.data.frame(results_combined)
results_combined_df$gene <- rownames(results_combined_df)
write.csv(results_combined_df, "{output_dir}/de_ion_cci_vs_sham_combined.csv", row.names=FALSE)

cat("DESeq2 analysis complete!\\n")
cat("Results saved to:", "{output_dir}/\\n")
"""
    
    with open(r_script_path, 'w') as f:
        f.write(r_script_content)
    
    logger.info(f"  Created DESeq2 R script: {r_script_path}")

def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("ION-CCI DIFFERENTIAL EXPRESSION ANALYSIS")
    logger.info("=" * 70)
    
    # Find processed samples
    processed_dir = project_root / "results" / "ion_cci_processing" / "processed"
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return False
    
    # Collect counts
    count_files = collect_star_counts(processed_dir)
    
    if len(count_files) == 0:
        logger.error("No count files found. Please run STAR alignment first.")
        return False
    
    # Create count matrix
    count_matrix = create_count_matrix(count_files)
    
    # Create metadata
    metadata = create_sample_metadata(count_files)
    
    # Run DESeq2
    output_dir = project_root / "results" / "aim3" / "ion_cci" / "de_analysis"
    success = run_deseq2_analysis(count_matrix, metadata, output_dir)
    
    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✅ DIFFERENTIAL EXPRESSION ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Review DE results")
        logger.info("  2. Run pathway enrichment")
        logger.info("  3. Integrate with molecular subtyping")
    else:
        logger.error("\n❌ Differential expression analysis failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

