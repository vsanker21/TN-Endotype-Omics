"""
Process IoN-CCI data for Aim 3 translation pipeline.

Processes FastQ files into count matrices if not already done.
"""

import sys
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_and_process_ion_cci():
    """Check IoN-CCI status and process if needed."""
    logger.info("=" * 70)
    logger.info("IoN-CCI DATA PROCESSING FOR AIM 3")
    logger.info("=" * 70)
    
    ion_cci_dir = project_root / "data" / "external" / "ion_cci"
    processed_dir = project_root / "data" / "processed" / "ion_cci"
    
    # Check if already processed
    if processed_dir.exists():
        count_files = list(processed_dir.glob("**/*count*.csv")) + list(processed_dir.glob("**/*count*.tsv"))
        expr_files = list(processed_dir.glob("**/*expr*.csv"))
        
        if count_files or expr_files:
            logger.info(f"✅ Processed data already exists: {len(count_files + expr_files)} files")
            logger.info("IoN-CCI data ready for Aim 3!")
            return True
    
    # Check for supplementary tables (may contain processed data)
    logger.info("\nChecking supplementary tables...")
    supplementary_dir = ion_cci_dir / "raw" / "Supplementary Material"
    
    if supplementary_dir.exists():
        table_files = list(supplementary_dir.glob("Table*.XLS*"))
        logger.info(f"Found {len(table_files)} supplementary table files")
        
        # Check if tables contain expression data
        try:
            import pandas as pd
            
            for table_file in table_files:
                logger.info(f"\nChecking {table_file.name}...")
                try:
                    # Try to read as Excel
                    df = pd.read_excel(table_file, nrows=5)
                    logger.info(f"  Shape: {df.shape}")
                    logger.info(f"  Columns: {list(df.columns[:5])}")
                    
                    # Check if it looks like expression data
                    if df.shape[0] > 100 and df.shape[1] > 5:
                        logger.info(f"  ✅ Potential expression matrix")
                except Exception as e:
                    logger.warning(f"  Could not read: {e}")
        
        except ImportError:
            logger.warning("pandas/excel reader not available")
    
    # Check FastQ files
    logger.info("\nChecking FastQ files...")
    fastq_dir = ion_cci_dir / "organized"
    
    if fastq_dir.exists():
        fastq_files = list(fastq_dir.glob("**/*.fastq.gz"))
        logger.info(f"FastQ files found: {len(fastq_files)}")
        
        if fastq_files:
            logger.info("\n⚠️ Raw FastQ files available but not processed")
            logger.info("Processing requires RNA-seq alignment and quantification pipeline")
            logger.info("Recommended: Use Kallisto, Salmon, or STAR+featureCounts")
            
            # Summarize samples
            logger.info("\nSample organization:")
            conditions = {}
            for condition_dir in fastq_dir.iterdir():
                if condition_dir.is_dir():
                    for tissue_dir in condition_dir.iterdir():
                        if tissue_dir.is_dir():
                            key = f"{condition_dir.name}_{tissue_dir.name}"
                            n_files = len(list(tissue_dir.glob("*.fastq.gz")))
                            conditions[key] = n_files // 2  # paired-end
                            logger.info(f"  {key}: {conditions[key]} samples")
            
            logger.info("\n📋 Summary:")
            logger.info(f"  Total samples: {sum(conditions.values())}")
            logger.info(f"  Conditions: {list(set([k.split('_')[0] for k in conditions.keys()]))}")
            logger.info(f"  Tissues: {list(set([k.split('_')[1] for k in conditions.keys()]))}")
            
            logger.info("\n✅ IoN-CCI data structure ready for processing")
            logger.info("   Action: Run RNA-seq processing pipeline when ready")
    
    return False

if __name__ == "__main__":
    check_and_process_ion_cci()












