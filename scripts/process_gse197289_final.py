"""
Final processing script for GSE197289 - handles RDS.gz files properly.
"""

import sys
from pathlib import Path
import subprocess
import gzip
import shutil as shutil_module
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def decompress_and_convert_rds():
    """Decompress RDS.gz and convert using R."""
    logger.info("=" * 70)
    logger.info("PROCESSING GSE197289 - DECOMPRESS AND CONVERT")
    logger.info("=" * 70)
    
    data_dir = project_root / "data" / "external" / "geo" / "GSE197289" / "GSE197289"
    input_file = data_dir / "GSE197289_snRNA-seq_human_raw_counts.RDS.gz"
    meta_file = data_dir / "GSE197289_snRNA-seq_human_barcode_meta.csv.gz"
    
    if not input_file.exists():
        return False
    
    # Decompress RDS file
    logger.info("Decompressing RDS file...")
    decompressed_rds = data_dir / "human_counts.RDS"
    
    try:
        with gzip.open(input_file, 'rb') as f_in:
            with open(decompressed_rds, 'wb') as f_out:
                shutil_module.copyfileobj(f_in, f_out)
        logger.info(f"✅ Decompressed to: {decompressed_rds}")
    except Exception as e:
        logger.error(f"Decompression error: {e}")
        return False
    
    # Decompress metadata
    logger.info("Decompressing metadata...")
    decompressed_meta = data_dir / "human_meta.csv"
    
    try:
        with gzip.open(meta_file, 'rb') as f_in:
            with open(decompressed_meta, 'wb') as f_out:
                shutil_module.copyfileobj(f_in, f_out)
        logger.info(f"✅ Decompressed to: {decompressed_meta}")
    except Exception as e:
        logger.warning(f"Metadata decompression error: {e}")
        decompressed_meta = None
    
    # Convert RDS to Matrix Market using R
    logger.info("\nConverting RDS to Matrix Market format...")
    
    input_rds = str(decompressed_rds).replace('\\', '/')
    output_mtx = str(data_dir / "human_counts.mtx").replace('\\', '/')
    output_genes = str(data_dir / "human_counts_genes.csv").replace('\\', '/')
    output_barcodes = str(data_dir / "human_counts_barcodes.csv").replace('\\', '/')
    
    r_script = f"""
    library(Matrix)
    data <- readRDS("{input_rds}")
    cat("Loaded:", dim(data)[1], "genes x", dim(data)[2], "cells\\n")
    
    # Transpose for scanpy (scanpy expects cells x genes)
    data_t <- t(data)
    cat("Transposed:", dim(data_t)[1], "cells x", dim(data_t)[2], "genes\\n")
    
    # Save as Matrix Market format (cells x genes)
    writeMM(data_t, "{output_mtx}")
    # Cell names (rows in transposed matrix)
    write.csv(colnames(data), "{output_barcodes}", row.names = FALSE, quote = FALSE)
    # Gene names (columns in transposed matrix)
    write.csv(rownames(data), "{output_genes}", row.names = FALSE, quote = FALSE)
    cat("Saved Matrix Market format\\n")
    """
    
    try:
        rscript_path = shutil_module.which("Rscript")
        if not rscript_path:
            import glob
            r_paths = glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe")
            if r_paths:
                rscript_path = r_paths[0]
        
        if not rscript_path:
            logger.error("Rscript not found")
            return False
        
        logger.info("Running R conversion...")
        result = subprocess.run(
            [rscript_path, "--vanilla", "-"],
            input=r_script,
            text=True,
            capture_output=True,
            timeout=300,
            cwd=str(project_root)
        )
        
        if result.returncode == 0:
            logger.info("✅ R conversion successful")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"R conversion error: {result.stderr}")
            logger.info(result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def load_and_process_mtx():
    """Load Matrix Market format and process with scanpy."""
    logger.info("\n" + "=" * 70)
    logger.info("LOADING AND PROCESSING MATRIX MARKET DATA")
    logger.info("=" * 70)
    
    data_dir = project_root / "data" / "external" / "geo" / "GSE197289" / "GSE197289"
    
    mtx_file = data_dir / "human_counts.mtx"
    genes_file = data_dir / "human_counts_genes.csv"
    barcodes_file = data_dir / "human_counts_barcodes.csv"
    
    if not mtx_file.exists():
        logger.error("Matrix Market file not found. Run conversion first.")
        return None
    
    try:
        import scanpy as sc
        import pandas as pd
        
        logger.info("Loading Matrix Market format...")
        adata = sc.read_mtx(mtx_file)
        
        # Load gene and cell names
        # Note: scanpy read_mtx creates (cells x genes), so:
        # - var_names (genes) should match shape[1]
        # - obs_names (cells) should match shape[0]
        
        genes_df = pd.read_csv(genes_file, header=None)
        barcodes_df = pd.read_csv(barcodes_file, header=None)
        
        # Matrix Market format from R: genes are rows, cells are columns
        # But scanpy read_mtx reads as (cells x genes) by default
        # Check dimensions
        logger.info(f"Matrix shape: {adata.shape} (cells x genes)")
        logger.info(f"Genes CSV rows: {len(genes_df)}, Barcodes CSV rows: {len(barcodes_df)}")
        
        # The RDS had genes as rows (32214) and cells as columns (38028)
        # But scanpy read_mtx transposes to (cells x genes)
        # So: adata.shape = (32214 cells, 38028 genes)
        # We need: 32214 cell names, 38028 gene names
        
        # Skip first row (header) if present
        if len(barcodes_df) == adata.shape[0] + 1:
            barcodes = barcodes_df[0].values[1:]  # Skip header for cells
        else:
            barcodes = barcodes_df[0].values
        
        if len(genes_df) == adata.shape[1] + 1:
            genes = genes_df[0].values[1:]  # Skip header for genes
        else:
            genes = genes_df[0].values
        
        # Assign: obs_names = cells, var_names = genes
        adata.obs_names = barcodes
        adata.var_names = genes
        
        logger.info(f"✅ Loaded: {adata.shape} (cells x genes)")
        
        # Load metadata if available
        meta_file = data_dir / "human_meta.csv"
        if meta_file.exists():
            logger.info("Loading metadata...")
            meta = pd.read_csv(meta_file)
            if len(meta) == adata.shape[0]:
                for col in meta.columns:
                    if col not in ['X', 'barcode']:
                        adata.obs[col] = meta[col].values
                logger.info(f"✅ Added metadata: {list(meta.columns)}")
        
        # Process
        logger.info("\nProcessing with scanpy...")
        
        # QC
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        logger.info(f"Before filtering: {adata.shape}")
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        logger.info(f"After filtering: {adata.shape}")
        
        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # HVG
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        logger.info(f"HVG: {adata.var['highly_variable'].sum()}")
        
        # PCA
        sc.tl.pca(adata, svd_solver='arpack')
        
        # Neighbors and UMAP (with fallback for PyTorch issues)
        try:
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata)
        except Exception as e:
            logger.warning(f"Neighbors/UMAP failed ({e}), using PCA-based clustering")
            # Use PCA for clustering instead
            pass
        
        # Clustering
        try:
            sc.tl.leiden(adata, resolution=0.5)
        except:
            # Fallback to KMeans if Leiden fails
            from sklearn.cluster import KMeans
            logger.info("Using KMeans clustering instead of Leiden")
            kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
            adata.obs['leiden'] = kmeans.fit_predict(adata.obsm['X_pca'][:, :50]).astype(str)
        n_clusters = adata.obs['leiden'].nunique()
        logger.info(f"Clusters: {n_clusters}")
        
        # Markers
        sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon', n_genes=50)
        
        # Annotate cell types
        try:
            from scripts.process_gse197289_single_nucleus import annotate_cell_types
            adata = annotate_cell_types(adata)
        except:
            logger.warning("Cell type annotation skipped")
        
        # Save
        output_dir = project_root / "data" / "processed" / "GSE197289"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "human_tg_single_nucleus_processed.h5ad"
        adata.write(output_file)
        logger.info(f"\n✅ Saved: {output_file}")
        
        return adata
        
    except Exception as e:
        logger.error(f"Error processing: {e}", exc_info=True)
        return None

def main():
    """Main execution."""
    # Step 1: Decompress and convert
    if decompress_and_convert_rds():
        # Step 2: Load and process
        adata = load_and_process_mtx()
        
        if adata is not None:
            logger.info("\n" + "=" * 70)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"\n✅ Processed: {adata.shape[0]} cells, {adata.shape[1]} genes")
            logger.info(f"   Clusters: {adata.obs['leiden'].nunique()}")
            if 'cell_type' in adata.obs.columns:
                logger.info(f"   Cell types: {adata.obs['cell_type'].nunique()}")
            logger.info("\nNext: Integrate with Aim 1")
        else:
            logger.error("Processing failed")
    else:
        logger.error("Conversion failed")

if __name__ == "__main__":
    main()

