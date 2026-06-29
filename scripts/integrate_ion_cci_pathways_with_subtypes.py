"""
Integrate IoN-CCI pathway enrichment results with molecular subtyping.

This script:
1. Loads IoN-CCI DE results and pathway enrichment
2. Loads molecular subtype assignments from Aim 1
3. Maps pathways to subtypes based on shared genes
4. Creates integrated summary
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")

def load_ion_cci_de_results(de_dir):
    """Load IoN-CCI differential expression results."""
    logger.info("Loading IoN-CCI DE results...")
    
    de_files = {
        'tg': de_dir / "de_ion_cci_vs_sham_tg.csv",
        'sp5c': de_dir / "de_ion_cci_vs_sham_sp5c.csv",
        'combined': de_dir / "de_ion_cci_vs_sham_combined.csv"
    }
    
    de_results = {}
    for tissue, file_path in de_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filter significant genes
            sig_df = df[(df['padj'] < 0.05) & (abs(df['log2FoldChange']) > 0.5)].copy()
            de_results[tissue] = {
                'all': df,
                'significant': sig_df
            }
            logger.info(f"  {tissue}: {len(df)} genes, {len(sig_df)} significant")
        else:
            logger.warning(f"  {tissue}: DE file not found: {file_path}")
            de_results[tissue] = None
    
    return de_results

def load_pathway_enrichment(pathway_dir):
    """Load pathway enrichment results for IoN-CCI."""
    logger.info("Loading pathway enrichment results...")
    
    # Look for pathway files with pattern: {tissue}_{direction}_pathways.csv
    pathway_files = list(pathway_dir.glob("*_pathways.csv"))
    
    pathways = {}
    for file_path in pathway_files:
        # Parse filename: tg_upregulated_pathways.csv or sp5c_downregulated_pathways.csv
        parts = file_path.stem.replace('_pathways', '').split('_')
        
        if len(parts) >= 2:
            tissue = parts[0]  # tg, sp5c, or combined
            direction = '_'.join(parts[1:])  # upregulated or downregulated
            
            if tissue in ['tg', 'sp5c', 'combined']:
                key = f"{tissue}_{direction}"
                df = pd.read_csv(file_path)
                # Filter significant pathways
                if 'Adjusted P-value' in df.columns:
                    sig_df = df[df['Adjusted P-value'] < 0.1].copy()
                elif 'P-value' in df.columns:
                    sig_df = df[df['P-value'] < 0.05].copy()
                else:
                    sig_df = df.copy()
                
                pathways[key] = {
                    'all': df,
                    'significant': sig_df,
                    'direction': direction
                }
                logger.info(f"  {key}: {len(df)} pathways, {len(sig_df)} significant")
    
    if len(pathways) == 0:
        logger.warning("  No pathway enrichment files found")
    
    return pathways

def convert_gene_ids_to_symbols(gene_ids):
    """Convert Ensembl gene IDs to gene symbols using mygene."""
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        
        gene_symbols = {}
        batch_size = 1000
        for i in range(0, len(gene_ids), batch_size):
            batch = gene_ids[i:i+batch_size]
            results = mg.querymany(batch, scopes='ensembl.gene', fields='symbol', species='mouse', returnall=True)
            
            for result in results['out']:
                query = result['query']
                if 'symbol' in result and result['symbol']:
                    gene_symbols[query] = result['symbol']
                else:
                    gene_symbols[query] = query  # Use ID if no symbol
        
        logger.info(f"  Converted {len(gene_symbols)}/{len(gene_ids)} gene IDs to symbols")
        return gene_symbols
    except ImportError:
        logger.warning("mygene not installed. Using gene IDs directly.")
        return {gid: gid for gid in gene_ids}
    except Exception as e:
        logger.warning(f"Error converting gene IDs: {e}. Using gene IDs directly.")
        return {gid: gid for gid in gene_ids}

def load_molecular_subtypes():
    """Load molecular subtype assignments from Aim 1."""
    logger.info("Loading molecular subtypes...")
    
    subtype_file = project_root / "results" / "molecular_subtypes.csv"
    if not subtype_file.exists():
        logger.warning(f"Subtype file not found: {subtype_file}")
        return None
    
    subtypes = pd.read_csv(subtype_file)
    logger.info(f"  Loaded {len(subtypes)} subtype assignments")
    logger.info(f"  Subtypes: {subtypes['subtype'].value_counts().to_dict()}")
    
    return subtypes

def load_subtype_de_results():
    """Load subtype-specific DE results from Aim 1."""
    logger.info("Loading subtype DE results...")
    
    de_dir = project_root / "results" / "subtype_characterization"
    de_files = {
        'subtype_0': de_dir / "de_subtype_0.csv",
        'subtype_1': de_dir / "de_subtype_1.csv",
        'subtype_2': de_dir / "de_subtype_2.csv"
    }
    
    subtype_de = {}
    for subtype, file_path in de_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Filter significant genes
            sig_df = df[(df['p_adj'] < 0.05) & (abs(df['log2fc']) > 0.5)].copy()
            subtype_de[subtype] = {
                'all': df,
                'significant': sig_df
            }
            logger.info(f"  {subtype}: {len(df)} genes, {len(sig_df)} significant")
        else:
            logger.warning(f"  {subtype}: DE file not found: {file_path}")
    
    return subtype_de

def map_pathways_to_subtypes(ion_cci_de, subtype_de, pathways):
    """Map IoN-CCI pathways to molecular subtypes based on shared genes."""
    logger.info("Mapping pathways to subtypes...")
    
    integration_results = []
    
    # For each IoN-CCI pathway result (format: {tissue}_{direction})
    for pathway_key, pathway_data in pathways.items():
        if pathway_data is None:
            continue
        
        # Parse key: e.g., "tg_upregulated" -> tissue="tg", direction="upregulated"
        parts = pathway_key.split('_', 1)
        if len(parts) < 2:
            continue
        tissue = parts[0]
        direction = parts[1]
        
        # Get IoN-CCI significant genes for this tissue
        if tissue in ion_cci_de and ion_cci_de[tissue] is not None:
            # Filter by direction if needed
            if direction == 'upregulated':
                ion_cci_genes_df = ion_cci_de[tissue]['significant'][
                    ion_cci_de[tissue]['significant']['log2FoldChange'] > 1
                ]
            elif direction == 'downregulated':
                ion_cci_genes_df = ion_cci_de[tissue]['significant'][
                    ion_cci_de[tissue]['significant']['log2FoldChange'] < -1
                ]
            else:
                ion_cci_genes_df = ion_cci_de[tissue]['significant']
            
            # Get gene IDs and convert to symbols for matching (pathways use symbols)
            gene_ids = ion_cci_genes_df['gene'].tolist()
            gene_id_to_symbol = convert_gene_ids_to_symbols(gene_ids)
            ion_cci_genes = set([gene_id_to_symbol.get(gid, gid) for gid in gene_ids])
        else:
            continue
        
        # For each significant pathway
        for _, pathway_row in pathway_data['significant'].iterrows():
            pathway_name = pathway_row['Term']
            pathway_genes_str = pathway_row.get('Genes', '')
            gene_set = pathway_row.get('Gene_set', 'Unknown')
            
            # Extract database name from Gene_set (e.g., "Reactome_2022" -> "Reactome")
            if '_' in gene_set:
                db = gene_set.split('_')[0]
            else:
                db = gene_set
            
            if pd.isna(pathway_genes_str) or pathway_genes_str == '':
                continue
            
            # Parse pathway genes (these are gene symbols)
            pathway_genes = set([g.strip() for g in str(pathway_genes_str).split(';')])
            
            # Find overlap with IoN-CCI genes
            # Note: This assumes gene IDs can match symbols, or we need conversion
            overlap = ion_cci_genes.intersection(pathway_genes)
            
            if len(overlap) == 0:
                continue
            
            # For each subtype, check overlap with subtype-specific genes
            for subtype_key, subtype_data in subtype_de.items():
                if subtype_data is None:
                    continue
                
                # Convert subtype gene IDs to symbols
                subtype_gene_ids = subtype_data['significant']['gene'].tolist()
                if len(subtype_gene_ids) > 0:
                    subtype_id_to_symbol = convert_gene_ids_to_symbols(subtype_gene_ids)
                    subtype_genes = set([subtype_id_to_symbol.get(gid, gid) for gid in subtype_gene_ids])
                else:
                    subtype_genes = set()
                
                subtype_overlap = overlap.intersection(subtype_genes)
                
                if len(subtype_overlap) > 0:
                    integration_results.append({
                        'pathway': pathway_name,
                        'pathway_database': db,
                        'pathway_direction': direction,
                        'ion_cci_tissue': tissue,
                        'subtype': subtype_key,
                        'ion_cci_genes_in_pathway': len(overlap),
                        'subtype_genes_in_pathway': len(subtype_overlap),
                        'shared_genes': '; '.join(sorted(list(subtype_overlap)[:20])),  # Limit to first 20
                        'pathway_pvalue': pathway_row.get('Adjusted P-value', np.nan),
                        'pathway_odds_ratio': pathway_row.get('Odds Ratio', np.nan)
                    })
    
    integration_df = pd.DataFrame(integration_results)
    
    if len(integration_df) > 0:
        logger.info(f"  Found {len(integration_df)} pathway-subtype associations")
    else:
        logger.warning("  No pathway-subtype associations found")
    
    return integration_df

def create_integration_summary(integration_df, output_dir):
    """Create integration summary report."""
    logger.info("Creating integration summary...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full integration results
    integration_file = output_dir / "ion_cci_pathway_subtype_integration.csv"
    integration_df.to_csv(integration_file, index=False)
    logger.info(f"  Saved: {integration_file}")
    
    # Create summary by subtype
    summary_by_subtype = integration_df.groupby('subtype').agg({
        'pathway': 'count',
        'ion_cci_genes_in_pathway': 'sum',
        'subtype_genes_in_pathway': 'sum'
    }).rename(columns={'pathway': 'n_pathways'})
    
    summary_file = output_dir / "integration_summary_by_subtype.csv"
    summary_by_subtype.to_csv(summary_file)
    logger.info(f"  Saved: {summary_file}")
    
    # Create summary by pathway database
    summary_by_db = integration_df.groupby(['pathway_database', 'subtype']).agg({
        'pathway': 'count'
    }).rename(columns={'pathway': 'n_pathways'})
    
    db_summary_file = output_dir / "integration_summary_by_database.csv"
    summary_by_db.to_csv(db_summary_file)
    logger.info(f"  Saved: {db_summary_file}")
    
    # Top pathways per subtype
    for subtype in integration_df['subtype'].unique():
        subtype_pathways = integration_df[integration_df['subtype'] == subtype].copy()
        subtype_pathways = subtype_pathways.sort_values('subtype_genes_in_pathway', ascending=False)
        
        top_file = output_dir / f"top_pathways_{subtype}.csv"
        subtype_pathways.head(20).to_csv(top_file, index=False)
        logger.info(f"  Saved: {top_file}")
    
    return {
        'integration_file': integration_file,
        'summary_by_subtype': summary_file,
        'summary_by_database': db_summary_file
    }

def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("INTEGRATING ION-CCI PATHWAYS WITH MOLECULAR SUBTYPES")
    logger.info("=" * 70)
    
    # Load IoN-CCI DE results
    de_dir = project_root / "results" / "aim3" / "ion_cci" / "de_analysis"
    ion_cci_de = load_ion_cci_de_results(de_dir)
    
    # Load pathway enrichment (from de_review directory where they were saved)
    pathway_dir = project_root / "results" / "aim3" / "ion_cci" / "de_review"
    pathways = load_pathway_enrichment(pathway_dir)
    
    if len(pathways) == 0:
        logger.warning("No pathway enrichment results found. Running pathway enrichment first...")
        # Could trigger pathway enrichment here if needed
        return False
    
    # Load molecular subtypes
    subtypes = load_molecular_subtypes()
    if subtypes is None:
        logger.error("Molecular subtypes not found. Please run molecular subtyping first.")
        return False
    
    # Load subtype DE results
    subtype_de = load_subtype_de_results()
    
    if len(subtype_de) == 0:
        logger.error("Subtype DE results not found. Please run subtype characterization first.")
        return False
    
    # Map pathways to subtypes
    integration_df = map_pathways_to_subtypes(ion_cci_de, subtype_de, pathways)
    
    if len(integration_df) == 0:
        logger.warning("No pathway-subtype associations found.")
        return False
    
    # Create integration summary
    output_dir = project_root / "results" / "aim_integration" / "ion_cci_subtypes"
    summary_files = create_integration_summary(integration_df, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ INTEGRATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"\nSummary:")
    logger.info(f"  Total pathway-subtype associations: {len(integration_df)}")
    logger.info(f"  Subtypes: {integration_df['subtype'].nunique()}")
    logger.info(f"  Pathways: {integration_df['pathway'].nunique()}")
    logger.info(f"  Databases: {integration_df['pathway_database'].nunique()}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


