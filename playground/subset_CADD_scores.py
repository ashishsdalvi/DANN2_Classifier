import pandas as pd
import duckdb
import tempfile
import pyarrow.parquet as pq 

import pandas as pd
import duckdb
import tempfile
import os

def fast_duckdb_cadd_merge_memsafe(cadd_tsv_path, variant_csv_path):
    # Load and clean df_variants in memory
    df = pd.read_csv(variant_csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    # Construct unique variant string
    df['variant'] = (
        df['chrom'].astype(str) + ':' +
        df['pos'].astype(str) + ':' +
        df['ref'] + '>' + df['alt']
    )

    # Optional safety: sanitize strings for CSV
    df = df.astype(str).replace({r'[\n\r\"]': ''}, regex=True)

    # Save to temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as tmp:
        temp_csv_path = tmp.name
        df.to_csv(temp_csv_path, index=False)

    # Define DuckDB join query
    con = duckdb.connect()
    query = f"""
    SELECT 
        v.chrom, v.pos, v.ref, v.alt, v.variant_label,
        c.raw_score, c.phred_score,
        (v.chrom || ':' || v.pos::VARCHAR || ':' || v.ref || '>' || v.alt) AS variant
    FROM read_csv('{temp_csv_path}',
                  delim=',',
                  header=True,
                  quote='"',
                  escape='"',
                  ignore_errors=true,
                  strict_mode=false
    ) v
    JOIN read_csv('{cadd_tsv_path}',
                  delim='\\t',
                  header=False,
                  columns={{
                      'chrom': 'VARCHAR',
                      'pos': 'INT',
                      'ref': 'VARCHAR',
                      'alt': 'VARCHAR',
                      'raw_score': 'FLOAT',
                      'phred_score': 'FLOAT'
                  }},
                  ignore_errors=true,
                  strict_mode=false
    ) c
    ON replace(v.chrom, 'chr', '') = c.chrom
       AND v.pos = c.pos
       AND v.ref = c.ref
       AND v.alt = c.alt
    """
    result = con.execute(query).df()

    # Reconstruct variant column for alignment
    result['variant'] = (
        result['chrom'].astype(str) + ':' +
        result['pos'].astype(str) + ':' +
        result['ref'] + '>' + result['alt']
    )

    # Align result to original input variant order
    result = result.set_index('variant').loc[df['variant']].reset_index()

    # Clean up temp file
    os.remove(temp_csv_path)

    return result




chr_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

# chr_list = ['chr22'] # testing 

for chr in chr_list:
    tsv_path = f'/media/walt/asdalvi/resources/CADD_v7_scores/{chr}_variants.tsv'
    csv_path = f'/media/walt/asdalvi/results/df_variants/df_variant_{chr}.csv'
    merged = fast_duckdb_cadd_merge_memsafe(tsv_path, csv_path)

    output_path = f'/media/walt/asdalvi/resources/CADD_v7_subsetted/CADD_scores_{chr}.parquet'
    merged.to_parquet(output_path, index=False)