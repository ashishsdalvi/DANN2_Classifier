import pyarrow.parquet as pq 
import numpy as np
import pandas as pd 

import pyarrow.parquet as pq
import numpy as np




def convert_parquet_to_npy(parquet_path, npy_path, chunk_size=100000):
# Open the Parquet file as a dataset
    dataset = pq.ParquetFile(parquet_path)
    columns = [name for name in dataset.schema.names if name != "variant"]

    # Determine the total number of rows and columns
    total_rows = sum(dataset.metadata.row_group(i).num_rows for i in range(dataset.num_row_groups))
    num_columns = len(columns)

    # Create a memory-mapped file for writing
    arr = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float32, shape=(total_rows, num_columns))

    start_idx = 0

    # Iterate over row groups and write in chunks
    for row_group in range(dataset.num_row_groups):
        # Read a row group as a table
        table = dataset.read_row_group(row_group, columns=columns)

        # Convert to numpy arrays
        arrays = [table.column(i).to_numpy(zero_copy_only=False) for i in range(table.num_columns)]

        # Stack columns to form a matrix (rows x columns)
        chunk_data = np.column_stack(arrays).astype(np.float32)
        end_idx = start_idx + chunk_data.shape[0]

        # Write to memory-mapped file
        arr[start_idx:end_idx, :] = chunk_data

        print(f"Processed row group {row_group + 1}/{dataset.num_row_groups}, Rows: {start_idx} - {end_idx}")

        # Update start index for the next chunk
        start_idx = end_idx

    print(f"Conversion complete. NPY saved at {npy_path}")


    

# chr_list = ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 
#  'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
#  'chr20', 'chr21', 'chrX'] # no chr22/chrY cause already converted

chr_list = ['chr1']

# chr_list = ['chr22', 'chrY'] # testing 

for chr in chr_list:
    print('Converting', chr, flush=True)
    # convert CADD parquet to npy
    # CADD_parquet_path = f'/media/walt/asdalvi/resources/CADD_v7_subsetted/CADD_scores_{chr}.parquet'
    # CADD_npy_path = f'/media/walt/asdalvi/resources/CADD_v7_subsetted/CADD_scores_{chr}.npy'
    # convert_parquet_to_npy(CADD_parquet_path, CADD_npy_path, mode='CADD')

    # convert preds parquet to npy 
    preds_parquet_path = f'/media/walt/asdalvi/results/predictions/{chr}_l2.parquet'
    preds_npy_path = f'/media/walt/asdalvi/results/predictions/npy_preds/{chr}_l2.npy'
    convert_parquet_to_npy(preds_parquet_path, preds_npy_path)