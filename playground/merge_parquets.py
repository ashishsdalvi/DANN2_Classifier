import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Point to the directory containing the Parquet files
dataset = ds.dataset("/media/walt/asdalvi/results/predictions/texera/chr1", format="parquet")

print('Reading dataset')

# Read the dataset (this is lazy, efficient, and scalable)
table = dataset.to_table()
df = table.to_pandas()  # Be cautious if it's very large

print('Writing table')

# Save as a single file
pq.write_table(table, "/media/walt/asdalvi/results/predictions/texera/chr1/chr1_texera_merged.parquet")


