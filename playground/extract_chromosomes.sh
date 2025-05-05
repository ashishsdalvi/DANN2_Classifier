#!/bin/bash

# Set paths
INPUT_FILE="/media/walt/asdalvi/resources/CADD_v7_scores/whole_genome_SNVs.tsv.gz"
OUTPUT_DIR="/media/walt/asdalvi/resources/CADD_v7_scores"

# Chromosomes to process (adjust as needed)
CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 X) # already did chr22 and Y 

# Loop through chromosomes
for CHR in "${CHROMOSOMES[@]}"; do
    echo "Extracting chr$CHR..."
    tabix "$INPUT_FILE" "$CHR" > "$OUTPUT_DIR/chr${CHR}_variants.tsv"
done

echo "✅ Extraction complete."
