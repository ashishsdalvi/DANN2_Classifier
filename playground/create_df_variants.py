import pandas as pd 


def create_model_input_df(variant_df):
    # returns the input for forzoi (a dataframe) and set of labels for common variants (0) and singletons (1)
    # common is >1% and rare is <0.5% in pop. 
    chrom_list, pos_list, ref_list, alt_list = [], [], [], []
    variant_labels = []

    for i, row in variant_df.iterrows():
        af = row['AF_var']
        ac = row['AC_var']
        chrom, pos, ref, alt = row['chrom_var'], row['end_var'], row['ref_var'], row['alt_var']
        # if common variant
        if af > 0.01 and ac > 1:
            chrom_list.append(chrom)
            pos_list.append(pos)
            ref_list.append(ref)
            alt_list.append(alt)
            variant_labels.append(0) # 0 is common
            variant_label = 0
        # if singleton 
        elif ac == 1:
            chrom_list.append(chrom)
            pos_list.append(pos)
            ref_list.append(ref)
            alt_list.append(alt)
            variant_labels.append(1) # 1 is singleton
            variant_label = 1 


    df_variants = pd.DataFrame({
        'chrom': chrom_list,
        'pos': pos_list,
        'ref': ref_list,
        'alt': alt_list,
        'variant_label': variant_label
    })

    return variant_labels, df_variants



# chromosome_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", 
#  "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", 
#  "chr20", "chr21", "chr22", "chrX", "chrY"]


chromosome_list = ["chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", 
 "chr20", "chr21", "chr22", "chrX", "chrY"]

# chromosome_list = ['chr22']


for chr in chromosome_list:
    csv_path = f"/media/walt/asdalvi/results/subset/genome_{chr}_ccres.csv"
    print('Reading', csv_path)
    chr_df = pd.read_csv(csv_path)
    print('Creating model input df for', chr)
    variant_labels, df_variants = create_model_input_df(chr_df)
    print('Writing', chr)
    df_variants.to_csv(f"/media/walt/asdalvi/results/df_variants/df_variant_{chr}.csv")