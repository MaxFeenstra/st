import pandas as pd 
import numpy as np 
import glob
from tqdm import tqdm
import os

# lines_discard = 3
lines_discard = 7

fp_txts = glob.glob('**/test_output_tcga/*.txt', recursive=True)

fp_labels = '../her2st_processing/processed_expr.csv'
df_labels = pd.read_csv(fp_labels)
gene_names = df_labels.columns.to_list()
gene_names = gene_names[1:]
print('Num genes %d' %len(gene_names))

for fp in fp_txts:
    print(fp)

    fp_output = fp.replace('.txt','.csv')
    if os.path.exists(fp_output):
        os.remove(fp_output)

    # Read the txt file and discard last 3 lines
    with open(fp, 'r') as f:
        file_lines = f.readlines()
        # file_lines = file_lines[:-lines_discard]   
        
        # Get list of patch IDs 
        patches = [line.split(' ')[0] for line in file_lines]
        patches = list(set(patches))

        df = pd.DataFrame(index=patches, columns=gene_names)

        # ['H3_16x12', '0', 'gt:', '1.0', 'predict:', '0.18543800711631775\n']   
        for fl in tqdm(file_lines):
            parts = fl.split(' ')
            predicted = float(parts[-1])
            patch_name = parts[0]
            gene_idx = int(parts[1])
            gene_name = gene_names[gene_idx]

            df.loc[patch_name, gene_name] = predicted

        df = df.sort_index()
        df.to_csv(fp_output)
