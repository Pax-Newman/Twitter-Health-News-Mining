import os
import sys

import re
import pandas as pd

# Find data files

data_path = sys.argv[1]
target_path = sys.argv[2]

datasets = os.listdir(data_path)

headers = ['id', 'datetime', 'publisher', 'content']

df = pd.DataFrame(columns=headers)

def yield_line(file):
    i=0
    for line in file:
        print(f'{i = }')
        i += 1
        yield line

df_rows = []

for dataset in datasets:
    print(f'Merging {dataset}...')
    path = data_path + dataset
    with open(path, encoding='unicode escape') as d:
        lines = (line for line in d.read().split('\n') if not (line == '' or line.isspace()))
        for line in lines:
            #print(dataset)
            delim_idxs = [match.start() for match in re.finditer(r'\|', line)]
            #print(line)
            #print(delim_idxs)
            df_rows += [{
                'id':line[:delim_idxs[0]],                    # From the beginning up to index of the index of the first delimiter
                'datetime':line[delim_idxs[0]+1:delim_idxs[1]], # From the first delimiter to the second (excluding both)
                'publisher':dataset.split('.')[0],          # The filename of the dataset without its extension
                'content':line[delim_idxs[1]+1:]              # The rest of the line after the second delimiter
                }]

# Merge rows into a single dataframe and export to a pickle
df = pd.concat([pd.DataFrame(row, index=[i]) for i, row in enumerate(df_rows)])

df.to_pickle(target_path)


