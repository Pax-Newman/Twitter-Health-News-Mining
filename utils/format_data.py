import csv
import os
import sys

# Find data files

data_path = sys.argv[1]
target_path = sys.argv[2]

datasets = os.listdir(data_path)

with open(target_path, 'w') as target:
    writer = csv.writer(target)

    headers = ['id', 'datetime', 'publisher', 'content']
    writer.writerow(headers)

    for dataset in datasets:
        path = data_path + dataset
    
        # Merge file into csv
        with open(path, encoding='unicode_escape') as d:
            reader = csv.reader(d, delimiter='|')

            # Original ordering:
            # id, datetime, content

            # New ordering:
            # id, datetime, publisher, content

            for row in reader:
                if row:
                    writer.writerow(row[0:2] + [dataset.split('.')[0]] + [row[2]])

