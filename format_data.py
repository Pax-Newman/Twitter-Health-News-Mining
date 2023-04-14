import csv
import os
import chardet

# Find data files
datasets = os.listdir('data')

target_csv = 'dataset.csv'
with open(target_csv, 'w') as target:
    writer = csv.writer(target)

    headers = ['id', 'datetime', 'publisher', 'content']
    writer.writerow(headers)

    for dataset in datasets:
        path = 'data/' + dataset

        # Determine the encoding of the file
        with open(path, 'rb', buffering=0) as d:
            encoding = chardet.detect(d.readall())['encoding']
            print(f'\n{dataset} -- {encoding}\n')
    
        # Merge file into csv
        with open(path, encoding=encoding) as d:
            reader = csv.reader(d, delimiter='|')

            # Original ordering:
            # id, datetime, content

            # New ordering:
            # id, datetime, publisher, content

            for row in reader:
                if row:
                    writer.writerow(row[0:2] + [dataset.split('.')[0]] + [row[2]])

