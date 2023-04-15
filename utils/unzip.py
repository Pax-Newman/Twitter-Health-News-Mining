import zipfile
import sys

target = sys.argv[1]
output = sys.argv[2]

with zipfile.ZipFile(target, 'r') as z:
    z.extractall(path=output)
