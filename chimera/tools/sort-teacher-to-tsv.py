import argparse
import csv
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-file", type=str, required=True)
parser.add_argument("--original-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=True)
args = parser.parse_args()

with open(args.teacher_file, 'r') as f:
    data = [line for line in csv.reader(f, delimiter="\t")]

data = [[int(line[0][2:]), line[-1]] for line in data]
data.sort(key=lambda line: int(line[0]))

with open(args.original_file, 'r') as f:
    original = [line.rstrip("\n").split('\t') for line in f.readlines()]

title = original[0]
original = original[1:]
new_data = deepcopy(original)
for i, line in enumerate(new_data):
    line[-3] = data[i][-1]

output = [title] + [
    _list[i]
    for i in range(len(new_data))
    for _list in (original, new_data)
]

with open(args.output_file, 'w') as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerows(output)
