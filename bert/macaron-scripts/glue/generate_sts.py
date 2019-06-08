import argparse
import csv
import os

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="path of data")
    parser.add_argument("--output", type=str, required=True,
                        help="path of output")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not os.path.isdir(args.output):
        raise FileExistsError(f"{args.output} is not a directory")
    labels = []
    with open(os.path.join(args.data, "train.tsv"), "r", encoding="utf-8") as fi, open(
            os.path.join(args.output, "train.txt"), "w", encoding="utf-8") as fo:
        reader = csv.reader(fi, delimiter="\t", quotechar=None)
        for line in reader:
            if line[7] == "sentence1" and line[8] == "sentence2" and line[9] == 'score':
                continue
            fo.write(f'{line[7]}\n{line[8]}\n')
            labels.append(0.5 *(float(line[9]) - 3))
    torch.save(torch.FloatTensor(labels), os.path.join(args.output, "train_labels.pt"))
    labels = []
    with open(os.path.join(args.output, "valid.txt"), "w", encoding="utf-8") as fo:
        with open(os.path.join(args.data, "dev.tsv"), "r", encoding="utf-8") as fi:
            reader = csv.reader(fi, delimiter="\t", quotechar=None)
            for line in reader:
                if line[7] == "sentence1" and line[8] == "sentence2" and line[9] == 'score':
                    continue
                fo.write(f'{line[7]}\n{line[8]}\n')
                labels.append(0.5 * (float(line[9]) - 3))
    torch.save(torch.FloatTensor(labels), os.path.join(args.output, "valid_labels.pt"))
    with open(os.path.join(args.output, "test.txt"), "w", encoding="utf-8") as fo:
        with open(os.path.join(args.data, "test.tsv"), "r", encoding="utf-8") as fi:
            reader = csv.reader(fi, delimiter="\t", quotechar=None)
            for line in reader:
                if line[7] == "sentence1" and line[8] == "sentence2":
                    continue
                fo.write(f'{line[7]}\n{line[8]}\n')


if __name__ == '__main__':
    main()
