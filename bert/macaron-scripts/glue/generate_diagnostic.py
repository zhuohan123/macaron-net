import argparse
import csv
import os

_label_to_id = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2
}


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
    with open(os.path.join(args.output, "test.txt"), "w", encoding="utf-8") as fo:
        with open(os.path.join(args.data, "diagnostic.tsv"), "r", encoding="utf-8") as fi:
            reader = csv.reader(fi, delimiter="\t", quotechar=None)
            for line in reader:
                if line[1] == "sentence1" and line[2] == "sentence2":
                    continue
                fo.write(f'{line[1]}\n{line[2]}\n')


if __name__ == '__main__':
    main()
