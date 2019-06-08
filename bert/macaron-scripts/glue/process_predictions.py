import argparse
import os

rte_labels = ['not_entailment', 'entailment']
mnli_labels = ['neutral', 'entailment', 'contradiction']
qnli_labels = ['not_entailment', 'entailment']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='input path of predictions')
    parser.add_argument('--output', type=str,
                        help='output path of submissions')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not os.path.isdir(args.output):
        raise FileExistsError(f'{args.output} is not a directory')

    with open(os.path.join(args.output, 'CoLA.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_CoLA.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{line.strip()}\n')
            cnt += 1

    with open(os.path.join(args.output, 'MRPC.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_MRPC.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{line.strip()}\n')
            cnt += 1

    with open(os.path.join(args.output, 'STS-B.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_STS-B.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{float(line.strip()) * 2.0 + 3.0}\n')
            cnt += 1

    with open(os.path.join(args.output, 'RTE.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_RTE.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{rte_labels[int(line.strip())]}\n')
            cnt += 1

    with open(os.path.join(args.output, 'MNLI-m.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_MNLI.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{mnli_labels[int(line.strip())]}\n')
            cnt += 1

    with open(os.path.join(args.output, 'MNLI-mm.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_MNLI-mm.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{mnli_labels[int(line.strip())]}\n')
            cnt += 1

    with open(os.path.join(args.output, 'QNLI.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_QNLI.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{qnli_labels[int(line.strip())]}\n')
            cnt += 1

    with open(os.path.join(args.output, 'QQP.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_QQP.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{line.strip()}\n')
            cnt += 1

    with open(os.path.join(args.output, 'SST-2.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_SST-2.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{line.strip()}\n')
            cnt += 1

    with open(os.path.join(args.output, 'AX.tsv'), 'w', encoding='utf-8') as fo, open(os.path.join(args.input, 'prediction_diagnostic.txt'), 'r', encoding='utf-8') as fi:
        fo.write('index\tprediction\n')
        cnt = 0
        for line in fi:
            fo.write(f'{cnt}\t{mnli_labels[int(line.strip())]}\n')
            cnt += 1

    with open(os.path.join(args.output, 'WNLI.tsv'), 'w', encoding='utf-8') as fo:
        fo.write('index\tprediction\n')
        for i in range(146):
            fo.write(f'{i}\t0\n')


if __name__ == '__main__':
    main()