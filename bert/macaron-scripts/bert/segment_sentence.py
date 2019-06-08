import re
import sys
from multiprocessing import Pool

import spacy

spacy.require_gpu()
nlp = None


def init():
    global nlp
    nlp = spacy.load('en', disable=['tagger', 'ner', 'textcat'])


def segment(line):
    global nlp
    return ''.join([str(sent) + '\n'
                    for sent in nlp(line).sents
                    if not re.match(r'^\W+$', str(sent))])


def main():
    with Pool(4, initializer=init) as pool:
        for text in pool.imap(segment, sys.stdin, chunksize=128):
            sys.stdout.write(text)


if __name__ == '__main__':
    main()
