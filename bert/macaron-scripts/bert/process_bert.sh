#!/usr/bin/env bash

git clone https://github.com/kevinboone/epub2txt
cd epub2txt
make
cd ..

# Crawl your own book_corpus data and put them at book_corpus/
echo 'BookCorpus(epub)'
rm book_corpus/data/English/instructors-manual-identifeye-worskhop.epub
find book_corpus/data/American book_corpus/data/British book_corpus/data/English -type f -name "*.epub" -print0 | \
xargs -0 epub2txt/epub2txt > book_corpus_epub.txt

echo 'BookCorpus(txt)'
find book_corpus/data/American book_corpus/data/British book_corpus/data/English -type f -name "*.txt" -print0 | \
xargs -0 cat > book_corpus_txt.txt

echo 'Wikipedia'
wget -t 0 -c -T 20 https://dumps.wikimedia.org/enwiki/20190220/enwiki-20190220-pages-articles.xml.bz2
python WikiExtractor.py enwiki-20190220-pages-articles.xml.bz2 -b 30G -q -o - > enwiki.txt

cat enwiki.txt book_corpus_txt.txt book_corpus_epub.txt | \
python ../common/remove_non_utf8_chars.py | \
python ../common/precleanup_english.py | \
perl ../common/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en | \
perl ../common/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
python filter_and_cleanup_lines.py > old_corpus.cleaned.txt

python split.py old_corpus.cleaned.txt old_corpus 13088055

cat old_corpus.valid.txt | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en | \
gawk '{print tolower($0);}' > old_corpus.valid.tok

for i in 0 1 2 3
do
cat old_corpus.train.txt.${i} | \
python segment_sentence.py | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 1 -no-escape -l en | \
gawk '{print tolower($0);}' > old_corpus.train.tok.${i}
done

rm corpus.train.tok ||:
for i in 0 1 2 3; do cat old_corpus.train.tok.${i} >> corpus.train.tok; done

cat old_corpus.valid.tok > corpus.valid.tok

../common/fastBPE/fast learnbpe 32640 corpus.train.tok > bpe-code

cat corpus.train.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > corpus.train.tok.tmp
../common/fastBPE/fast applybpe corpus.train.tok.bpe corpus.train.tok.tmp bpe-code
rm corpus.train.tok.tmp

cat corpus.valid.tok | \
python concat_short_sentences.py | \
python ../common/length_filter_by_char.py 20 1000000 > corpus.valid.tok.tmp
../common/fastBPE/fast applybpe corpus.valid.tok.bpe corpus.valid.tok.tmp bpe-code
rm corpus.valid.tok.tmp

cd ../..
python preprocess.py --only-source --workers 16 --nwordssrc 32768 \
--trainpref macaron-scripts/bert/corpus.train.tok.bpe \
--validpref macaron-scripts/bert/corpus.valid.tok.bpe \
--destdir data-bin/bert_corpus

cp macaron-scripts/bert/bpe-code data-bin/bert_corpus/
