#!/usr/bin/env bash

BPE_CODE_PATH=$1
DICT_PATH=$2

python download_glue_data.py --data_dir glue --tasks CoLA,SST,MRPC,QQP,STS,MNLI,QNLI,RTE,WNLI,diagnostic

python generate_cola.py glue/CoLA --output glue/CoLA
python single_sentence.py glue/SST-2 --output glue/SST-2
python generate_mrpc.py glue/MRPC --output glue/MRPC
python generate_qqp.py glue/QQP --output glue/QQP
python generate_sts.py glue/STS-B --output glue/STS-B
python generate_mnli.py glue/MNLI --output glue/MNLI
python generate_mnli_mm.py glue/MNLI --output glue/MNLI-mm
python generate_qnli.py glue/QNLI --output glue/QNLI
python generate_rte.py glue/RTE --output glue/RTE
python generate_wnli.py glue/WNLI --output glue/WNLI
python generate_diagnostic.py glue/diagnostic --output glue/diagnostic

for TASK in CoLA SST-2 MRPC QQP STS-B MNLI MNLI-mm QNLI RTE WNLI
do
for SPLIT in train valid test
do
cat glue/${TASK}/${SPLIT}.txt | \
python ../common/remove_non_utf8_chars.py | \
python ../common/precleanup_english.py | \
perl ../common/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en | \
perl ../common/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
python align_text.py | \
sed 's/\\/ /g' | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -no-escape -l en | \
gawk '{print tolower($0);}' > ${SPLIT}.tok.tmp
#../common/fastBPE/fast applybpe glue/${TASK}/${SPLIT}.tok.bpe ${SPLIT}.tok.tmp ${BPE_CODE_PATH}
../common/subword-nmt/subword_nmt/apply_bpe.py -c ${BPE_CODE_PATH} < ${SPLIT}.tok.tmp > glue/${TASK}/${SPLIT}.tok.bpe
rm ${SPLIT}.tok.tmp
done
done

cat glue/diagnostic/test.txt | \
python ../common/remove_non_utf8_chars.py | \
python ../common/precleanup_english.py | \
perl ../common/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl en | \
perl ../common/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
python align_text.py | \
sed 's/\\/ /g' | \
../common/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -no-escape -l en | \
gawk '{print tolower($0);}' > test.tok.tmp
#../common/fastBPE/fast applybpe glue/diagnostic/test.tok.bpe test.tok.tmp ${BPE_CODE_PATH}
../common/subword-nmt/subword_nmt/apply_bpe.py -c ${BPE_CODE_PATH} < test.tok.tmp > glue/diagnostic/test.tok.bpe
rm test.tok.tmp

cd ../..

for TASK in CoLA SST-2 MRPC QQP STS-B MNLI MNLI-mm QNLI RTE WNLI
do
python preprocess_bert.py --only-source --workers 8 \
--trainpref macaron-scripts/glue/glue/${TASK}/train.tok.bpe \
--validpref macaron-scripts/glue/glue/${TASK}/valid.tok.bpe \
--testpref macaron-scripts/glue/glue/${TASK}/test.tok.bpe \
--srcdict macaron-scripts/glue/${DICT_PATH} \
--destdir data-bin/glue/${TASK}
cp macaron-scripts/glue/glue/${TASK}/train_labels.pt data-bin/glue/${TASK}/
cp macaron-scripts/glue/glue/${TASK}/valid_labels.pt data-bin/glue/${TASK}/
done

python preprocess_bert.py --only-source --workers 8 \
--testpref macaron-scripts/glue/glue/diagnostic/test.tok.bpe \
--srcdict macaron-scripts/glue/${DICT_PATH} \
--destdir data-bin/glue/diagnostic
