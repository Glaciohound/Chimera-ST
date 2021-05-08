#!/bin/bash

echo "applying bpe on other test sets"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --code-dir) export TEXT="$2"; shift;;
        --test-set) export test_set="$2"; shift;;
        --lang-pair) export lang1="$2"; export lang2="$3"; shift; shift;;
        --tokenizer) export TOKENIZER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

file_head="$MUSTC_ROOT/$lang1-$lang2-text/data/$test_set/txt/$test_set"
tmp=$TEXT/tmp

echo "pre-processing MUST-C test data..."
for lang in $lang1 $lang2; do
    cat $file_head.$lang | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 30 -a -l $lang > "$tmp/mustc-$test_set.$lang"
    echo ""
done

# also applying BPE on MUSC-C test set
for lang in $lang1 $lang2; do
    file="$tmp/mustc-$test_set.$lang"
    outfile="$TEXT/mustc-$test_set.$lang"
    echo "apply_bpe.py to $file, saving to $outfile..."
    python3 $WMT_ROOT/subword-nmt/subword_nmt/apply_bpe.py \
        -c $TEXT/code \
        < $file \
        > $outfile
done
