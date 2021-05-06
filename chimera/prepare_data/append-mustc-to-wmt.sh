#!/bin/bash

echo "appending mustc text train+test into wmt dir"

lang1=$1
lang2=$2

# bash chi/experiments/prepare_data/prepare-mustc-any.sh \
#     --task wave --ignore_fbank80 --text-only --target $lang2
# mkdir -p $WMT_ROOT/orig

for lang in $lang1 $lang2; do
    cp $MUSTC_ROOT/$lang1-$lang2/data/train/txt/train.$lang \
        $WMT_ROOT/orig/external-mustc.$lang2-$lang1.$lang
done
