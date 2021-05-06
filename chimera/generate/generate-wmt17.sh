#!/bin/bash
CUDA_VISIBLE_DEVICES= fairseq-generate $WMT_ROOT/bin \
    --path $1 --max-tokens 300 \
    --beam 5 --remove-bpe
