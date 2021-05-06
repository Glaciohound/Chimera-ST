#!/bin/bash
fairseq-generate $WMT_ROOT/bin \
    --path $1 --max-tokens 400 \
    --beam 5 --remove-bpe --gen-subset test1
