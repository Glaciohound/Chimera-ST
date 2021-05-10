#!/bin/bash

# prerequisits and environment variables
export MUSTC_ROOT="speech_data/mustc"
export WAVE2VEC_DIR="$SAVE_ROOT/wave2vec"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $WAVE2VEC_DIR $MUSTC_ROOT

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# making a fake config file
python3 chimera/tools/hand-make-config.py \
    --data-dir $MUSTC_ROOT/en-$target
cp chimera/resources/$dataset-en-$target-spm/* $MUSTC_ROOT/en-$target

# Train on MuST-C data
fairseq-interactive ${MUSTC_ROOT}/en-$target \
    --task triplet \
    --config-yaml config_wave.yaml \
    --max-tokens 2000000 --max-source-positions 2000000 \
    --path $checkpoint
