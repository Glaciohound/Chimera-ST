#!/bin/bash

# prerequisits and environment variables
export MUSTC_ROOT="speech_data/mustc"
export ST_SAVE_DIR="$SAVE_ROOT/st"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$ST_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/wave2vec"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $ASR_SAVE_DIR $WAVE2VEC_DIR $WMT_ROOT $MUSTC_ROOT $LS_ROOT
reset_optimizer="--reset-optimizer"

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# making a fake config file
python3 chimera/tools/hand-make-yaml.py \
    --data-dir $MUSTC_ROOT/en-$target
cp chimera/resources/$dataset-en-$target-spm/* $MUSTC_ROOT/en-$target

# Train on MuST-C data
fairseq-interactive ${MUSTC_ROOT}/en-$target \
    --task triplet \
    --config-yaml config_wave.yaml \
    --max-tokens 2000000 --max-source-positions 2000000 \
    --path $checkpoint
