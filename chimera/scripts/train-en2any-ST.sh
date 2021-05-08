#!/bin/bash

# prerequisits and environment variables
export ST_SAVE_DIR="$SAVE_ROOT/st"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$ST_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/wave2vec"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $ASR_SAVE_DIR $WAVE2VEC_DIR $WMT_ROOT $MUSTC_ROOT $LS_ROOT
reset_optimizer="--reset-optimizer"

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# loading MT pre-trained ckpt
if [[ $resume == "True" ]]; then
    reset_optimizer=""
else
    cp $MT_SAVE_DIR/checkpoint_best.pt $ST_SAVE_DIR/checkpoint_last.pt
fi

# Auto-evaluating
CUDA_VISIBLE_DEVICES= python3 chimera/generate/auto-generate.py \
    --dirname ${SAVE_DIR} \
    --generate_script \
        chimera/generate/generate-mustc.sh \
    --silent &
generate_pid=$!
SUICIDE_CODE='chimera/tools/auto-generate-suicide.code'

# Train on MuST-C data
fairseq-train ${MUSTC_ROOT}/en-$target \
    --task triplet \
    --train-subset train_wave --valid-subset dev_wave \
    --max-tokens 2000000 --max-source-positions 2000000 \
    --save-dir $SAVE_DIR \
    --config-yaml config_wave.yaml \
    \
    --criterion triplet_st_mt_contrastive --label-smoothing 0.1 \
    \
    --arch s2t_transformer_w2v2_interlingua_base --share-decoder-input-output-embed \
    --w2v2-model-path $WAVE2VEC_DIR/$pretrained_ckpt \
    --encoder-layers 6 --encoder-embed-dim 512 \
    --interlingua-length 64 \
    --dropout 0.1 \
    \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --max-update $max_updates --warmup-updates 4000 \
    --fp16 \
    $reset_optimizer \
    \
    --update-freq $(expr 8 / $num_gpus) --num-workers 1 \
    --ddp-backend no_c10d \
    --best-checkpoint-metric st_loss \
    --seed $seed

# wait for evaluation process to finish
touch $SUICIDE_CODE
tail --pid=$generate_pid -f /dev/null
