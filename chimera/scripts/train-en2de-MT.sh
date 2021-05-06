#!/bin/bash

# prerequisits and environment variables
export ST_SAVE_DIR="$SAVE_ROOT/st"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$MT_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/wave2vec"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $WAVE2VEC_DIR $MUSTC_ROOT $WMT_ROOT
target=de
dataset=wmt14

# loading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# WMT-MUSTC joint data and spm
TEXT=$WMT_ROOT/${dataset}_en_$target
spm_model=$TEXT/spm/spm_unigram10000_wave_joint.model
spm_dict=$TEXT/spm/spm_unigram10000_wave_joint.txt
fairseq-preprocess \
    --source-lang en --target-lang $target \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --testpref $TEXT/test,$TEXT/mustc-tst-COMMON \
    --destdir $WMT_ROOT/bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict $spm_dict --tgtdict $spm_dict \
    --workers 100

# Auto-evaluating
python3 chimera/generate/auto-generate.py \
    --dirname ${SAVE_DIR} \
    --generate_script \
        chimera/generate/generate-wmt17-1-gpu.sh \
        chimera/generate/generate-wmt17-gpu.sh \
    --haruna_logdir $chi_ckpt_dir \
    --eval_log_suffix mustc wmt \
    --silent &
generate_pid=$!
SUICIDE_CODE='chimera/tools/auto-generate-suicide.code'

# Train on WMT data
fairseq-train $WMT_ROOT/bin \
    --task translation \
    --train-subset train --valid-subset valid \
    --save-dir $SAVE_DIR \
    --max-tokens 4096 \
    \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $spm_model \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    \
    --arch s2t_transformer_w2v2_interlingua_base --share-decoder-input-output-embed \
    --w2v2-model-path $WAVE2VEC_DIR/$pretrained_ckpt \
    --encoder-layers 6 --encoder-embed-dim 512 \
    --interlingua-length 64 \
    --dropout 0.1 \
    \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --max-update 500000 --warmup-updates 4000 \
    --fp16 \
    \
    --update-freq $(expr 8 / $num_gpus) --num-workers 1 \
    --ddp-backend no_c10d \
    --seed $seed

touch $SUICIDE_CODE
tail --pid=$generate_pid -f /dev/null
