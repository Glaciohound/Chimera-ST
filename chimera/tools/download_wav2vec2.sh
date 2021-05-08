pretrained_ckpt=$1
WAVE2VEC_DIR=$2
external_url=http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera
if [[ ! -f $WAVE2VEC_DIR/$pretrained_ckpt ]]; then
    wget -P $WAVE2VEC_DIR $external_url/$pretrained_ckpt
fi
