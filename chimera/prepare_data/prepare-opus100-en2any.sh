#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

export start_from_scratch="true"
export devset="split-train"
export subword="bpe"
export subword_tokens=40000
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) export DATA_ROOT="$2"; shift ;;
        --target) export target="$2"; shift ;;
        --original-dev) export devset="original";;
        --external) export external="$2"; shift ;;
        --subword) export subword="$2"; shift ;;
        --subword-tokens) export subword_tokens="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "DATA_ROOT: $DATA_ROOT"
echo "dev set: $devset"
echo "target language: $target"
echo "subword: $subword"
echo "subword-tokens: $subword_tokens"
echo

SCRIPTPATH=$(pwd)
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=$subword_tokens
src=en
tgt=$target
version=opus100
OUTDIR=${version}_${src}_${tgt}
mkdir -p $DATA_ROOT
cd $DATA_ROOT


# cloning github repository
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

CORPORA=(\
    "opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-train"
)
DEV_CORPORA="opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-dev"
TEST_CORPORA="opus-100-corpus/v1.0/supervised/$src-$tgt/opus.${src}-${tgt}-test"


# processing from scratch
if [[ ! -d "$SCRIPTS" ]]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi
lang=$src-$tgt
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=$DEV_CORPORA
mkdir -p $orig $tmp $prep

if [[ $external == 'mustc' ]]; then
    cd $SCRIPTPATH
    bash chimera/prepare_data/append-mustc-to-wmt.sh $src $tgt
    cd $DATA_ROOT
fi

echo "pre-processing train data..."
for l in $src $tgt; do
    train_file=$tmp/train.tags.$lang.tok.$l
    if [[ -f $train_file ]]; then
        rm $train_file
    fi
    for f in "${CORPORA[@]}"; do
        if [[ $f != "null" ]]; then
            train_raw=$orig/$f.$l
            echo "containing: $train_raw"
            cat $orig/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 30 -a -l $l >> $train_file
        fi
    done
done

echo "pre-processing dev data..."
if [[ $devset == "original" ]]; then
    for l in $src $tgt; do
        dev_file=$tmp/valid.tags.$lang.tok.$l
        dev_raw=$orig/$dev.$l
        echo "containing: $dev_raw"
        cat $orig/$dev.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 30 -a -l $l > $dev_file
    done
fi

echo "pre-processing test data..."
for l in $src $tgt; do
    test_file=$tmp/test.$l
    test_raw=$orig/$TEST_CORPORA.$l
    echo "containing $test_raw"
    cat $orig/$TEST_CORPORA.$l | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 30 -a -l $l > $test_file
done
if [[ $external == 'mustc' ]]; then
    test_set=tst-COMMON
    file_head="$SCRIPTPATH/$MUSTC_ROOT/$src-$tgt/data/$test_set/txt/$test_set"
    echo "pre-processing MUST-C test data..."
    for l in $src $tgt; do
        cat $file_head.$l | \
            sed -e "s/\’/\'/g" | \
            perl $TOKENIZER -threads 30 -a -l $l > "$tmp/mustc-$test_set.$l"
        echo ""
    done
fi

wc -l $tmp/*

if [[ $devset == "split-train" ]]; then
    echo "splitting train and valid..."
    for l in $src $tgt; do
        awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
        awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
    done
else
    echo "using original valid set"
    for l in $src $tgt; do
        cp $tmp/valid.tags.$lang.tok.$l $tmp/valid.$l
        cp $tmp/train.tags.$lang.tok.$l $tmp/train.$l
    done
fi

# learning BPE
TRAIN=$tmp/train.${tgt}-en
if [[ -f $TRAIN ]]; then
    rm -f $TRAIN
fi
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done
if [[ $subword == "bpe" ]]; then
    BPE_CODE=$prep/code
    echo "learn_bpe.py on ${TRAIN}..."
    python3 $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    # applying BPE
    for L in $src $tgt; do
        for f in train.$L valid.$L test.$L; do
            echo "apply_bpe.py to ${f}..."
            python3 $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
        done
    done

    if [[ $external == 'mustc' ]]; then
        cd $SCRIPTPATH
        bash $SCRIPTPATH/chimera/prepare_data/apply-bpe-to-mustc.sh \
            --code-dir $DATA_ROOT/$OUTDIR --tokenizer $DATA_ROOT/$TOKENIZER \
            --test-set tst-COMMON --lang-pair $src $tgt
        cd $DATA_ROOT
    fi

    # cleaning train and valid, moving all to $prep
    perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
    perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
    for L in $src $tgt; do
        cp $tmp/bpe.test.$L $prep/test.$L
    done

elif [[ $subword == "spm" ]]; then
    spm=$prep/spm
    mkdir -p $spm
    BPE_CODE=$spm/spm_unigram10000_wave_joint

    # learning spm or copying an existing one
    resource_spm=chimera/resources/$version-$src-$tgt-spm
    if [[ -d $SCRIPTPATH/$resource_spm ]]; then
        echo "Existing spm dictionary $resource_spm detected. Copying..."
        cp $SCRIPTPATH/$resource_spm/* $spm/
    else
        echo "No existing spm detected. Learning unigram spm on $TRAIN ..."
        python3 $SCRIPTPATH/chimera/prepare_data/learn_spm.py \
            --input $prep/tmp/train.$target-en \
            --vocab-size 10000 \
            --model-prefix $BPE_CODE
    fi

    # applying BPE
    for L in $src $tgt; do
        for f in train.$L valid.$L test.$L; do
            echo "apply_spm.py to ${f}..."
            python3 $SCRIPTPATH/chimera/prepare_data/apply_spm.py \
                --input-file $tmp/$f --output-file $tmp/spm.$f \
                --model $BPE_CODE.model
        done
    done

    if [[ $external == 'mustc' ]]; then
        # also applying BPE on MUSC-C test set
        for lang in $src $tgt; do
            infile="$tmp/mustc-$test_set.$lang"
            outfile="$prep/mustc-$test_set.$lang"
            echo "apply_spm.py to $infile, saving to $outfile..."
            python3 $SCRIPTPATH/chimera/prepare_data/apply_spm.py \
                --input-file $infile --output-file $outfile \
                --model $BPE_CODE.model
        done
    fi

    # cleaning train and valid, moving all to $prep
    perl $CLEAN -ratio 1.5 $tmp/spm.train $src $tgt $prep/train 1 250
    perl $CLEAN -ratio 1.5 $tmp/spm.valid $src $tgt $prep/valid 1 250
    for L in $src $tgt; do
        cp $tmp/spm.test.$L $prep/test.$L
    done

else
    echo "unsupported subword algorithm $subword"
    exit 1
fi
