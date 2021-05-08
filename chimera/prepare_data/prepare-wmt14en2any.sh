#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

export version="wmt17"
export devset="split-train"
export target=de
export subword=bpe
export subword_tokens=40000
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-dir) export DATA_ROOT="$2"; shift ;;
        --external) export external="$2"; shift ;;
        --subword) export subword="$2"; shift ;;
        --subword-tokens) export subword_tokens="$2"; shift ;;
        --target) export target="$2"; shift ;;
        --icml17) export version="wmt14" ;;
        --wmt14) export version="wmt14" ;;
        --wmt17) export version="wmt17" ;;
        --original-dev) export devset="original" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "DATA_ROOT: $DATA_ROOT"
echo "version: $version"
echo "dev set: $devset"
echo "target language: $target"
echo "subword: $subword"
echo

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=$subword_tokens
SCRIPTPATH=$(pwd)
mkdir -p $DATA_ROOT
cd $DATA_ROOT

if [[ $target == "de" ]]; then
    CORPORA=(
        "training/europarl-v7.de-en"
        "commoncrawl.de-en"
        "training/news-commentary-v12.de-en"
    )
elif [[ $target == "fr" ]]; then
    CORPORA=(
        "training/europarl-v7.fr-en"
        "commoncrawl.fr-en"
        "training/news-commentary-v12.fr-en"
    )
elif [[ $target == "ru" ]]; then
    CORPORA=(
        "null"
        "commoncrawl.ru-en"
        "training/news-commentary-v12.ru-en"
    )
elif [[ $target == "es" ]]; then
    CORPORA=(
        "training/europarl-v7.es-en"
        "commoncrawl.es-en"
        "null"
    )
else
    echo "target language not supported"
    exit 1
fi

OUTDIR=${version}_en_$target
if [[ $external != "" ]]; then
    CORPORA+=("external-$external.$target-en")
    cache_name_tail+="-ext$external"
fi
if [[ $subword != "bpe" ]]; then
    cache_name_tail+="-$subword"
fi


if [[ ! -d "$SCRIPTS" ]]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi
src=en
tgt=$target
lang=$src-$tgt
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013
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
            echo "containing: $orig/$f.$l"
            cat $orig/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 30 -a -l $l >> $train_file
        fi
    done
done

if [[ $devset == "original" ]]; then
    echo "pre-processing original newstest2013 data..."
    for l in $src $tgt; do
        if [[ -f $tmp/valid.tags.$lang.tok.$l ]]; then
            rm $tmp/valid.tags.$lang.tok.$l
        fi
        cat $orig/$dev.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 30 -a -l $l >> $tmp/valid.tags.$lang.tok.$l
    done
fi

echo "pre-processing test data..."
for l in $src $tgt; do
    if [[ "$l" == "$src" ]]; then
        t="src"
    else
        t="ref"
    fi
    if [[ $tgt != "es" ]]; then
        test_file=$orig/test-full/newstest2014-${tgt}en-$t.$l.sgm
    else
        test_file=$orig/dev/newstest2012-$t.$l.sgm
    fi
    grep '<seg id' $test_file | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 30 -a -l $l > $tmp/test.$l
    echo ""
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
    echo "using original newstest2013 as valid set"
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

    # applying SPM algorithm
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
