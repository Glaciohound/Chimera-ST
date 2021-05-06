#!/bin/bash

for split in train test valid; do
    for lang in en pt-br; do
        if [[ $lang == 'pt-br' ]]; then
            lang_tag='pt'
        else
            lang_tag=$lang
        fi
        for type in bin idx; do
            cp $WMT_ROOT/bin/$split.en-pt-br.$lang.$type\
                $WMT_ROOT/bin/$split.en-pt.$lang_tag.$type
        done
    done
done

cp $WMT_ROOT/bin/dict.pt-br.txt $WMT_ROOT/bin/dict.pt.txt
