#!/bin/bash

export tool="sacrebleu"
export tmp="tmp"
export clean="true"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-file|-i) export input_file="$2"; shift ;;
        --output-file|-o) export output_file="$2"; shift ;;
        --tool|-t) export tool="$2"; shift ;;
        --tmp) export tmp="$2"; shift ;;
        --do-not-clean) export clean="false";;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
echo "arguments"
echo "input file: $input_file"
echo "output file: $output_file"
echo "tool to use: $tool"
echo "tmp directory: $tmp"
echo

mkdir $tmp
target_file=$tmp/bleu-target.txt
hypo_file=$tmp/bleu-hypo.txt
tokenizer="perl chi/scripts/detokenizer.perl"

TAB=$'\t'
grep "^T-" $input_file | \
    sed -e "s/^T-[0-9]*${TAB}//g" | \
    $tokenizer | \
    cat > $target_file
grep "^D-" $input_file | \
    sed -e "s/^D-[0-9]*${TAB}[-.0-9]*${TAB}//g" | \
    $tokenizer | \
    cat > $hypo_file

if [[ $output_file != '' ]]; then
    command_tail="2>&1 | tee $output_file"
    echo "saving output to $output_file"
fi

if [[ $tool == 'sacrebleu' ]]; then
    sacrebleu $target_file -i $hypo_file $command_tail
else
    echo "tool $tool not recognized!"
fi

if [[ $clean == 'true' ]]; then
    rm -r $tmp
else
    echo "not clean tmp dir $tmp/"
fi
