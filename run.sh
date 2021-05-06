#!/bin/bash

export version=a
export seed=1
export resume=False
export num_gpus=8


# parsing arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--script) export script="$2"; shift ;;
        -g|--num-gpus) export num_gpus="$2"; shift ;;
        -s|--seed) export seed="$2"; shift ;;
        -r|--resume) export resume=True ;;
        -v|--version) export version="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


bash $SCRIPTPATH/$script
