#!/bin/bash

export version=a
export seed=1
export resume=False
export num_gpus=$(python3 chimera/tools/count_gpus.py)
export SCRIPTPATH=$(pwd)
export target=de
export dataset=wmt14
export max_updates=500000


# parsing arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--script) export script="$2"; shift ;;
        -g|--num-gpus) export num_gpus="$2"; shift ;;
        --seed) export seed="$2"; shift ;;
        -r|--resume) export resume=True ;;
        -v|--version) export version="$2"; shift ;;
        -t|--target) export target="$2"; shift ;;
        -d|--dataset) export dataset="$2"; shift ;;
        -m|--max_updates) export max_updates="$2"; shift ;;
        -c|--checkpoint) export checkpoint="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

sudo rm /root/.pip/pip.conf
sudo pip3 install -e . -v

bash $script
