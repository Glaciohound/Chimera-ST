# Chimera: Learning Shared Semantic Space for Speech-to-Text Translation (Nightly Version)

This is a Pytorch implementation for the [Chimera][Chimera] paper (ACL Findings 2021), which aims to bridge the modality gap by unifying the task of MT (textual Machine Translation) and ST (Speech-to-Text Translation).
It has achieved new SOTA performance on all 8 language pairs in MuST-C benchmark, by utilizing an external MT corpus.

This repository is up to now a nightly version,
and not fully tested on configurations other than the authors' working environment.
However, we encourage you to first have a look at the results and model codes to get a general impression of what this project is about.

You are also more than welcome to test our code on your machines,
and report feedbacks on results, bugs and performance!


## Using a Trained Checkpoint

Our trained checkpoints are available at:

| Translation Direction | filename | External url |
| --------------------- | -------- | ------------ |
| English-to-Deutsch    | Chimera_EN2DE.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2DE.pt |
| English-to-French     | Chimera_EN2FR.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2FR.pt |
| English-to-Russian    | Chimera_EN2RU.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2RU.pt |
| English-to-Espanol    | Chimera_EN2ES.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2ES.pt |
| English-to-Italiano   | Chimera_EN2IT.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2IT.pt |
| English-to-Romanian   | Chimera_EN2RO.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2RO.pt |
| English-to-Portuguese | Chimera_EN2PT.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2PT.pt |
| English-to-Dutch      | Chimera_EN2NL.pt | http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/chimera/Chimera_EN2NL.pt |



## Evaluating Our Checkpoints



## Training a Model on MuST-C

Let's first take a look at training an English-to-Deutsch model as an example.


### Data Preparation

0. For configuration, please set the global variables of
`$WMT_ROOT`, `$MUSTC_ROOT` and `SAVE_ROOT`
These will be where to put the datasets and checkpoints.
For example:
```
export MUSTC_ROOT="speech_data/mustc"
export WMT_ROOT="wmt_data"
export SAVE_ROOT="checkpoints"
mkdir -p $MUSTC_ROOT $WMT_ROOT $SAVE_ROOT
```

1. Download and uncompress the EN-to-DE MuST-C dataset to `$MUSTC_ROOT/en-de`.
Tip: to speed up uncompressing a too large file,
you can replace `tar xzvf` with: ` pigz -dc $TARFILE | tar xvf - `

2. Download the WMT to `$WMT_ROOT/orig` via:
```
bash chimera/prepare_data/download-wmt.sh --wmt14 --data-dir $WMT_ROOT --target de
```

3. Append MuST-C text data to $WMT_ROOT, and prepare the datasets and produce a joint spm dictionary:
```
bash chimera/prepare_data/prepare-wmt14en2any.sh \
    --data-dir $WMT_ROOT --wmt14 --original-dev \
    --external mustc --target de --subword spm
python3 chimera/prepare_data/prep_mustc_data.py \
    --data-root $MUSTC_ROOT --task wave \
    --ignore_fbank80 --joint_spm wmt14-en-de-spm \
    --languages de --vocab-type unigram --vocab-size 10000
```
Notice: if the first command is executed correctly, you will see one line in the output:
```
Existing spm dictionary chimera/resources/wmt14-en-de-spm detected. Copying...
```
If not, the program will still produce one dictionary on the run and reports
`No existing spm detected. Learning unigram spm on wmt14_en_de/tmp/train.de-en ...`
This is okay.
The only risk is a potential mismatch to already trained checkpoints we provided.


### Training

To reproduce the results in the last row in Figure 1 in paper,
you can directly use the training scripts available as follows.

4. Pre-training on MT data:
```
bash run.sh --script chimera/scripts/train-en2de-MT.sh
```

If you like, you can specify some arguments other than default values.
The default setting is `--seed 1 --num-gpus 8`, which makes the command look like
`bash run.sh --script chimera/scripts/train-en2de-MT.sh --seed 1 --num-gpus 8`.
Value for `--num-gpus` is recommended to be power of 2, and smaller than 8, e.g. {1, 2, 4, 8}.


5. Fine-tuning on MuST-C data:

```
bash run.sh --script chimera/scripts/train-en2de-ST.sh
```
This script moves the MT-pre-trained model from `${MT_SAVE_DIR}` to `${ST_SAVE_DIR}`
as a initialization for ST fine-tuning.

Optionally, if you need to resume a single ST training,
you can run `bash chimera/scripts/train-en2de-ST.sh --resume`
to avoid overwriting the existing `${ST_SAVE_DIR}/checkpoint_last.pt`.

The scripts in step 4 and 5 forks a separate background evaluation process while running.
The process monitors `$MT_SAVE_ROOT` or `$ST_SAVE_ROOT`
and evaluates any new checkpoints.
Don't worry, it will be automatically killed after the training finishes,
unless the script is Ctrl-C'ed,
in which case, you can manually raise the suicide flag by
`touch chimera/tools/auto-generate-suicide.code`
to kill the background generation process

### Other Language Pairs
Simiarly, for other 7 language pairs, you can run commands:
```
bash run.sh --script chimera/scripts/train-EN2FR.sh
bash run.sh --script chimera/scripts/train-EN2RU.sh
bash run.sh --script chimera/scripts/train-EN2ES.sh
bash run.sh --script chimera/scripts/train-EN2IT.sh
bash run.sh --script chimera/scripts/train-EN2RO.sh
bash run.sh --script chimera/scripts/train-EN2PT.sh
bash run.sh --script chimera/scripts/train-EN2NL.sh
```
which gives results for English-to-{Franch, Russian, Espanol, Italiano, Romanian, Portuguese, Dutch}
