The following are original files Chi Han:
```
run.sh
chimera/generate
chimera/prepare_data/a*
chimera/prepare_data/download-opus100.sh
chimera/resources
chimera/scripts
chimera/tools/*.sh
chimera/tools/*gpu*.py
chimera/tools/hand-make-config.py
chimera/tools/plot_output.py
```

The following file(s) is a combination of Fairseq and Chi Han's efforts:
```
chimera/prepare_data/download_wmt.sh
chimera/prepare_data/prepare*
chimera/tools/eval-average-checkpoint.py
fairseq/criterions/ctc_chi.py
fairseq/criterions/joint_criterion.py
fairseq/criterions/triplet_st_mt_contrastive.py
fairseq/criterions/triplet_st_mt_samplecontrastive.py
fairseq/criterions/wav2vec_criterion_bimodal.py
fairseq/data/audio/triplet_dataset.py
fairseq/data/joint_dataset.py
fairseq/models/chimera
fairseq/models/joint_model.py
fairseq/models/wav2vec/wav2vec2_quasiwave.py
fairseq/models/composite_encoder_chi.py
fairseq/tasks/joint_task.py
fairseq/tasks/joint_mtst.py
fairseq/trainer.py
fairseq_cli/generate.py
fairseq_cli/interactive*.py
fairseq_cli/train.py
```

The following file(s) is from Mosesdecoder codebase
(https://github.com/moses-smt/mosesdecoder):
```
chimera/tools/detokenizer.perl
```

All other files are directly from the Fairseq toolkit,
(https://github.com/pytorch/fairseq), or with only minor modifications.
and thereby carrying its MIT License.
