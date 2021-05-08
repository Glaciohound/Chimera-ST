import argparse
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
args = parser.parse_args()

os.makedirs(
    args.data_dir,
    exist_ok=True
)

output_config = {
    "audio_root": "",
    "bpe_tokenizer": {
        "bpe": "sentencepiece",
        "sentencepiece_model":
        os.path.join(args.data_dir, "spm_unigram10000_wave_joint.model")
    },
    "input_channels": 1,
    "input_feat_per_channel": 80,
    "sampling_alpha": 1.0,
    "src_bpe_tokenizer": {
        "bpe": "sentencepiece",
        "sentencepiece_model":
        os.path.join(args.data_dir, "spm_unigram10000_wave_joint.model")
    },
    "src_vocab_filename": "spm_unigram10000_wave_joint.txt",
    "use_audio_input": True,
    "vocab_filename": "spm_unigram10000_wave_joint.txt",
}

with open(os.path.join(args.data_dir, "config_wave.yaml"), 'w') as f:
    yaml.dump(output_config, f)
