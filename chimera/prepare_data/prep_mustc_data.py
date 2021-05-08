#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
import os
import os.path as op
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from chimera.prepare_data.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    # load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def os_system(command):
    logger.info(command)
    return os.system(command)


chi_spm_dir = "chimera/resources"


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = op.join(root, f"en-{lang}", "data", split)
        wav_root, txt_root = op.join(_root, "wav"), op.join(_root, "txt")
        wav_relative_root = op.join(f"en-{lang}", "data", split, "wav")
        assert op.isdir(_root) and op.isdir(wav_root) and op.isdir(txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            pass
        logger.info(f"loading task yaml config: {split}.yaml")
        with open(op.join(txt_root, f"{split}.yaml")) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            logger.info(f"loading text sources {split}.{_lang}")
            with open(op.join(txt_root, f"{split}.{_lang}")) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = op.join(wav_root, wav_filename)
            wav_relative_path = op.join(wav_relative_root, wav_filename)
            sample_rate = torchaudio.info(wav_path)[0].rate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{op.splitext(wav_filename)[0]}_{i}"
                self.data.append(
                    (
                        wav_path,
                        wav_relative_path,
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[
            Tensor, int, str, str, str, str, str]:
        wav_path, wav_rel_path, offset, n_frames, sr,\
            src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = torchaudio.load(
            wav_path, offset=offset, num_frames=n_frames)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id,\
            wav_rel_path, offset

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]
    if args.triplet:
        MANIFEST_COLUMNS.append("src_text")
    logger.info(f"MANIFEST_COLUMNS: {MANIFEST_COLUMNS}")

    for lang in args.languages:
        if args.triplet or args.joint_spm is not None:
            folder = f"en-{lang}-triplet"
        else:
            folder = f"en-{lang}"
        cur_root = op.join(args.data_root, f"en-{lang}")
        if not op.isdir(cur_root):
            logger.info(f"{cur_root} does not exist. Skipped.")
            continue
        # Extract features
        feature_root = op.join(cur_root, "fbank80")
        zip_filename = "fbank80.zip"
        os.makedirs(feature_root, exist_ok=True)
        if os.path.exists(os.path.join(cur_root, zip_filename)):
            logger.info(f"{zip_filename} already existed and downloaded")
        elif not args.ignore_fbank80:
            logger.info(f"generating fbank features: {zip_filename}")
            for split in MUSTC.SPLITS:
                logger.info(f"Fetching split {split}...")
                dataset = MUSTC(args.data_root, lang, split)
                logger.info("Extracting log mel filter bank features...")
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    extract_fbank_features(
                        waveform, sample_rate,
                        op.join(feature_root, f"{utt_id}.npy")
                    )
            # Pack features into ZIP
            zip_path = op.join(cur_root, zip_filename)
            logger.info("ZIPing features...")
            create_zip(feature_root, zip_path)
        if not args.ignore_fbank80:
            logger.info("Fetching ZIP manifest...")
            zip_manifest = get_zip_manifest(
                args.data_root, f"en-{lang}/{zip_filename}")
        else:
            zip_manifest = None
        # Generate TSV manifest
        logger.info("Generating manifest...")
        train_text = []
        if args.triplet:
            train_src_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")

            # try downloading from hdfs first
            if os.path.exists(f'{cur_root}/{split}_{args.task}.tsv'):
                logger.info(f"downloaded existing {split}_{args.task}.tsv")
                continue
            else:
                logger.info("generating tsv records")

            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id,\
                    wav_file, offset in tqdm(dataset):
                manifest["id"].append(utt_id)
                if args.task in ['asr', 'st']:
                    manifest["audio"].append(zip_manifest[utt_id])
                    duration_ms = int(wav.size(1) / sr * 1000)
                    manifest["n_frames"].append(int(1 + (duration_ms - 25)/10))
                elif args.task in ['wave']:
                    length = int(wav.size(1))
                    manifest["audio"].append(f"{wav_file}:{offset}:{length}")
                    manifest["n_frames"].append(length)
                else:
                    raise Exception("task not supported:" + args.task)
                if args.triplet:
                    manifest["src_text"].append(src_utt)
                manifest["tgt_text"].append(src_utt if args.task == "asr"
                                            else tgt_utt)
                manifest["speaker"].append(speaker_id)

            if is_train_split:
                train_text.extend(manifest["tgt_text"])
                if args.triplet:
                    train_src_text.extend(manifest["src_text"])
            df = pd.DataFrame.from_dict(manifest)
            if args.task in ['asr', 'st']:
                df = filter_manifest_df(df, is_train_split=is_train_split)
            else:
                df = filter_manifest_df(df, is_train_split=is_train_split,
                                        max_n_frames=1000000)
            save_df_to_tsv(df, op.join(cur_root, f"{split}_{args.task}.tsv"))

        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        if args.joint_spm is not None:
            spm_filename_prefix = \
                f"spm_{args.vocab_type}{v_size_str}_{args.task}_joint"
            os_system(f"cp"
                      f" {chi_spm_dir}/{args.joint_spm}/{spm_filename_prefix}*"
                      f" {cur_root}")
        else:
            spm_filename_prefix = \
                f"spm_{args.vocab_type}{v_size_str}_{args.task}"
            os_system(f"cp"
                      f" {chi_spm_dir}/{folder}/{spm_filename_prefix}*"
                      f" {cur_root}")
        if os.path.exists(
                f"{cur_root}/{spm_filename_prefix}.txt"):
            logger.info("downloaded existing vocab files")
        elif len(train_text) != 0:
            logger.info("generating vocab files for target language")
            with NamedTemporaryFile(mode="w") as f:
                for t in train_text:
                    f.write(t + "\n")
                gen_vocab(
                    f.name,
                    op.join(cur_root, spm_filename_prefix),
                    args.vocab_type,
                    args.vocab_size,
                )
            if args.triplet or args.joint_spm is not None:
                logger.info("generating vocab files for source language")
                if args.triplet:
                    src_spm_filename_prefix = spm_filename_prefix+'_src'
                else:
                    src_spm_filename_prefix = spm_filename_prefix
                with NamedTemporaryFile(mode="w") as f:
                    for t in train_src_text:
                        f.write(t + "\n")
                    gen_vocab(
                        f.name,
                        op.join(cur_root, src_spm_filename_prefix),
                        args.vocab_type,
                        args.vocab_size,
                    )
        else:
            raise Exception("you need to process raw file from start")

        # Generate config YAML
        config_yaml = f'config_{args.task}.yaml'
        logger.info(f"generating config: {config_yaml}")
        gen_config_yaml(
            args.data_root,
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=config_yaml,
            specaugment_policy="lb",
            use_audio_input=(args.task in ['wave']),
            src_spm_filename=spm_filename_prefix+"_src.model" if args.triplet
            else spm_filename_prefix+".model" if args.joint_spm is not None
            else None
        )

        # Clean up
        shutil.rmtree(feature_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str,
                        choices=["asr", "st", "wave"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--languages", type=str, nargs='+')
    parser.add_argument("--ignore_fbank80", action='store_true')
    parser.add_argument("--triplet", action='store_true')
    parser.add_argument("--joint_spm", type=str, default=None)
    args = parser.parse_args()

    assert not args.triplet or args.joint_spm is None

    process(args)


if __name__ == "__main__":
    main()
