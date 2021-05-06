import os
import sys
import argparse
import numpy as np
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("eval-average-checkpoint")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-codename", type=str, default=None)
    parser.add_argument("--hdfs-url", type=str, default=None)
    parser.add_argument(
        "--eval-script", type=str,
        default="chi/experiments/generate/generate-mustc-final-gpu.sh")
    parser.add_argument("--number-of-ckpts", type=int, default=7)
    parser.add_argument("--center-of-ckpts", type=int)
    parser.add_argument("--ckpt-dir", default="checkpoints/st")
    parser.add_argument("--output-name", type=str, default="average")
    parser.add_argument("--stride-of-ckpts", type=int, default=1)
    parser.add_argument("--ckpt-prefix", type=str, default="checkpoint")
    parser.add_argument("--average-only", action="store_true")
    parser.add_argument("--ckpt-dir-n-segments", type=int, default=2)
    parser.add_argument("--ckpt-root", type=str, default=None)
    return parser


def os_system(command):
    logger.info(f"running commmand: {command}")
    os.system(command)


def get_ckpt_info(ckpt_root, trial_codename):
    assert ckpt_root is not None
    codebook = {
        "de":       {"center": 101, "hdfs_url": f"{ckpt_root}/20210126-13/c"},
        "fr":       {"center": 101, "hdfs_url": f"{ckpt_root}/20210126-13/d"},
        "ru":       {"center": 95, "hdfs_url": f"{ckpt_root}/20210126-13/g"},
        "es":       {"center": 125, "hdfs_url": f"{ckpt_root}/20210126-13/h"},
        "it":       {"center": 74, "hdfs_url": f"{ckpt_root}/20210129-4/a"},
        "ro":       {"center": 79, "hdfs_url": f"{ckpt_root}/20210129-4/b"},
        "pt":       {"center": 68, "hdfs_url": f"{ckpt_root}/20210129-4/c"},
        "nl":       {"center": 53, "hdfs_url": f"{ckpt_root}/20210129-4/d"},
        "de-mem16": {"center": 107, "hdfs_url": f"{ckpt_root}/20210126-17/a"},
        "fr-mem16": {"center": 100, "hdfs_url": f"{ckpt_root}/20210126-7/b"},
        "ru-mem16": {"center": 155, "hdfs_url": f"{ckpt_root}/20210126-4/c"},
        "es-mem16": {"center": 81, "hdfs_url": f"{ckpt_root}/20210126-4/d"},
        "it-mem16": {"center": 85, "hdfs_url": f"{ckpt_root}/20210129-2/a"},
        "ro-mem16": {"center": 87, "hdfs_url": f"{ckpt_root}/20210129-2/b"},
        "pt-mem16": {"center": 60, "hdfs_url": f"{ckpt_root}/20210129-2/c"},
        "nl-mem16": {"center": 40, "hdfs_url": f"{ckpt_root}/20210129-2/d"},
    }
    return codebook[trial_codename]


def main():
    # os_system(f"export target={args.target}")
    if args.trial_codename is not None:
        assert args.hdfs_url is None and args.center_of_ckpts is None
        info = get_ckpt_info(args.ckpt_root, args.trial_codename)
        args.hdfs_url = info["hdfs_url"]
        args.center_of_ckpts = info["center"]
    start_of_ckpts = (
        args.center_of_ckpts -
        (args.number_of_ckpts // 2) * args.stride_of_ckpts
    )
    end_of_ckpts = (
        start_of_ckpts + (args.number_of_ckpts-1) *
        args.stride_of_ckpts
    )
    ckpt_epoches = (
        np.arange(args.number_of_ckpts) * args.stride_of_ckpts
        + start_of_ckpts
    ).tolist()
    ckpt_names = [f"{args.ckpt_prefix}{i}.pt" for i in ckpt_epoches]
    logger.info(f"using checkpoints {ckpt_names}")
    if args.hdfs_url is not None:
        if not args.hdfs_url.endswith("checkpoints"):
            args.hdfs_url = os.path.join(args.hdfs_url, "checkpoints")
        args.ckpt_dir = os.path.join(
            args.ckpt_dir,
            '-'.join(args.hdfs_url.split('/')[
                -(1+args.ckpt_dir_n_segments):-1
            ])
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_exists = [
        os.path.exists(os.path.join(args.ckpt_dir, _ckpt_name))
        for _ckpt_name in ckpt_names
    ]

    if all(ckpt_exists):
        logger.info("all checkpoints already exists")
    else:
        assert args.hdfs_url is not None, "not HDFS url for downloading ckpts"
        for _ckpt_name, _exists in zip(ckpt_names, ckpt_exists):
            if not _exists:
                os_system(f"hdfs dfs -get {args.hdfs_url}/{_ckpt_name} "
                          f"{args.ckpt_dir}"
                          )

    # if not os.path.exists(os.path.join(
    #     args.ckpt_dir, args.output_name+'.pt'
    # )):
    os_system(
        f"python3 scripts/average_checkpoints.py "
        f"--inputs {args.ckpt_dir} "
        f"--output {args.ckpt_dir}/{args.output_name}.pt "
        f"--num-epoch-checkpoints {args.number_of_ckpts} "
        f"--checkpoint-upper-bound {end_of_ckpts}"
    )
    os_system(
        f"hdfs dfs -put -f "
        f"{args.ckpt_dir}/{args.output_name}.pt "
        f"{args.hdfs_url}"
    )

    if not args.average_only:
        '''
        for _ckpt_name in ckpt_names:
            os.remove(os.path.join(args.ckpt_dir, _ckpt_name))
    else:
        '''
        os_system(
            f"bash {args.eval_script} "
            f"{args.ckpt_dir}/{args.output_name}.pt "
            f"2>&1 | tee {args.ckpt_dir}/{args.output_name}_eval.txt"
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()
