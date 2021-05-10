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
    return parser


def os_system(command):
    logger.info(f"running commmand: {command}")
    os.system(command)


def main():
    # os_system(f"export target={args.target}")
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
    ckpt_exists = [
        os.path.exists(os.path.join(args.ckpt_dir, _ckpt_name))
        for _ckpt_name in ckpt_names
    ]

    if all(ckpt_exists):
        logger.info("all checkpoints already exists")
    else:
        ckpts_not_found = [
            _ckpt_name
            for _ckpt_name, _ckpt_exists in zip(ckpt_names, ckpt_exists)
            if _ckpt_exists
        ]
        logger.info(f"checkpoints not found: {ckpts_not_found}")

    os_system(
        f"python3 scripts/average_checkpoints.py "
        f"--inputs {args.ckpt_dir} "
        f"--output {args.ckpt_dir}/{args.output_name}.pt "
        f"--num-epoch-checkpoints {args.number_of_ckpts} "
        f"--checkpoint-upper-bound {end_of_ckpts}"
    )

    if not args.average_only:
        os_system(
            f"bash {args.eval_script} "
            f"{args.ckpt_dir}/{args.output_name}.pt "
            f"2>&1 | tee {args.ckpt_dir}/{args.output_name}_eval.txt"
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()
