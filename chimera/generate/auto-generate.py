import glob
import sys
import os
import time
import argparse
import pprint
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('chi-auto-generate')

SUICIDE_CODE = 'chimera/tools/auto-generate-suicide.code'
parser = argparse.ArgumentParser()
parser.add_argument("--dirname", "-d", type=str)
parser.add_argument("--log_dirname", type=str)
parser.add_argument("--silent", "-s", action='store_true')
parser.add_argument("--eval_log_suffix", type=str, default=['eval'],
                    nargs='+')
parser.add_argument("--no_log_each", "-n", action='store_true')
parser.add_argument("--haruna_logdir", "-l", type=str, default=None)
parser.add_argument("--generate_script", "-g", type=str, nargs='+', default=[])
parser.add_argument("--save_only", action='store_true')
parser.add_argument("--one_pass", action='store_true')
args = parser.parse_args()
assert args.save_only or args.generate_script, \
    "either save-only or provide a generate script"


if args.log_dirname is None:
    args.log_dirname = args.dirname

generated_pts = {}
if args.haruna_logdir is not None:
    commands = []
    commands.append(f'hdfs dfs -mkdir -p {args.haruna_logdir}/checkpoints')
    if not args.save_only:
        commands.append(f'hdfs dfs -mkdir -p {args.haruna_logdir}/logs')
    for command in commands:
        logger.info(f'running command {command}')
        if os.system(command) != 0:
            logger.info('failed')


def search_gpu_script(script_name):
    if script_name.endswith('.sh'):
        candidate = script_name[:-3] + '-gpu' + '.sh'
        if os.path.exists(candidate):
            script_name = candidate
    return script_name


def evaluate(ckpt_file):
    ckpt_file_tail = ckpt_file.split('/')[-1]
    mtime = os.path.getmtime(ckpt_file)
    generated_pts[ckpt_file] = {
        'mtime': mtime,
        'evaluate_time': time.ctime(),
    }

    if args.haruna_logdir is not None:
        logger.info(f'stored to'
                    f" {args.haruna_logdir}/checkpoints/{ckpt_file_tail}")
        if os.system(
            f"hdfs dfs -put -f {ckpt_file}"
                f" {args.haruna_logdir}/checkpoints/{ckpt_file_tail}") != 0:
            logger.info('failed')

    if args.save_only:
        # only saving the ckpt
        return

    # evaluating ckpts
    for i, script in enumerate(args.generate_script):
        if args.one_pass:
            script = search_gpu_script(script)
        suffix = args.eval_log_suffix[i]
        output_file = os.path.join(
            args.log_dirname,
            ckpt_file.replace('.pt', f'_{suffix}.txt').split('/')[-1]
        )
        output_file_tail = output_file.split('/')[-1]

        if args.silent:
            stream_op = f' &> {output_file}'
        elif args.no_log_each:
            stream_op = ''
        else:
            stream_op = f' 2>&1 | tee {output_file}'
        command = f"bash {script} {ckpt_file}" + stream_op
        logger.info(f'Evaluating {ckpt_file}, command: {command}')
        if os.system("/bin/bash -c \"" + command + "\"") != 0:
            logger.info('failed')

        if args.haruna_logdir is not None:
            logger.info(f'stored to'
                        f" {args.haruna_logdir}/logs/{output_file_tail}")
            if os.system(
                f"hdfs dfs -put -f {output_file}"
                    f" {args.haruna_logdir}/logs/{output_file_tail}") != 0:
                logger.info('failed')
            with open(output_file, 'r') as output_contents:
                lines = output_contents.readlines()
                pprint.pprint(lines[-20:])


def main():
    while True:
        if os.path.exists(SUICIDE_CODE):
            args.one_pass = True

        cur_list = glob.glob(os.path.join(args.dirname, "*checkpoint*.pt"))
        to_evaluate = []
        for ckpt_file in cur_list:
            mtime = os.path.getmtime(ckpt_file)
            if (ckpt_file not in generated_pts or
                    generated_pts[ckpt_file]['mtime'] != mtime):
                to_evaluate.append(ckpt_file)

        for ckpt_file in to_evaluate:
            evaluate(ckpt_file)

        if args.one_pass:
            break
        time.sleep(3)


if __name__ == '__main__':
    main()
