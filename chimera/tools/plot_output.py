import argparse
import math
import pprint
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from copy import deepcopy

# open the file
parser = argparse.ArgumentParser()
parser.add_argument("--files", "-f", type=str, nargs='+')
parser.add_argument("--save_image", "-i", type=str, default='')
parser.add_argument("--save_log", "-l", type=str, default='')
parser.add_argument("--no_fig", "-n", action='store_true')
parser.add_argument("--train_flag", default='train')
parser.add_argument("--dev_flag", default='dev')
parser.add_argument("--test_flag", default='eval')
parser.add_argument("--stat_types", default=[], type=str, nargs='+')
parser.add_argument("--best-epoch-according-to", default="dev_wave_loss",
                    type=str)
parser.add_argument("--best-epoch-polarity", default="min", type=str)
args = parser.parse_args()

result_logs = {}
log_lines = {}
output = {}


def split_message(line):
    if ' | ' not in line or line.startswith('Chi-Log'):
        return {}
    splitted = line.split(' | ')
    if len(splitted) < 3:
        return {}
    return {
        'time': splitted[0],
        'entry': splitted[2],
        'messages': splitted[3:],
    }


def update_callbacks(callbacks, step_name: str):
    for item in callbacks:
        step = int(item.pop(step_name))
        to_pop = []
        for name, value in item.items():
            if value == 'nan':
                to_pop.append(name)
        for name in to_pop:
            item.pop(name)
        if step not in result_logs:
            result_logs[step] = item
        else:
            result_logs[step].update(item)


for filename in args.files:
    with open(filename, 'r') as log_file:
        log_lines[filename] = log_file.readlines()

for filename, lines in log_lines.items():
    for line in lines:
        if '| INFO |' in line:
            head_line = line
            break
    # head_entry = split_message(head_line)['entry']
    # if head_entry in ['fairseq_cli.train', 'fairseq_cli.preprocess']:
    # training log
    parsed_lines = list(filter(lambda x: x != {},
                               map(split_message, lines)))
    train_messages = [line['messages'] for line in parsed_lines
                      if line['entry'] == args.train_flag]
    dev_messages = [line['messages'] for line in parsed_lines
                    if line['entry'] == args.dev_flag]

    def extract_log(messages, prefix: str):
        skips = (
            "valid on",
        )

        def prefix_when_not(_prefix, string):
            if not string.startswith(_prefix):
                return _prefix + string
            else:
                return string

        def extract_from_line(line):
            result = {}
            for piece in line:
                if '{' not in piece:
                    if all(skip not in piece for skip in skips):
                        result.update({
                            prefix_when_not(prefix, piece.split(' ')[0]):
                            float(piece.split(' ')[1].strip())
                        })
                    else:
                        pass
                else:
                    result.update({
                        prefix_when_not(prefix, key): float(value)
                        for key, value in json.loads(piece).items()
                    })
            return result

        messages = [extract_from_line(line) for line in messages]
        return messages

    update_callbacks(extract_log(train_messages, args.train_flag+'_'),
                     args.train_flag+'_num_updates')
    update_callbacks(extract_log(dev_messages, args.dev_flag+'_'),
                     args.dev_flag+'_num_updates')
    # maybe it is also an evaluation log ?
    epoch = None
    to_update = []
    lines.insert(0, filename)
    for line in lines:
        if ('|| Evaluating ' in line and 'Chi-Log || ' in line) or \
                'chi-auto-generate' in line:
            find = re.search(
                r"checkpoint(\d+|_best|_last)\.pt", line)
            if find is not None:
                epoch = find.group()[10:-3]
                if epoch == '_best':
                    epoch = 'best'
                elif epoch == '_last':
                    epoch = 'last'
                else:
                    epoch = int(epoch)

        epoch_name = args.test_flag + '_epoch'
        logs = {epoch_name: epoch}
        if 'BLEU' in line:
            bleu = re.search(r'BLEU[4]* = [\d\.]+', line).group().\
                split(' ')[-1]
            if 'test1' in line:
                logs['eval_BLEUST4'] = float(bleu)
            else:
                logs['eval_BLEU4'] = float(bleu)
        elif 'BLEU+' in line:
            bleu = re.search(r'BLEU[a-z\.\+\-\d]+ = [\d\.]+', line).group().\
                split(' ')[-1]
            logs['eval_BLEU4'] = float(bleu)
        elif ' acc ' in line:
            acc = re.search(r'acc [\d\.]+', line)
            if acc is not None:
                acc = acc.group().split(' ')[-1]
                logs['eval_acc'] = float(acc)
        elif 'WER: ' in line:
            wer = re.search(r"WER: [\d\.]+", line).group().\
                split(' ')[-1]
            logs['eval_WER'] = float(wer)
        if len(logs) > 1:
            if logs[epoch_name] is not None:
                if logs[epoch_name] not in ['best', 'last']:
                    to_update.append(logs)
                elif logs[epoch_name] == 'best':
                    output.setdefault(logs.pop(epoch_name), {}).update(logs)
    update_callbacks(to_update, epoch_name)

# check results
log_classes = (
    'clip', 'gnorm', 'num_updates',
    'loss', 'lr', 'best_loss',
    'ppl',
    'wer', 'uer', 'raw_wer', 'best_wer',
    'BLEU4', 'BLEUST4', 'acc', 'WER', 'accuracy',
    # 'wall',
) + tuple(args.stat_types)
log_names = set.union(*tuple(
    set(epoch_log.keys()) for epoch_log in result_logs.values()
))
log_names = list(name for name in log_names if
                 any(name.endswith(_class) for _class in log_classes))
log_names.sort()
n_column = int(math.sqrt(len(log_names)))
if n_column != 1:
    n_row = (len(log_names) - 1) // n_column + 1
else:
    n_row = len(log_names)
fig, axes = plt.subplots(n_row, n_column)
fig.set_size_inches(8, 20)
if n_column != 1 and n_row != 1:
    axes = [ax for row in axes for ax in row]
elif n_column == 1 and n_row == 1:
    axes = [axes]

log_values = {
    name: np.array([[step, logs[name]] for step, logs in result_logs.items()
                    if name in logs])
    for name in log_names
}
log_values = {name: values for name, values in log_values.items()
              if values.shape[0] != 0}
for name, values in log_values.items():
    log_values[name] = values[values[:, 0].argsort()]

'''
_epochs = dict()
for name, best in output['best'].items():
    values = log_values[name]
    epoch = int(values[np.where(values[:, 1] == best)[0][0], 0])
    _epochs[name] = epoch
for name, epoch in _epochs.items():
    output['best'][name+"_epoch"] = epoch
'''
if args.best_epoch_according_to in log_values:
    _according_to = deepcopy(log_values[args.best_epoch_according_to])
    _according_to = _according_to.tolist()
    _according_to.sort(key=lambda item: item[0])
    _values = log_values[args.best_epoch_according_to][:, 1]
    if args.best_epoch_polarity == "min":
        _index = _values.argmin()
    elif args.best_epoch_polarity == "max":
        _index = _values.argmax()
    else:
        raise Exception()
    _epoch = _index + int(log_values[list(output['best'].keys())[0]][0, 0])
    output['best']["epoch"] = _epoch

for index, (name, values) in enumerate(log_values.items()):
    ax = axes[index]
    ax.plot(values[:, 0], values[:, 1])
    ax.set_title(name)
    ax.grid()
    if len(values) <= 1:
        output[name] = {
            'single-value': values[0, 1]
        }
        continue
    mean = values[-5:, 1].mean()
    std = values[-5:, 1].std(ddof=1)
    output[name] = {
        'final': values[-1, 1],
        'min': values[:, 1].min(),
        'max': values[:, 1].max(),
        'mean': mean,
        'std': std,
    }
    # plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
pprint.pprint(output)

fig.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.25,
                    hspace=0.15)

if args.save_log != '':
    os.makedirs(os.path.dirname(args.save_log), exist_ok=True)
    with open(args.save_log, 'w') as output_logfile:
        output_logfile.write(pprint.pformat(output))
if args.save_image != '':
    if '/' not in args.save_image:
        args.save_image = args.save_log[:args.save_log.rfind('/') + 1] \
            + args.save_image
        print(args.save_image, flush=True)
    plt.savefig(args.save_image)
if not args.no_fig:
    plt.show()
