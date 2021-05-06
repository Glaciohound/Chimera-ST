from fairseq import scoring
from fairseq.data import encoders
from fairseq.models.joint_model import JointModel


class JointTokenizer:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers
        self.main_task_name = None

    @classmethod
    def build_tokenizer(cls, task_configs):
        tokenizers = {}
        for _name, _config in task_configs.items():
            tokenizers[_name] = encoders.build_tokenizer(_config.tokenizer)
        return cls(tokenizers)

    @property
    def _tokenizer(self):
        return self.tokenizers[self.main_task_name]

    def decode(self, x):
        if self._tokenizer is not None:
            return self._tokenizer.decode(x)
        else:
            return x


class JointBPE:
    def __init__(self, bpes):
        self.bpes = bpes
        self.main_task_name = None

    @classmethod
    def build_bpe(cls, task_configs, tasks):
        bpes = {}
        for _name, _config in task_configs.items():
            bpes[_name] = tasks[_name].build_bpe(_config.bpe)
        return cls(bpes)

    @property
    def _bpe(self):
        return self.bpes[self.main_task_name]

    def decode(self, x):
        if self._bpe is not None:
            return self._bpe.decode(x)
        else:
            return x


class JointScorer:
    def __init__(self, scorers):
        self.scorers = scorers
        self.main_task_name = None

    @classmethod
    def build_scorer(cls, task_configs, tgt_dict):
        scorers = {}
        for _name, _config in task_configs.items():
            scorers[_name] = scoring.build_scorer(
                _config.scoring, tgt_dict[_name])
        return cls(scorers)

    @property
    def _scorer(self):
        return self.scorers[self.main_task_name]

    def add_string(self, target_str, detok_hypo_str):
        if hasattr(self._scorer, 'add_string'):
            return self._scorer.add_string(target_str, detok_hypo_str)
        else:
            raise AttributeError()

    def add(self, target_tokens, hypo_tokens):
        if hasattr(self._scorer, 'add'):
            return self._scorer.add(target_tokens, hypo_tokens)
        else:
            raise AttributeError()

    def result_string(self):
        strings = {
            _name: _scorer.result_string()
            for _name, _scorer in self.scorers.items()
        }
        return str(strings)


class JointTargetDictionary:
    def __init__(self, dictionaries, main_task_name):
        self.dictionaries = dictionaries
        self.main_task_name = main_task_name

    @classmethod
    def build_dictionary(cls, tasks, main_task_name):
        dictionaries = {
            _name: _task.target_dictionary
            for _name, _task in tasks.items()
        }
        return cls(dictionaries, main_task_name)

    def __getitem__(self, name):
        return self.dictionaries[name]

    @property
    def _tgt_dict(self):
        return self.dictionaries[self.main_task_name]

    def pad(self):
        return self._tgt_dict.pad()

    def eos(self):
        return self._tgt_dict.eos()

    def unk(self):
        return self._tgt_dict.unk()

    def string(self, *args, **kwargs):
        return self._tgt_dict.string(*args, **kwargs)

    def unk_string(self):
        return self._tgt_dict.unk_string()

    def encode_line(self, *args, **kwargs):
        return self._tgt_dict.encode_line(*args, **kwargs)


class JointSourceDictionary:
    def __init__(self, dictionaries, main_task_name):
        self.dictionaries = dictionaries
        self.main_task_name = main_task_name

    @classmethod
    def build_dictionary(cls, tasks, main_task_name):
        dictionaries = {
            _name: _task.source_dictionary
            for _name, _task in tasks.items()
        }
        return cls(dictionaries, main_task_name)

    def __getitem__(self, name):
        return self.dictionaries[name]

    @property
    def _src_dict(self):
        return self.dictionaries[self.main_task_name]

    def string(self, *args, **kwargs):
        if self._src_dict is not None:
            return self._src_dict.string(*args, **kwargs)
        else:
            return ""


class ShiftModelContext:
    def __init__(self, joint_task, task_name, model):
        self.joint_task = joint_task
        self.task_name = task_name
        if isinstance(model, JointModel):
            self.model = model
        elif '_DistributedFairseqModel' in str(model.__class__):
            self.model = model.module
        else:
            raise Exception()

    def __enter__(self):
        self.joint_task.shift_model(self.task_name, self.model)

    def __exit__(self, type, value, traceback):
        self.joint_task.shift_model_back(self.task_name, self.model)


class JointConfig:
    def __init__(self, root_config, configs):
        self.root_config = root_config
        self.configs = configs
        self.main_task_name = None

    @property
    def _config(self):
        return self.configs[self.main_task_name]
