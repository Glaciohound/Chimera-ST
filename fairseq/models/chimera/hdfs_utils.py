import torch
import io
import os
import shutil
import subprocess
from contextlib import contextmanager
import numpy as np

HADOOP_BIN = 'PATH=/usr/bin/:$PATH hdfs'

@contextmanager
def hopen(hdfs_path, mode):
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def torchHLoad(filepath: str, **kwargs):
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict


def torchHSave(obj, filepath: str, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


def numpyHSave(array, filepath: str, **kwargs):
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "w") as writer:
            np.save(writer, array, **kwargs)
    else:
        np.save(filepath, array, **kwargs)


def makeHpath(filepath: str):
    if not filepath.startswith("hdfs://"):
        os.makedirs(filepath, exist_ok=True)
    else:
        RunCmd(f'{HADOOP_BIN} dfs -mkdir -p {filepath}')


def rmHpath(filepath: str):
    if not filepath.startswith("hdfs://"):
        shutil.rmtree(filepath)
    else:
        RunCmd(f"{HADOOP_BIN} dfs -rm -r {filepath}")


def PutHDFS(local: str, remote: str):
    makeHpath(remote)
    RunCmd(f"{HADOOP_BIN} dfs -put {local} {remote}")


def HPathExists(filepath: str):
    if not filepath.startswith("hdfs://"):
        return os.path.exists(filepath)

        pass


# def TorchHLoad(filepath: str, **kwargs):
#     import torch, tensorflow as tf
#     if not filepath.startswith("hdfs://"):
#         return torch.load(filepath, **kwargs)
#     else:
#         with tf.io.gfile.GFile(filepath, 'rb') as reader:
#             return torch.load(io.BytesIO(reader.read()), **kwargs)


# def TorchHSave(obj, filepath: str, **kwargs):
#     import torch, tensorflow as tf
#     if filepath.startswith("hdfs://"):
#         with tf.io.gfile.GFile(filepath, 'wb') as f:
#             buffer = io.BytesIO()
#             torch.save(obj, buffer, **kwargs)
#             f.write(buffer.getvalue())
#     else:
#         torch.save(obj, filepath, **kwargs)


# def PutHDFS(local: str, remote: str):
#     import tensorflow as tf
#     assert remote.startswith('hdfs://')
#     if not tf.io.gfile.exists(remote):
#         tf.io.gfile.makedirs(remote)
#     RunCmd(f'{HADOOP_BIN} dfs -put {local} {remote}')


def GetHDFS(remote: str, local: str):
    assert remote.startswith('hdfs://')
    os.makedirs(local, exist_ok=True)
    RunCmd(f'{HADOOP_BIN} dfs -get {remote} {local}')


def RunCmd(command):
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res, err = pipe.communicate()
    res = res.decode('utf-8')
    err = err.decode('utf-8')
    return res, err