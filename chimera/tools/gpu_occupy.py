import torch
import time
# from multiprocessing.dummy import Pool as ThreadPool

n_cuda = torch.cuda.device_count()
assert n_cuda > 0

tensors = [
    torch.randn(3000, 3000).to(torch.device(f'cuda:{i}'))
    for i in range(n_cuda)
]


def compute(i):
    tensor = tensors[i]
    torch.mm(tensor, tensor)


while True:
    # pool = ThreadPool(4)
    # pool.map(compute, list(range(n_cuda)))
    # torch.cuda.empty_cache()
    for i in range(n_cuda):
        compute(i)
    time.sleep(0.01)
