import os
# need to set OMP_NUM_THREADS=1 to avoid MKL/OpenBlas using internal multithreading to confound the measurement
# also much faster overall due to less thread contention (only python threads)
os.environ["OMP_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

def func():
    for _ in range(10):
        np.dot(np.random.rand(1000, 1000), np.random.rand(1000, 1000))


if __name__ == "__main__":
    N_tasks = 32
    N_workers = 8

    pool = ThreadPoolExecutor(max_workers=N_workers)

    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(func))

    for future in tqdm(futures):
        _ = future.result()
