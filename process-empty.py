from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
from loguru import logger

SIZE_ARRAY_DIM = 512
COUNT_PER_MSGPACK_UPPERBOUND = 50000

def func():
    return np.empty((50000, 512), dtype=np.float32)


if __name__ == "__main__":
    N_tasks = 32
    N_workers = 8

    pool = ProcessPoolExecutor(max_workers=N_workers)

    logger.info("With Python / process pool:")
    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(func))

    for future in tqdm(futures):
        _ = future.result()
