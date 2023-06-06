import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import msgpack
from tqdm import tqdm
from loguru import logger
import rust_ext

SIZE_ARRAY_DIM = 512
COUNT_PER_MSGPACK_UPPERBOUND = 50000

def iterate_msgpack(filename):
    with open(filename, "rb") as handle:
        unpacker = msgpack.Unpacker(handle)
        try:
            for item in unpacker:
                yield item

        except Exception as e:
            logger.error(e)


def take_iter_py(iterator):
    # np_vectors = []
    # for bytes_vector in iterator:
    #     vector = np.frombuffer(bytes_vector, dtype=np.float32).reshape(SIZE_ARRAY_DIM)
    #     np_vectors.append(vector)
    # np_vectors = np.array(np_vectors)

    np_vectors = np.empty((COUNT_PER_MSGPACK_UPPERBOUND, SIZE_ARRAY_DIM), dtype=np.float32)
    idx = 0
    for bytes_vector in iterator:
        vector = np.frombuffer(bytes_vector, dtype=np.float32).reshape(SIZE_ARRAY_DIM)
        np_vectors[idx] = vector
        idx += 1
    np_vectors = np_vectors[:idx]

    return np_vectors


def process_py(filepath):
    t = time.time()
    np_vectors = take_iter_py(iterate_msgpack(filepath))

    # print(f"{np_vectors[-1, :5]}")
    # logger.debug(f"take_iter_py took {time.time() - t:.2f} seconds.")
    return np_vectors


def process_rs(filepath):
    t = time.time()
    np_vectors = np.empty((COUNT_PER_MSGPACK_UPPERBOUND, SIZE_ARRAY_DIM), dtype=np.float32)
    count = rust_ext.take_iter(iterate_msgpack(filepath), np_vectors)
    np_vectors = np_vectors[:count]

    # print(f"{np_vectors[-1, :5]}")
    # logger.debug(f"rust_ext.take_iter took {time.time() - t:.2f} seconds.")
    return np_vectors


if __name__ == "__main__":
    filepath = "bytes_vectors.msgpack"

    N_tasks = 32
    N_workers = 8

    pool = ProcessPoolExecutor(max_workers=N_workers)

    logger.info("With native Python / process pool:")
    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(process_py, filepath))

    for future in tqdm(futures):
        _ = future.result()

    logger.info("With Rust extension / process pool:")
    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(process_rs, filepath))

    for future in tqdm(futures):
        _ = future.result()

    pool = ThreadPoolExecutor(max_workers=N_workers)

    logger.info("With native Python / thread pool:")
    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(process_py, filepath))

    for future in tqdm(futures):
        _ = future.result()

    logger.info("With Rust extension / thread pool:")
    futures = []
    for _ in range(N_tasks):
        futures.append(pool.submit(process_rs, filepath))

    for future in tqdm(futures):
        _ = future.result()
