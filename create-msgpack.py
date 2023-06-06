import msgpack
import numpy as np

SIZE_ARRAY_DIM = 512
COUNT_PER_MSGPACK_UPPERBOUND = 50000

if __name__ == "__main__":
    with open("bytes_vectors.msgpack", "wb") as f:
        for i in range(COUNT_PER_MSGPACK_UPPERBOUND):
            vector = np.random.rand(SIZE_ARRAY_DIM).astype(np.float32)
            msgpack.pack(vector.tobytes(), f)
