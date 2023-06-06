# Rust Extension Example

This is an example of a Rust extension for Python, using PyO3 and rust-numpy. See the blog [post](https://terencezl.github.io/blog/2023/06/06/a-week-of-pyo3-rust-numpy/) for a more detailed walkthrough.

## Files

```
create-msgpack.py - script to create the input msgpack file
process-empty.py - script that demonstrates the (de-)serialization cost of big NumPy arrays through process workers
rust-ext-example.py - script that demonstrates the main example
src/ - Rust source code
```

## Setup

Create a venv and install from `requirements.txt`. Then run `create-msgpack.py` to create the input msgpack file.

For the Rust extension, run `(cd take_iter && maturin develop --release)`. This will build the extension and install it in the venv.

Run `python rust-ext-example.py` to run the example.
