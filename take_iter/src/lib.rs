use numpy::ndarray::{s, ArrayViewMutD};
use numpy::PyReadwriteArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};

static SIZE_ARRAY_DIM: usize = 512;

fn copy_array(vectors: &mut ArrayViewMutD<'_, f32>, bytes_vector: &[u8], idx: usize) {
    if bytes_vector.len() == SIZE_ARRAY_DIM * 4 {
        // f32 from msgpack
        // copy bytes in f32 le format to vectors at idx
        let mut arrays_slice = vectors.slice_mut(s![idx, ..]);
        for (dst, src) in arrays_slice.iter_mut().zip(
            bytes_vector
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap())),
        ) {
            *dst = src;
        }
    } else {
        println!("array size does not match!");
    };
}

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn take_iter<'py>(
        py: Python<'_>,
        iter: &PyIterator,
        mut np_vectors: PyReadwriteArrayDyn<f32>,
    ) -> PyResult<usize> {
        // First collect bytes in a Rust-native vector.
        // We can’t release the GIL here because we are dealing with a Python object.
        let mut raw_list: Vec<&[u8]> = vec![];
        for item in iter {
            raw_list.push(item?.downcast::<PyBytes>()?.as_bytes());
        }
        let mut vectors = np_vectors.as_array_mut();

        // Bytes are read as f32 arrays and copied into the passed in NumPy array.
        // We release the GIL here so other Python threads can run in true parallelism.
        py.allow_threads(|| {
            let mut idx = 0;
            for &bytes_vector in raw_list.iter() {
                copy_array(&mut vectors, bytes_vector, idx);
                idx += 1;
            }
            Ok(idx)
        })
    }

    #[pyfn(m)]
    fn func<'py>(_py: Python<'_>) -> PyResult<()> {
        Ok(())
    }

    Ok(())
}
