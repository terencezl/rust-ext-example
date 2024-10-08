use anyhow::Result;
use numpy::ndarray::{s, ArrayViewMut1};
use numpy::PyReadwriteArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use std::fs::File;
use std::io::{BufReader, Read};

const SIZE_ARRAY_DIM: usize = 512;
const F32_SIZE: usize = 4;

fn copy_array(src_bytes_vector: &[u8], dst_vector: &mut ArrayViewMut1<f32>) -> Result<()> {
    if src_bytes_vector.len() == dst_vector.len() * F32_SIZE {
        // f32 from msgpack

        // copy bytes in f32 le format to dst_vector
        // safe but slower
        // for (dst, src) in dst_vector.iter_mut().zip(
        //     src_bytes_vector
        //         .chunks_exact(F32_SIZE)
        //         .map(|x| f32::from_le_bytes(x.try_into().unwrap())),
        // ) {
        //     *dst = src;
        // }

        // unsafe but correct & faster
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_bytes_vector.as_ptr() as *const f32,
                dst_vector.as_mut_ptr(),
                dst_vector.len(),
            );
        }
        return Ok(());
    } else {
        anyhow::bail!(
            "Array size is {}, does not match {}!",
            src_bytes_vector.len(),
            dst_vector.len() * F32_SIZE
        );
    };
}

struct RawListIterator {
    reader: BufReader<File>,
}

impl RawListIterator {
    fn new(filepath: &str) -> Result<Self> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        Ok(RawListIterator { reader })
    }
}

impl Iterator for RawListIterator {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        match rmp::decode::read_bin_len(&mut self.reader) {
            Ok(len) => {
                let mut buffer = vec![0u8; len as usize];
                match self.reader.read_exact(&mut buffer) {
                    Ok(_) => Some(Ok(buffer)),
                    Err(e) => Some(Err(e.into())),
                }
            }
            Err(rmp::decode::ValueReadError::InvalidMarkerRead(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                None
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

#[pymodule]
fn rust_ext(m: &Bound<PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn take_iter(
        py: Python,
        iter: Bound<PyIterator>,
        mut np_vectors: PyReadwriteArray2<f32>,
    ) -> PyResult<usize> {
        // First collect bytes into a Rust-native vector.
        // We can't release the GIL here because we are dealing with a Python object.
        let mut raw_list: Vec<Vec<u8>> = vec![];
        for item in iter {
            raw_list.push(item?.downcast::<PyBytes>()?.as_bytes().to_vec());
        }

        let mut vectors = np_vectors.as_array_mut();

        // Bytes are read as f32 arrays and copied into the passed in NumPy array.
        // We release the GIL here so other Python threads can run in true parallelism.
        py.allow_threads(|| {
            if raw_list.len() > vectors.dim().0 {
                return Err(PyValueError::new_err("Too many items in iterator!"));
            }
            if vectors.dim().1 != SIZE_ARRAY_DIM {
                return Err(PyValueError::new_err(format!(
                    "2D NumPy array has {} columns, does not match {}!",
                    vectors.dim().1,
                    SIZE_ARRAY_DIM
                )));
            }

            let mut idx = 0;
            for src_bytes_vector in raw_list {
                let mut dst_vector = vectors.slice_mut(s![idx, ..]);
                if let Err(e) = copy_array(&src_bytes_vector, &mut dst_vector) {
                    eprintln!("Error: {}. Skipping...", e);
                    continue;
                }
                idx += 1;
            }
            Ok(idx)
        })
    }

    #[pyfn(m)]
    fn take_filepath(
        py: Python,
        filepath: &str,
        mut np_vectors: PyReadwriteArray2<f32>,
    ) -> PyResult<usize> {
        let mut vectors = np_vectors.as_array_mut();

        // Bytes are read as f32 arrays and copied into the passed in NumPy array.
        // We release the GIL here so other Python threads can run in true parallelism.
        py.allow_threads(|| {
            let raw_list_iter = RawListIterator::new(filepath)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if vectors.dim().1 != SIZE_ARRAY_DIM {
                return Err(PyValueError::new_err(format!(
                    "2D NumPy array has {} columns, does not match {}!",
                    vectors.dim().1,
                    SIZE_ARRAY_DIM
                )));
            }

            let mut idx = 0;
            for src_bytes_vector in raw_list_iter {
                let src_bytes_vector =
                    src_bytes_vector.map_err(|e| PyValueError::new_err(e.to_string()))?;
                if idx >= vectors.dim().0 {
                    return Err(PyValueError::new_err("Too many items in iterator!"));
                }
                let mut dst_vector = vectors.slice_mut(s![idx, ..]);
                if let Err(e) = copy_array(&src_bytes_vector, &mut dst_vector) {
                    eprintln!("Error: {}. Skipping...", e);
                    continue;
                }
                idx += 1;
            }
            Ok(idx)
        })
    }

    Ok(())
}
