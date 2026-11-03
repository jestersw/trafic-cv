use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use traffic_core::tracker::{ByteTrack, Sort, Tracker};
use traffic_core::Det;

fn to_dets(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<Det>> {
    let v = a.as_array();
    if v.ncols() != 6 && v.nrows() > 0 {
        return Err(PyValueError::new_err("dets must have shape (N, 6)"));
    }
    Ok(v.rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]])
        .collect())
}

fn to_array<'py>(py: Python<'py>, rows: Vec<[f64; 7]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = rows.len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    numpy::PyArray1::from_vec(py, flat).reshape([n, 7])
}

#[pyclass(name = "ByteTrack", unsendable)]
struct PyByteTrack(ByteTrack);

#[pymethods]
impl PyByteTrack {
    #[new]
    #[pyo3(signature = (track_thresh=0.5, low_thresh=0.1, new_track_thresh=0.6, match_iou=0.2, low_match_iou=0.5, unconfirmed_match_iou=0.3, track_buffer=30))]
    fn new(
        track_thresh: f64,
        low_thresh: f64,
        new_track_thresh: f64,
        match_iou: f64,
        low_match_iou: f64,
        unconfirmed_match_iou: f64,
        track_buffer: u32,
    ) -> Self {
        let mut t = ByteTrack::default();
        t.track_thresh = track_thresh;
        t.low_thresh = low_thresh;
        t.new_track_thresh = new_track_thresh;
        t.match_iou = match_iou;
        t.low_match_iou = low_match_iou;
        t.unconfirmed_match_iou = unconfirmed_match_iou;
        t.track_buffer = track_buffer;
        Self(t)
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        dets: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = to_dets(&dets)?;
        to_array(py, self.0.update(&d))
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

#[pyclass(name = "SORT", unsendable)]
struct PySort(Sort);

#[pymethods]
impl PySort {
    #[new]
    #[pyo3(signature = (max_age=1, min_hits=3, iou_threshold=0.3, min_score=0.25))]
    fn new(max_age: u32, min_hits: u32, iou_threshold: f64, min_score: f64) -> Self {
        let mut t = Sort::default();
        t.max_age = max_age;
        t.min_hits = min_hits;
        t.iou_threshold = iou_threshold;
        t.min_score = min_score;
        Self(t)
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        dets: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = to_dets(&dets)?;
        to_array(py, self.0.update(&d))
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

#[pymodule]
fn traffic_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyByteTrack>()?;
    m.add_class::<PySort>()?;
    Ok(())
}
