#![allow(clippy::needless_range_loop)]

pub mod counter;
pub mod geometry;
pub mod hungarian;
pub mod kalman;
pub mod tracker;

pub type Det = [f64; 6];
pub type TrackRow = [f64; 7];
