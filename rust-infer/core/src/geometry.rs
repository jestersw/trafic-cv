pub type Point = (f64, f64);

pub fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

pub fn signed_distance(p1: Point, p2: Point, q: Point) -> f64 {
    let len = ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
    cross(p1, p2, q) / len.max(1e-9)
}

pub fn projection(p1: Point, p2: Point, q: Point) -> f64 {
    let (dx, dy) = (p2.0 - p1.0, p2.1 - p1.1);
    ((q.0 - p1.0) * dx + (q.1 - p1.1) * dy) / (dx * dx + dy * dy).max(1e-9)
}

pub fn bottom_center(b: &[f64]) -> Point {
    ((b[0] + b[2]) / 2.0, b[3])
}

pub fn iou(a: &[f64], b: &[f64]) -> f64 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter).max(1e-9)
}

pub fn iou_matrix(a: &[[f64; 4]], b: &[[f64; 4]]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|x| b.iter().map(|y| iou(x, y)).collect())
        .collect()
}
