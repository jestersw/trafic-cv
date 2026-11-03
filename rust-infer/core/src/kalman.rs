type M7 = [[f64; 7]; 7];

fn eye7() -> M7 {
    let mut m = [[0.0; 7]; 7];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn mul7(a: &M7, b: &M7) -> M7 {
    let mut r = [[0.0; 7]; 7];
    for i in 0..7 {
        for k in 0..7 {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..7 {
                r[i][j] += aik * b[k][j];
            }
        }
    }
    r
}

fn transpose7(a: &M7) -> M7 {
    let mut r = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            r[j][i] = a[i][j];
        }
    }
    r
}

fn inv4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut a = [[0.0; 8]; 4];
    for i in 0..4 {
        a[i][..4].copy_from_slice(&m[i]);
        a[i][4 + i] = 1.0;
    }
    for c in 0..4 {
        let piv = (c..4)
            .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
            .unwrap();
        a.swap(c, piv);
        let d = a[c][c];
        for v in a[c].iter_mut() {
            *v /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = a[r][c];
                if f != 0.0 {
                    for k in 0..8 {
                        a[r][k] -= f * a[c][k];
                    }
                }
            }
        }
    }
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        out[i].copy_from_slice(&a[i][4..]);
    }
    out
}

fn f_matrix() -> M7 {
    let mut f = eye7();
    f[0][4] = 1.0;
    f[1][5] = 1.0;
    f[2][6] = 1.0;
    f
}

fn q_matrix() -> M7 {
    let mut q = eye7();
    q[4][4] = 0.01;
    q[5][5] = 0.01;
    q[6][6] = 0.0001;
    q
}

const R_DIAG: [f64; 4] = [1.0, 1.0, 10.0, 10.0];

pub fn xyxy_to_z(b: &[f64]) -> [f64; 4] {
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    [b[0] + w / 2.0, b[1] + h / 2.0, w * h, w / h.max(1e-6)]
}

pub fn x_to_xyxy(x: &[f64; 7]) -> [f64; 4] {
    let s = x[2].max(0.0);
    let r = x[3];
    let w = if r > 0.0 { (s * r).sqrt() } else { 0.0 };
    let h = if w > 0.0 { s / w } else { 0.0 };
    [
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]
}

#[derive(Clone, Debug)]
pub struct KalmanBoxFilter {
    pub x: [f64; 7],
    pub p: M7,
}

impl KalmanBoxFilter {
    pub fn new(b: &[f64]) -> Self {
        let mut x = [0.0; 7];
        x[..4].copy_from_slice(&xyxy_to_z(b));
        let mut p = eye7();
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = if i >= 4 { 10_000.0 } else { 10.0 };
        }
        Self { x, p }
    }

    pub fn predict(&mut self) -> [f64; 4] {
        if self.x[2] + self.x[6] <= 0.0 {
            self.x[6] = 0.0;
        }
        let f = f_matrix();
        let mut nx = [0.0; 7];
        for i in 0..7 {
            nx[i] = (0..7).map(|j| f[i][j] * self.x[j]).sum();
        }
        self.x = nx;
        let mut p = mul7(&mul7(&f, &self.p), &transpose7(&f));
        let q = q_matrix();
        for i in 0..7 {
            for j in 0..7 {
                p[i][j] += q[i][j];
            }
        }
        self.p = p;
        x_to_xyxy(&self.x)
    }

    pub fn update(&mut self, b: &[f64]) {
        let z = xyxy_to_z(b);
        let y: Vec<f64> = (0..4).map(|i| z[i] - self.x[i]).collect();
        let mut s = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = self.p[i][j];
            }
            s[i][i] += R_DIAG[i];
        }
        let si = inv4(s);
        let mut k = [[0.0; 4]; 7];
        for i in 0..7 {
            for j in 0..4 {
                k[i][j] = (0..4).map(|l| self.p[i][l] * si[l][j]).sum();
            }
        }
        for i in 0..7 {
            self.x[i] += (0..4).map(|j| k[i][j] * y[j]).sum::<f64>();
        }
        let mut ikh = eye7();
        for i in 0..7 {
            for j in 0..4 {
                ikh[i][j] -= k[i][j];
            }
        }
        self.p = mul7(&ikh, &self.p);
    }

    pub fn bbox(&self) -> [f64; 4] {
        x_to_xyxy(&self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_tracking() {
        let b = [100.0, 50.0, 140.0, 80.0];
        let mut kf = KalmanBoxFilter::new(&b);
        let r = kf.bbox();
        for i in 0..4 {
            assert!((r[i] - b[i]).abs() < 1e-9);
        }
        for t in 1..20 {
            kf.predict();
            let s = t as f64 * 5.0;
            kf.update(&[100.0 + s, 50.0, 140.0 + s, 80.0]);
        }
        let p = kf.predict();
        assert!((p[0] - 200.0).abs() < 3.0, "{p:?}");
    }
}
