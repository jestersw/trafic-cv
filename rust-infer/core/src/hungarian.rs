pub fn linear_sum_assignment(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    if m == 0 {
        return Vec::new();
    }
    if n > m {
        let t: Vec<Vec<f64>> = (0..m)
            .map(|j| (0..n).map(|i| cost[i][j]).collect())
            .collect();
        let mut r: Vec<(usize, usize)> = linear_sum_assignment(&t)
            .into_iter()
            .map(|(a, b)| (b, a))
            .collect();
        r.sort_unstable();
        return r;
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out: Vec<(usize, usize)> = (1..=m)
        .filter(|&j| p[j] != 0)
        .map(|j| (p[j] - 1, j - 1))
        .collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square() {
        let c = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let r = linear_sum_assignment(&c);
        let total: f64 = r.iter().map(|&(i, j)| c[i][j]).sum();
        assert_eq!(total, 5.0);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn rectangular() {
        let c = vec![vec![1.0, 9.0], vec![9.0, 1.0], vec![5.0, 5.0]];
        let r = linear_sum_assignment(&c);
        assert_eq!(r, vec![(0, 0), (1, 1)]);
    }
}
