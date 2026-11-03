use crate::geometry::iou_matrix;
use crate::hungarian::linear_sum_assignment;
use crate::kalman::KalmanBoxFilter;
use crate::{Det, TrackRow};

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub kf: KalmanBoxFilter,
    pub score: f64,
    pub class_votes: Vec<(i64, u32)>,
    pub hits: u32,
    pub hit_streak: u32,
    pub age: u32,
    pub time_since_update: u32,
    pub activated: bool,
}

impl Track {
    pub fn new(id: u64, det: &Det) -> Self {
        Self {
            id,
            kf: KalmanBoxFilter::new(&det[..4]),
            score: det[4],
            class_votes: vec![(det[5] as i64, 1)],
            hits: 1,
            hit_streak: 1,
            age: 0,
            time_since_update: 0,
            activated: false,
        }
    }

    pub fn predict(&mut self) -> [f64; 4] {
        let b = self.kf.predict();
        self.age += 1;
        if self.time_since_update > 0 {
            self.hit_streak = 0;
        }
        self.time_since_update += 1;
        b
    }

    pub fn update(&mut self, det: &Det) {
        self.kf.update(&det[..4]);
        self.score = det[4];
        let c = det[5] as i64;
        match self.class_votes.iter_mut().find(|(k, _)| *k == c) {
            Some((_, n)) => *n += 1,
            None => self.class_votes.push((c, 1)),
        }
        self.hits += 1;
        self.hit_streak += 1;
        self.time_since_update = 0;
    }

    pub fn class_id(&self) -> i64 {
        let mut best = self.class_votes[0];
        for &(c, n) in &self.class_votes[1..] {
            if n > best.1 {
                best = (c, n);
            }
        }
        best.0
    }

    pub fn row(&self) -> TrackRow {
        let b = self.kf.bbox();
        [
            b[0],
            b[1],
            b[2],
            b[3],
            self.id as f64,
            self.score,
            self.class_id() as f64,
        ]
    }
}

pub type Matches = (Vec<(usize, usize)>, Vec<usize>, Vec<usize>);

pub fn associate(tracks: &[[f64; 4]], dets: &[[f64; 4]], thr: f64) -> Matches {
    let (nt, nd) = (tracks.len(), dets.len());
    if nt == 0 || nd == 0 {
        return (Vec::new(), (0..nt).collect(), (0..nd).collect());
    }
    let iou = iou_matrix(tracks, dets);
    let cost: Vec<Vec<f64>> = iou.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
    let matches: Vec<(usize, usize)> = linear_sum_assignment(&cost)
        .into_iter()
        .filter(|&(r, c)| iou[r][c] >= thr)
        .collect();
    let um_t = (0..nt)
        .filter(|i| !matches.iter().any(|m| m.0 == *i))
        .collect();
    let um_d = (0..nd)
        .filter(|j| !matches.iter().any(|m| m.1 == *j))
        .collect();
    (matches, um_t, um_d)
}

fn boxes(dets: &[Det]) -> Vec<[f64; 4]> {
    dets.iter().map(|d| [d[0], d[1], d[2], d[3]]).collect()
}

pub trait Tracker {
    fn update(&mut self, dets: &[Det]) -> Vec<TrackRow>;
    fn reset(&mut self);
}

#[derive(Clone, Debug)]
pub struct ByteTrack {
    pub track_thresh: f64,
    pub low_thresh: f64,
    pub new_track_thresh: f64,
    pub match_iou: f64,
    pub low_match_iou: f64,
    pub unconfirmed_match_iou: f64,
    pub track_buffer: u32,
    tracked: Vec<Track>,
    lost: Vec<Track>,
    frame_count: u64,
    next_id: u64,
}

impl Default for ByteTrack {
    fn default() -> Self {
        Self {
            track_thresh: 0.5,
            low_thresh: 0.1,
            new_track_thresh: 0.6,
            match_iou: 0.2,
            low_match_iou: 0.5,
            unconfirmed_match_iou: 0.3,
            track_buffer: 30,
            tracked: Vec::new(),
            lost: Vec::new(),
            frame_count: 0,
            next_id: 1,
        }
    }
}

impl ByteTrack {
    fn new_track(&mut self, det: &Det) -> Track {
        let mut t = Track::new(self.next_id, det);
        self.next_id += 1;
        t.activated = self.frame_count == 1;
        t
    }
}

impl Tracker for ByteTrack {
    fn reset(&mut self) {
        self.tracked.clear();
        self.lost.clear();
        self.frame_count = 0;
        self.next_id = 1;
    }

    fn update(&mut self, dets: &[Det]) -> Vec<TrackRow> {
        self.frame_count += 1;
        let high: Vec<Det> = dets
            .iter()
            .copied()
            .filter(|d| d[4] >= self.track_thresh)
            .collect();
        let low: Vec<Det> = dets
            .iter()
            .copied()
            .filter(|d| d[4] >= self.low_thresh && d[4] < self.track_thresh)
            .collect();

        let (confirmed, mut unconfirmed): (Vec<Track>, Vec<Track>) =
            std::mem::take(&mut self.tracked)
                .into_iter()
                .partition(|t| t.activated);
        let n_conf = confirmed.len();
        let mut pool: Vec<Option<Track>> = confirmed
            .into_iter()
            .chain(std::mem::take(&mut self.lost))
            .map(Some)
            .collect();
        for t in pool.iter_mut().flatten() {
            t.predict();
        }
        for t in unconfirmed.iter_mut() {
            t.predict();
        }

        let pool_boxes: Vec<[f64; 4]> =
            pool.iter().map(|t| t.as_ref().unwrap().kf.bbox()).collect();
        let (m1, um_t1, um_d1) = associate(&pool_boxes, &boxes(&high), self.match_iou);
        let mut next: Vec<Track> = Vec::new();
        for (ti, di) in m1 {
            let mut t = pool[ti].take().unwrap();
            t.update(&high[di]);
            next.push(t);
        }

        let mut remain: Vec<Option<Track>> = Vec::new();
        let mut lost_pool: Vec<Track> = Vec::new();
        for i in um_t1 {
            let t = pool[i].take().unwrap();
            if i < n_conf {
                remain.push(Some(t));
            } else {
                lost_pool.push(t);
            }
        }
        let remain_boxes: Vec<[f64; 4]> = remain
            .iter()
            .map(|t| t.as_ref().unwrap().kf.bbox())
            .collect();
        let (m2, um_t2, _) = associate(&remain_boxes, &boxes(&low), self.low_match_iou);
        for (ti, di) in m2 {
            let mut t = remain[ti].take().unwrap();
            t.update(&low[di]);
            next.push(t);
        }
        let mut still_lost: Vec<Track> = um_t2
            .into_iter()
            .map(|i| remain[i].take().unwrap())
            .collect();
        still_lost.extend(lost_pool);

        let left_high: Vec<Det> = um_d1.iter().map(|&i| high[i]).collect();
        let unc_boxes: Vec<[f64; 4]> = unconfirmed.iter().map(|t| t.kf.bbox()).collect();
        let (m3, _, um_d3) = associate(&unc_boxes, &boxes(&left_high), self.unconfirmed_match_iou);
        let mut unconfirmed: Vec<Option<Track>> = unconfirmed.into_iter().map(Some).collect();
        for (ti, di) in m3 {
            let mut t = unconfirmed[ti].take().unwrap();
            t.update(&left_high[di]);
            t.activated = true;
            next.push(t);
        }
        for di in um_d3 {
            if left_high[di][4] >= self.new_track_thresh {
                let t = self.new_track(&left_high[di]);
                next.push(t);
            }
        }
        for t in next.iter_mut() {
            if t.hits >= 2 {
                t.activated = true;
            }
        }
        self.tracked = next;
        let buf = self.track_buffer;
        self.lost = still_lost
            .into_iter()
            .filter(|t| t.time_since_update <= buf)
            .collect();
        self.tracked
            .iter()
            .filter(|t| t.activated)
            .map(|t| t.row())
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Sort {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f64,
    pub min_score: f64,
    tracks: Vec<Track>,
    frame_count: u64,
    next_id: u64,
}

impl Default for Sort {
    fn default() -> Self {
        Self {
            max_age: 1,
            min_hits: 3,
            iou_threshold: 0.3,
            min_score: 0.25,
            tracks: Vec::new(),
            frame_count: 0,
            next_id: 1,
        }
    }
}

impl Tracker for Sort {
    fn reset(&mut self) {
        self.tracks.clear();
        self.frame_count = 0;
        self.next_id = 1;
    }

    fn update(&mut self, dets: &[Det]) -> Vec<TrackRow> {
        self.frame_count += 1;
        let dets: Vec<Det> = dets
            .iter()
            .copied()
            .filter(|d| d[4] >= self.min_score)
            .collect();
        let mut predicted = Vec::new();
        let mut kept = Vec::new();
        for mut t in std::mem::take(&mut self.tracks) {
            let b = t.predict();
            if b.iter().all(|v| v.is_finite()) {
                predicted.push(b);
                kept.push(t);
            }
        }
        self.tracks = kept;
        let (matches, _, um_d) = associate(&predicted, &boxes(&dets), self.iou_threshold);
        for (ti, di) in matches {
            self.tracks[ti].update(&dets[di]);
        }
        for di in um_d {
            self.tracks.push(Track::new(self.next_id, &dets[di]));
            self.next_id += 1;
        }
        let out = self
            .tracks
            .iter()
            .filter(|t| {
                t.time_since_update < 1
                    && (t.hit_streak >= self.min_hits || self.frame_count <= self.min_hits as u64)
            })
            .map(|t| t.row())
            .collect();
        let max_age = self.max_age;
        self.tracks.retain(|t| t.time_since_update <= max_age);
        out
    }
}

pub fn build(name: &str) -> Option<Box<dyn Tracker + Send>> {
    match name {
        "bytetrack" => Some(Box::new(ByteTrack::default())),
        "sort" => Some(Box::new(Sort::default())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving(n: usize) -> Vec<Vec<Det>> {
        (0..n)
            .map(|f| {
                let s = f as f64 * 5.0;
                vec![
                    [100.0 + s, 100.0, 140.0 + s, 130.0, 0.9, 0.0],
                    [500.0 - s, 300.0, 560.0 - s, 350.0, 0.8, 1.0],
                ]
            })
            .collect()
    }

    #[test]
    fn bytetrack_keeps_ids() {
        let mut bt = ByteTrack::default();
        let mut ids = std::collections::BTreeSet::new();
        for dets in moving(30) {
            for r in bt.update(&dets) {
                ids.insert(r[4] as u64);
            }
        }
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn bytetrack_recovers_low_score() {
        let mut bt = ByteTrack::default();
        let frames = moving(20);
        let mut ids = std::collections::BTreeSet::new();
        for (f, mut dets) in frames.into_iter().enumerate() {
            if (8..12).contains(&f) {
                dets[0][4] = 0.3;
            }
            for r in bt.update(&dets) {
                ids.insert(r[4] as u64);
            }
        }
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn sort_tracks() {
        let mut s = Sort::default();
        let mut last = Vec::new();
        for dets in moving(10) {
            last = s.update(&dets);
        }
        assert_eq!(last.len(), 2);
    }
}
