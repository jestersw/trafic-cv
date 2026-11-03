use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::geometry::{bottom_center, projection, signed_distance, Point};
use crate::TrackRow;

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub frame: u64,
    pub time_s: f64,
    pub track_id: u64,
    pub class_id: i64,
    pub class_name: String,
    pub event: String,
    pub detail: String,
    pub source: String,
}

#[derive(Clone, Debug)]
pub struct LineCounter {
    pub p1: Point,
    pub p2: Point,
    pub name: String,
    pub names: Vec<String>,
    pub count_once: bool,
    pub margin: f64,
    pub counts: BTreeMap<String, [u64; 2]>,
    side: HashMap<u64, i8>,
    counted: HashSet<u64>,
}

impl LineCounter {
    pub fn new(p1: Point, p2: Point, names: Vec<String>, name: &str) -> Self {
        Self {
            p1,
            p2,
            name: name.to_string(),
            names,
            count_once: true,
            margin: 0.0,
            counts: BTreeMap::new(),
            side: HashMap::new(),
            counted: HashSet::new(),
        }
    }

    pub fn total_in(&self) -> u64 {
        self.counts.values().map(|c| c[0]).sum()
    }

    pub fn total_out(&self) -> u64 {
        self.counts.values().map(|c| c[1]).sum()
    }

    fn class_name(&self, cls: i64) -> String {
        usize::try_from(cls)
            .ok()
            .and_then(|i| self.names.get(i).cloned())
            .unwrap_or_else(|| cls.to_string())
    }

    pub fn update(&mut self, tracks: &[TrackRow], frame: u64, fps: f64) -> Vec<Event> {
        let mut events = Vec::new();
        for row in tracks {
            let tid = row[4] as u64;
            let cls = row[6] as i64;
            let cur = bottom_center(row);
            let d = signed_distance(self.p1, self.p2, cur);
            let t = projection(self.p1, self.p2, cur);
            if d.abs() <= self.margin || !(0.0..=1.0).contains(&t) {
                continue;
            }
            let s: i8 = if d > 0.0 { 1 } else { -1 };
            let prev = self.side.insert(tid, s);
            match prev {
                Some(p) if p != s => {}
                _ => continue,
            }
            if self.count_once && self.counted.contains(&tid) {
                continue;
            }
            let dir = if s > 0 { "in" } else { "out" };
            let cname = self.class_name(cls);
            let c = self.counts.entry(cname.clone()).or_insert([0, 0]);
            c[if s > 0 { 0 } else { 1 }] += 1;
            self.counted.insert(tid);
            events.push(Event {
                frame,
                time_s: frame as f64 / fps,
                track_id: tid,
                class_id: cls,
                class_name: cname,
                event: "line_cross".into(),
                detail: dir.into(),
                source: self.name.clone(),
            });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trk(id: f64, cx: f64, bottom: f64, cls: f64) -> TrackRow {
        [cx - 20.0, bottom - 30.0, cx + 20.0, bottom, id, 0.9, cls]
    }

    fn names() -> Vec<String> {
        ["car", "bus", "van", "others"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn counts_direction_and_class() {
        let mut lc = LineCounter::new((0.0, 300.0), (960.0, 300.0), names(), "line1");
        for f in 0..10 {
            let f64f = f as f64;
            let t = vec![
                trk(1.0, 100.0, 250.0 + 12.0 * f64f, 0.0),
                trk(2.0, 500.0, 360.0 - 12.0 * f64f, 1.0),
                trk(3.0, 800.0, 100.0 + f64f, 0.0),
            ];
            lc.update(&t, f, 25.0);
        }
        assert_eq!(lc.counts["car"], [1, 0]);
        assert_eq!(lc.counts["bus"], [0, 1]);
    }

    #[test]
    fn ignores_beyond_segment_and_jitter() {
        let mut lc = LineCounter::new((0.0, 300.0), (400.0, 300.0), names(), "l");
        for f in 0..10 {
            lc.update(&[trk(1.0, 700.0, 250.0 + 12.0 * f as f64, 0.0)], f, 25.0);
        }
        assert_eq!(lc.total_in() + lc.total_out(), 0);
        let mut lc = LineCounter::new((0.0, 300.0), (960.0, 300.0), names(), "l");
        for (f, y) in [290.0, 305.0, 295.0, 306.0, 294.0, 310.0]
            .iter()
            .enumerate()
        {
            lc.update(&[trk(7.0, 100.0, *y, 0.0)], f as u64, 25.0);
        }
        assert_eq!(lc.total_in() + lc.total_out(), 1);
    }
}
