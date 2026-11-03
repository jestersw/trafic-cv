mod detector;
mod draw;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::videoio;
use serde_json::json;
use traffic_core::counter::LineCounter;
use traffic_core::tracker;

use crate::detector::Detector;

#[derive(Parser, Debug)]
#[command(about = "Vehicle detection, tracking and line counting (ONNX Runtime + OpenCV)")]
struct Args {
    #[arg(long)]
    video: String,
    #[arg(long)]
    model: String,
    #[arg(long = "line", value_parser = parse_line)]
    lines: Vec<[f64; 4]>,
    #[arg(long, default_value = "bytetrack")]
    tracker: String,
    #[arg(long, default_value_t = 0.1)]
    conf: f32,
    #[arg(long, default_value_t = 0.7)]
    iou: f64,
    #[arg(long, default_value = "cpu")]
    ep: String,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value = "outputs/rust")]
    out_dir: PathBuf,
    #[arg(long)]
    no_video: bool,
    #[arg(long)]
    max_frames: Option<u64>,
    #[arg(long)]
    dump_dets: bool,
}

fn parse_line(s: &str) -> Result<[f64; 4], String> {
    let v: Vec<f64> = s
        .split(',')
        .map(|x| x.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    if v.len() != 4 {
        return Err("--line expects x1,y1,x2,y2".into());
    }
    Ok([v[0], v[1], v[2], v[3]])
}

fn percentile(xs: &mut [f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let pos = (xs.len() - 1) as f64 * q / 100.0;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    xs[lo] + (xs[hi] - xs[lo]) * (pos - lo as f64)
}

fn peak_rss_mb() -> f64 {
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let v = u.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        v / 1024.0 / 1024.0
    } else {
        v / 1024.0
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();
    let mut det = Detector::new(&args.model, &args.ep, args.threads, args.conf, args.iou)?;
    if det.names.is_empty() {
        det.names = ["car", "bus", "van", "others"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    let mut trk =
        tracker::build(&args.tracker).ok_or_else(|| anyhow!("unknown tracker {}", args.tracker))?;
    let mut counters: Vec<LineCounter> = args
        .lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            LineCounter::new(
                (l[0], l[1]),
                (l[2], l[3]),
                det.names.clone(),
                &format!("line{}", i + 1),
            )
        })
        .collect();

    let mut cap = videoio::VideoCapture::from_file(&args.video, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow!("cannot open video {}", args.video));
    }
    let mut vfps = cap.get(videoio::CAP_PROP_FPS)?;
    if vfps.is_nan() || vfps <= 0.0 {
        vfps = 25.0;
    }
    let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    fs::create_dir_all(&args.out_dir)?;
    let mut writer = if args.no_video {
        None
    } else {
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
        let path = args.out_dir.join("video.mp4");
        Some(videoio::VideoWriter::new(
            path.to_str().unwrap(),
            fourcc,
            vfps,
            Size::new(w, h),
            true,
        )?)
    };
    let mut events_f = BufWriter::new(File::create(args.out_dir.join("events.csv"))?);
    writeln!(
        events_f,
        "frame,time_s,track_id,class_id,class_name,event,detail,source"
    )?;
    let mut tracks_f = BufWriter::new(File::create(args.out_dir.join("tracks.txt"))?);
    let mut dets_f = if args.dump_dets {
        Some(BufWriter::new(File::create(
            args.out_dir.join("dets.jsonl"),
        )?))
    } else {
        None
    };

    let load_ms = t_start.elapsed().as_secs_f64() * 1e3;
    let (mut pre, mut inf, mut post, mut trk_ms, mut total_ms) =
        (vec![], vec![], vec![], vec![], vec![]);
    let mut recent: std::collections::VecDeque<f64> = Default::default();
    let mut frame = Mat::default();
    let mut idx: u64 = 0;
    let t_loop = Instant::now();
    loop {
        if args.max_frames.is_some_and(|m| idx >= m) {
            break;
        }
        if !cap.read(&mut frame)? || frame.empty() {
            break;
        }
        idx += 1;
        let t0 = Instant::now();
        let lb = det.preprocess(&frame)?;
        let t1 = Instant::now();
        let (shape, out) = det.infer()?;
        let t2 = Instant::now();
        let dets = det.postprocess(&shape, &out, &lb, w, h);
        let t3 = Instant::now();
        let tracks = trk.update(&dets);
        let t4 = Instant::now();
        pre.push((t1 - t0).as_secs_f64() * 1e3);
        inf.push((t2 - t1).as_secs_f64() * 1e3);
        post.push((t3 - t2).as_secs_f64() * 1e3);
        trk_ms.push((t4 - t3).as_secs_f64() * 1e3);

        for c in counters.iter_mut() {
            for e in c.update(&tracks, idx, vfps) {
                writeln!(
                    events_f,
                    "{},{:.3},{},{},{},{},{},{}",
                    e.frame,
                    e.time_s,
                    e.track_id,
                    e.class_id,
                    e.class_name,
                    e.event,
                    e.detail,
                    e.source
                )?;
            }
        }
        for r in &tracks {
            writeln!(
                tracks_f,
                "{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{},-1",
                idx,
                r[4] as u64,
                r[0],
                r[1],
                r[2] - r[0],
                r[3] - r[1],
                r[5],
                r[6] as i64
            )?;
        }
        if let Some(f) = dets_f.as_mut() {
            writeln!(f, "{}", json!({"frame": idx, "dets": dets}))?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        recent.push_back(elapsed);
        if recent.len() > 30 {
            recent.pop_front();
        }
        if let Some(wr) = writer.as_mut() {
            let fps = recent.len() as f64 / recent.iter().sum::<f64>();
            draw::draw(
                &mut frame,
                &tracks,
                &det.names,
                &counters,
                fps,
                idx as f64 / vfps,
            )?;
            wr.write(&frame)?;
        }
        total_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let seconds = t_loop.elapsed().as_secs_f64();
    if let Some(mut wr) = writer {
        wr.release()?;
    }
    events_f.flush()?;
    tracks_f.flush()?;

    let counts: serde_json::Map<String, serde_json::Value> = counters
        .iter()
        .map(|c| {
            let per: serde_json::Map<String, serde_json::Value> = c
                .counts
                .iter()
                .map(|(k, v)| (k.clone(), json!({"in": v[0], "out": v[1]})))
                .collect();
            (c.name.clone(), serde_json::Value::Object(per))
        })
        .collect();
    let totals: serde_json::Map<String, serde_json::Value> = counters
        .iter()
        .map(|c| {
            (
                c.name.clone(),
                json!({"in": c.total_in(), "out": c.total_out()}),
            )
        })
        .collect();
    let summary = json!({
        "video": args.video,
        "model": args.model,
        "tracker": args.tracker,
        "ep": args.ep,
        "counts": counts,
        "totals": totals,
        "perf": {
            "frames": idx,
            "fps_end_to_end": round2(idx as f64 / seconds),
            "model_load_ms": round2(load_ms),
            "preprocess_ms_p50": round2(percentile(&mut pre, 50.0)),
            "inference_ms_p50": round2(percentile(&mut inf.clone(), 50.0)),
            "inference_ms_p95": round2(percentile(&mut inf, 95.0)),
            "postprocess_ms_p50": round2(percentile(&mut post, 50.0)),
            "tracker_ms_p50": round2(percentile(&mut trk_ms, 50.0)),
            "frame_ms_p50": round2(percentile(&mut total_ms.clone(), 50.0)),
            "frame_ms_p95": round2(percentile(&mut total_ms, 95.0)),
            "peak_rss_mb": round2(peak_rss_mb()),
        }
    });
    let text = serde_json::to_string_pretty(&summary)?;
    fs::write(args.out_dir.join("summary.json"), &text).context("write summary")?;
    println!("{text}");
    Ok(())
}
