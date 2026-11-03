use anyhow::{anyhow, Context, Result};
use opencv::core::{self as cv, Mat, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use traffic_core::geometry::iou;
use traffic_core::Det;

pub struct Letterbox {
    pub gain: f64,
    pub left: i32,
    pub top: i32,
}

pub struct Detector {
    session: Session,
    input_name: String,
    output_name: String,
    pub imgsz: i32,
    pub conf: f32,
    pub iou: f64,
    pub max_det: usize,
    pub names: Vec<String>,
    buf: Vec<f32>,
}

pub fn parse_names(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' {
            let mut name = String::new();
            for d in chars.by_ref() {
                if d == c {
                    break;
                }
                name.push(d);
            }
            out.push(name);
        }
    }
    out
}

fn execution_providers(ep: &str) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    match ep {
        "cpu" => {}
        #[cfg(feature = "tensorrt")]
        "tensorrt" => {
            v.push(
                ort::ep::TensorRT::default()
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path("trt_cache")
                    .build(),
            );
            v.push(ort::ep::CUDA::default().build());
        }
        #[cfg(feature = "cuda")]
        "cuda" => v.push(ort::ep::CUDA::default().build()),
        #[cfg(feature = "coreml")]
        "coreml" => v.push(ort::ep::CoreML::default().build()),
        other => return Err(anyhow!("execution provider {other:?} is not compiled in")),
    }
    Ok(v)
}

impl Detector {
    pub fn new(model: &str, ep: &str, threads: usize, conf: f32, iou: f64) -> Result<Self> {
        let env = ort::init().build()?;
        let mut builder = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::<()>::from)?
            .with_execution_providers(execution_providers(ep)?)
            .map_err(ort::Error::<()>::from)?;
        if threads > 0 {
            builder = builder
                .with_intra_threads(threads)
                .map_err(ort::Error::<()>::from)?;
        }
        let session = builder
            .commit_from_file(model)
            .with_context(|| format!("load {model}"))?;
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        let names = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("names"))
            .map(|s| parse_names(&s))
            .unwrap_or_default();
        Ok(Self {
            session,
            input_name,
            output_name,
            imgsz: 640,
            conf,
            iou,
            max_det: 300,
            names,
            buf: Vec::new(),
        })
    }

    pub fn preprocess(&mut self, frame: &Mat) -> Result<Letterbox> {
        let (w, h) = (frame.cols(), frame.rows());
        let s = self.imgsz;
        let gain = (s as f64 / h as f64).min(s as f64 / w as f64);
        let (nw, nh) = (
            (w as f64 * gain).round() as i32,
            (h as f64 * gain).round() as i32,
        );
        let dw = (s - nw) as f64 / 2.0;
        let dh = (s - nh) as f64 / 2.0;
        let (top, bottom) = ((dh - 0.1).round() as i32, (dh + 0.1).round() as i32);
        let (left, right) = ((dw - 0.1).round() as i32, (dw + 0.1).round() as i32);
        let mut resized = Mat::default();
        if (nw, nh) != (w, h) {
            imgproc::resize(
                frame,
                &mut resized,
                Size::new(nw, nh),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
        } else {
            resized = frame.try_clone()?;
        }
        let mut padded = Mat::default();
        cv::copy_make_border(
            &resized,
            &mut padded,
            top,
            bottom,
            left,
            right,
            cv::BORDER_CONSTANT,
            Scalar::all(114.0),
        )?;
        let bytes = padded.data_bytes()?;
        let plane = (s * s) as usize;
        self.buf.resize(3 * plane, 0.0);
        for (i, px) in bytes.chunks_exact(3).enumerate() {
            self.buf[i] = px[2] as f32 / 255.0;
            self.buf[plane + i] = px[1] as f32 / 255.0;
            self.buf[2 * plane + i] = px[0] as f32 / 255.0;
        }
        Ok(Letterbox { gain, left, top })
    }

    pub fn infer(&mut self) -> Result<(Vec<usize>, Vec<f32>)> {
        let s = self.imgsz as usize;
        let input = Tensor::from_array(([1usize, 3, s, s], self.buf.clone()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        let (shape, data) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok((shape, data.to_vec()))
    }

    pub fn postprocess(
        &self,
        shape: &[usize],
        data: &[f32],
        lb: &Letterbox,
        w: i32,
        h: i32,
    ) -> Vec<Det> {
        let (ch, n) = (shape[1], shape[2]);
        let nc = ch - 4;
        let mut cand: Vec<Det> = Vec::new();
        for i in 0..n {
            let mut best = (0usize, f32::MIN);
            for c in 0..nc {
                let v = data[(4 + c) * n + i];
                if v > best.1 {
                    best = (c, v);
                }
            }
            if best.1 <= self.conf {
                continue;
            }
            let (cx, cy, bw, bh) = (data[i], data[n + i], data[2 * n + i], data[3 * n + i]);
            cand.push([
                (cx - bw / 2.0) as f64,
                (cy - bh / 2.0) as f64,
                (cx + bw / 2.0) as f64,
                (cy + bh / 2.0) as f64,
                best.1 as f64,
                best.0 as f64,
            ]);
        }
        cand.sort_by(|a, b| b[4].total_cmp(&a[4]));
        cand.truncate(30_000);
        let off = |d: &Det| {
            let o = d[5] * 7680.0;
            [d[0] + o, d[1] + o, d[2] + o, d[3] + o]
        };
        let mut keep: Vec<Det> = Vec::new();
        let mut suppressed = vec![false; cand.len()];
        for i in 0..cand.len() {
            if suppressed[i] {
                continue;
            }
            keep.push(cand[i]);
            if keep.len() >= self.max_det {
                break;
            }
            let a = off(&cand[i]);
            for j in (i + 1)..cand.len() {
                if !suppressed[j] && iou(&a, &off(&cand[j])) > self.iou {
                    suppressed[j] = true;
                }
            }
        }
        for d in keep.iter_mut() {
            d[0] = ((d[0] - lb.left as f64) / lb.gain).clamp(0.0, w as f64);
            d[2] = ((d[2] - lb.left as f64) / lb.gain).clamp(0.0, w as f64);
            d[1] = ((d[1] - lb.top as f64) / lb.gain).clamp(0.0, h as f64);
            d[3] = ((d[3] - lb.top as f64) / lb.gain).clamp(0.0, h as f64);
        }
        keep
    }
}
