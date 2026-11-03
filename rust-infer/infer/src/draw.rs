use anyhow::Result;
use opencv::core::{self as cv, Mat, Point, Rect, Scalar};
use opencv::imgproc;
use opencv::prelude::*;
use traffic_core::counter::LineCounter;
use traffic_core::TrackRow;

const FONT: i32 = imgproc::FONT_HERSHEY_SIMPLEX;

pub fn id_color(id: u64) -> Scalar {
    let h = (id as f64 * 0.618_033_988_749_895) % 1.0;
    let (s, v) = (0.75, 1.0);
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Scalar::new(b * 255.0, g * 255.0, r * 255.0, 0.0)
}

fn label(img: &mut Mat, text: &str, x: i32, y: i32, color: Scalar) -> Result<()> {
    let mut base = 0;
    let sz = imgproc::get_text_size(text, FONT, 0.5, 1, &mut base)?;
    let y = y.max(sz.height + base);
    imgproc::rectangle(
        img,
        Rect::new(x, y - sz.height - base, sz.width + 2, sz.height + base),
        color,
        -1,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::put_text(
        img,
        text,
        Point::new(x + 1, y - base),
        FONT,
        0.5,
        Scalar::all(0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

pub fn draw(
    img: &mut Mat,
    tracks: &[TrackRow],
    names: &[String],
    lines: &[LineCounter],
    fps: f64,
    t_s: f64,
) -> Result<()> {
    let yellow = Scalar::new(0.0, 255.0, 255.0, 0.0);
    for ln in lines {
        let p1 = Point::new(ln.p1.0 as i32, ln.p1.1 as i32);
        let p2 = Point::new(ln.p2.0 as i32, ln.p2.1 as i32);
        imgproc::line(img, p1, p2, yellow, 2, imgproc::LINE_AA, 0)?;
        let text = format!("{}  in {}  out {}", ln.name, ln.total_in(), ln.total_out());
        label(img, &text, p1.x, p1.y - 6, yellow)?;
    }
    for r in tracks {
        let id = r[4] as u64;
        let color = id_color(id);
        let (x1, y1, x2, y2) = (r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32);
        imgproc::rectangle(
            img,
            Rect::new(x1, y1, x2 - x1, y2 - y1),
            color,
            2,
            imgproc::LINE_8,
            0,
        )?;
        let cls = r[6] as usize;
        let name = names.get(cls).cloned().unwrap_or_else(|| cls.to_string());
        label(img, &format!("#{id} {name}"), x1, y1 - 2, color)?;
    }
    let mut rows = vec![format!("FPS {fps:5.1}   t={t_s:6.1}s   [rust]")];
    for ln in lines {
        for (cname, c) in &ln.counts {
            rows.push(format!(
                "{}/{:<7} in {:>3}  out {:>3}",
                ln.name, cname, c[0], c[1]
            ));
        }
    }
    let (pad, lh) = (8, 20);
    let mut w = 0;
    for r in &rows {
        let mut base = 0;
        w = w.max(imgproc::get_text_size(r, FONT, 0.5, 1, &mut base)?.width);
    }
    let mut overlay = img.try_clone()?;
    imgproc::rectangle(
        &mut overlay,
        Rect::new(0, 0, w + 2 * pad, pad + lh * rows.len() as i32),
        Scalar::all(0.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;
    let mut blended = Mat::default();
    cv::add_weighted(&overlay, 0.55, img, 0.45, 0.0, &mut blended, -1)?;
    *img = blended;
    for (i, r) in rows.iter().enumerate() {
        imgproc::put_text(
            img,
            r,
            Point::new(pad, pad + lh * (i as i32 + 1) - 6),
            FONT,
            0.5,
            Scalar::all(255.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}
