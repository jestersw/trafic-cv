"""Exploratory data analysis of the converted dataset.

Usage::

    python -m traffic_cv.data.eda --data data/processed/detrac --out reports/eda

Writes ``stats.json``, ``stats.md`` (tables pasted into the README) and plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# COCO-style size buckets (by sqrt(area) in pixels)
SIZE_BINS = [0, 32, 96, float("inf")]
SIZE_LABELS = ["small (<32²)", "medium", "large (>96²)"]


def _md_table(df) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |"]
    lines.append("| " + " | ".join(["---"] * (len(cols) + 1)) + " |")
    for idx, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [str(idx)] + [f"{v:,}" if isinstance(v, int) else str(v) for v in row.tolist()]
            )
            + " |"
        )
    return "\n".join(lines)


def compute(data_root: Path) -> dict:
    import pandas as pd

    boxes = pd.read_csv(data_root / "boxes.csv")
    seqs = pd.read_csv(data_root / "sequences.csv")
    names = yaml.safe_load((data_root / "dataset.yaml").read_text())["names"]
    boxes["class"] = boxes["class_id"].map(lambda i: names[int(i)])
    side = (boxes["width"] * boxes["height"]) ** 0.5
    boxes["size"] = pd.cut(side, SIZE_BINS, labels=SIZE_LABELS, right=False)

    order = ["train", "val", "test"]
    class_by_split = (
        boxes.pivot_table(index="class", columns="split", values="frame", aggfunc="count")
        .reindex(columns=order)
        .fillna(0)
        .astype(int)
    )
    class_by_split["total"] = class_by_split.sum(axis=1)
    class_by_split["share"] = (class_by_split["total"] / class_by_split["total"].sum()).map(
        lambda v: f"{v:.1%}"
    )

    size_by_class = (
        boxes.pivot_table(
            index="class", columns="size", values="frame", aggfunc="count", observed=False
        )
        .fillna(0)
        .astype(int)
    )
    weather_seqs = (
        seqs.pivot_table(index="weather", columns="split", values="sequence", aggfunc="count")
        .reindex(columns=order)
        .fillna(0)
        .astype(int)
    )
    split_summary = (
        seqs.groupby("split")
        .agg(
            sequences=("sequence", "count"),
            frames=("num_frames", "sum"),
            exported_frames=("exported_frames", "sum"),
            tracks=("num_tracks", "sum"),
        )
        .reindex(order)
        .fillna(0)
        .astype(int)
    )
    split_summary["boxes"] = boxes.groupby("split").size().reindex(order).fillna(0).astype(int)

    for df, name in [
        (class_by_split, "class"),
        (size_by_class, "class"),
        (weather_seqs, "weather"),
        (split_summary, "split"),
    ]:
        df.index.name = name

    return {
        "tables": {
            "split_summary": split_summary,
            "class_by_split": class_by_split,
            "size_by_class": size_by_class,
            "weather_sequences": weather_seqs,
        },
        "boxes": boxes,
    }


def plots(boxes, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    boxes["class"].value_counts().plot.bar(ax=axes[0], title="Boxes per class")
    axes[1].hist((boxes["width"] * boxes["height"]) ** 0.5, bins=60)
    axes[1].set(title="Box size, sqrt(area) px", yscale="log")
    axes[2].hist2d(boxes["cx"], boxes["cy"], bins=[96, 54], cmap="magma")
    axes[2].set(title="Box centre heatmap", xlim=(0, 960), ylim=(540, 0))
    fig.tight_layout()
    fig.savefig(out / "eda.png", dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/detrac")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = compute(Path(args.data))
    tables = res["tables"]

    titles = {
        "split_summary": "Split summary (split is by video)",
        "class_by_split": "Boxes per class",
        "size_by_class": "Object size per class",
        "weather_sequences": "Sequences per weather condition",
    }
    md = "\n\n".join(f"### {titles[k]}\n\n{_md_table(v)}" for k, v in tables.items())
    (out / "stats.md").write_text(md + "\n")
    (out / "stats.json").write_text(
        json.dumps({k: v.astype(str).to_dict() for k, v in tables.items()}, indent=2)
    )
    plots(res["boxes"], out)
    print(md)


if __name__ == "__main__":
    main()
