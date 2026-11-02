from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

XML_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="{name}">
  <sequence_attribute camera_state="unstable" sence_weather="{weather}"/>
  <ignored_region>
    <box left="800" top="0" width="160" height="100"/>
  </ignored_region>
{frames}
</sequence>
"""

FRAME_TMPL = """  <frame density="{n}" num="{num}">
    <target_list>
{targets}
    </target_list>
  </frame>"""

TARGET_TMPL = """      <target id="{tid}">
        <box left="{l}" top="{t}" width="{w}" height="{h}"/>
        <attribute orientation="0" speed="1" trajectory_length="5" truncation_ratio="0.1" vehicle_type="{vt}"/>
      </target>"""


def make_sequence(
    root: Path, split_dir: str, ann_dir: str, name: str, weather: str, n_frames: int = 6
) -> None:
    img_dir = root / split_dir / name
    img_dir.mkdir(parents=True)
    frames = []
    for num in range(1, n_frames + 1):
        cv2.imwrite(str(img_dir / f"img{num:05d}.jpg"), np.full((540, 960, 3), 200, np.uint8))
        targets = [
            TARGET_TMPL.format(tid=1, l=100 + 5 * num, t=300, w=80, h=60, vt="car"),
            TARGET_TMPL.format(tid=2, l=400, t=200, w=120, h=90, vt="bus"),
            # inside the ignored region -> must be dropped when masking
            TARGET_TMPL.format(tid=3, l=820, t=10, w=40, h=30, vt="van"),
        ]
        frames.append(FRAME_TMPL.format(n=len(targets), num=num, targets="\n".join(targets)))
    (root / ann_dir).mkdir(exist_ok=True)
    (root / ann_dir / f"{name}.xml").write_text(
        XML_TMPL.format(name=name, weather=weather, frames="\n".join(frames))
    )


@pytest.fixture
def fake_detrac(tmp_path: Path) -> Path:
    root = tmp_path / "UA-DETRAC"
    weathers = ["sunny", "night", "rainy", "cloudy"]
    for i in range(8):
        make_sequence(
            root,
            "Insight-MVT_Annotation_Train",
            "DETRAC-Train-Annotations-XML",
            f"MVI_2{i:04d}",
            weathers[i % 4],
        )
    for i in range(3):
        make_sequence(
            root,
            "Insight-MVT_Annotation_Test",
            "DETRAC-Test-Annotations-XML",
            f"MVI_3{i:04d}",
            weathers[i % 4],
        )
    return root
