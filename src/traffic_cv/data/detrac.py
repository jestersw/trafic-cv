from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    def clip(self, img_w: int, img_h: int) -> Box:
        x1, y1, x2, y2 = self.xyxy
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
        return Box(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


@dataclass(frozen=True, slots=True)
class Target:
    track_id: int
    box: Box
    vehicle_type: str
    truncation_ratio: float = 0.0


@dataclass(slots=True)
class Sequence:
    name: str
    weather: str
    camera_state: str
    ignored_regions: list[Box] = field(default_factory=list)
    frames: dict[int, list[Target]] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return max(self.frames, default=0)


def _box(el: ET.Element) -> Box:
    return Box(
        float(el.get("left", 0)),
        float(el.get("top", 0)),
        float(el.get("width", 0)),
        float(el.get("height", 0)),
    )


def parse_sequence(xml_path: str | Path) -> Sequence:
    root = ET.parse(xml_path).getroot()
    if root.tag != "sequence":
        raise ValueError(f"{xml_path}: expected <sequence> root, got <{root.tag}>")

    attr = root.find("sequence_attribute")
    weather = camera_state = "unknown"
    if attr is not None:
        weather = attr.get("sence_weather") or attr.get("scene_weather") or "unknown"
        camera_state = attr.get("camera_state", "unknown")

    seq = Sequence(
        name=root.get("name", Path(xml_path).stem), weather=weather, camera_state=camera_state
    )

    ignored = root.find("ignored_region")
    if ignored is not None:
        seq.ignored_regions = [_box(b) for b in ignored.findall("box")]

    for frame_el in root.iter("frame"):
        num = int(frame_el.get("num"))
        targets: list[Target] = []
        for t in frame_el.iter("target"):
            box_el = t.find("box")
            a = t.find("attribute")
            if box_el is None:
                continue
            targets.append(
                Target(
                    track_id=int(t.get("id")),
                    box=_box(box_el),
                    vehicle_type=(a.get("vehicle_type") if a is not None else None) or "others",
                    truncation_ratio=float(a.get("truncation_ratio", 0)) if a is not None else 0.0,
                )
            )
        seq.frames[num] = targets
    return seq


def frame_filename(num: int) -> str:
    return f"img{num:05d}.jpg"
