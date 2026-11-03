from __future__ import annotations

Point = tuple[float, float]


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side(p1: Point, p2: Point, q: Point, eps: float = 1e-9) -> int:
    c = cross(p1, p2, q)
    return 0 if abs(c) <= eps else (1 if c > 0 else -1)


def signed_distance(p1: Point, p2: Point, q: Point) -> float:
    length = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return cross(p1, p2, q) / max(length, 1e-9)


def projection(p1: Point, p2: Point, q: Point) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return ((q[0] - p1[0]) * dx + (q[1] - p1[1]) * dy) / max(dx * dx + dy * dy, 1e-9)


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    d1, d2 = side(b1, b2, a1), side(b1, b2, a2)
    d3, d4 = side(a1, a2, b1), side(a1, a2, b2)
    return d1 != d2 and d3 != d4 and d1 * d2 <= 0 and d3 * d4 <= 0


def point_in_polygon(q: Point, poly: list[Point]) -> bool:
    x, y = q
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at:
                inside = not inside
    return inside


def anchor(box_xyxy, mode: str = "bottom_center") -> Point:
    x1, y1, x2, y2 = (float(v) for v in box_xyxy[:4])
    if mode == "bottom_center":
        return ((x1 + x2) / 2, y2)
    if mode == "center":
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    raise ValueError(f"unknown anchor mode {mode!r}")
