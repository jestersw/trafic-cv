from traffic_cv.tracking.base import Tracker
from traffic_cv.tracking.bytetrack import ByteTrack
from traffic_cv.tracking.sort import SORT

TRACKERS = {"bytetrack": ByteTrack, "sort": SORT}

try:
    import traffic_rs

    TRACKERS["bytetrack-rs"] = traffic_rs.ByteTrack
    TRACKERS["sort-rs"] = traffic_rs.SORT
except ImportError:
    pass


def build_tracker(name: str, **kwargs) -> Tracker:
    try:
        return TRACKERS[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"unknown tracker {name!r}, choose from {sorted(TRACKERS)}") from None


__all__ = ["SORT", "TRACKERS", "ByteTrack", "Tracker", "build_tracker"]
