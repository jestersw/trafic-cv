from traffic_cv.tracking.base import Tracker
from traffic_cv.tracking.bytetrack import ByteTrack
from traffic_cv.tracking.sort import SORT

TRACKERS = {"bytetrack": ByteTrack, "sort": SORT}


def build_tracker(name: str, **kwargs) -> Tracker:
    try:
        return TRACKERS[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"unknown tracker {name!r}, choose from {sorted(TRACKERS)}") from None


__all__ = ["SORT", "TRACKERS", "ByteTrack", "Tracker", "build_tracker"]
