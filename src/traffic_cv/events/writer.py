from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import IO

from traffic_cv.events.counters import Event

FIELDS = [f.name for f in fields(Event)]


class EventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fmt = "csv" if self.path.suffix.lower() == ".csv" else "jsonl"
        self._f: IO[str] = open(self.path, "w", newline="")
        self._csv = None
        if self.fmt == "csv":
            self._csv = csv.DictWriter(self._f, fieldnames=FIELDS)
            self._csv.writeheader()

    def write(self, events: list[Event]) -> None:
        for e in events:
            d = e.to_dict()
            d["time_s"] = round(d["time_s"], 3)
            if self._csv is not None:
                self._csv.writerow(d)
            else:
                self._f.write(json.dumps(d) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
