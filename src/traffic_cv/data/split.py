"""Video-level train/val/test split.

Frames of one sequence are strongly correlated, so splitting by frame leaks the
test set into training. We split whole sequences, stratified by a sequence
attribute (weather) so that night/rain footage appears in every split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeqInfo:
    name: str
    official_split: str  # "train" | "test"
    stratum: str  # e.g. weather
    num_frames: int = 0


def _take_fraction(
    seqs: list[SeqInfo], fraction: float, rng: random.Random
) -> tuple[list[SeqInfo], list[SeqInfo]]:
    """Move ~fraction of sequences (per stratum) out. Returns (kept, taken)."""
    by_stratum: dict[str, list[SeqInfo]] = defaultdict(list)
    for s in seqs:
        by_stratum[s.stratum].append(s)

    kept: list[SeqInfo] = []
    taken: list[SeqInfo] = []
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda s: s.name)
        rng.shuffle(group)
        # at least one taken per stratum if the stratum has >= 2 sequences
        n_take = round(len(group) * fraction)
        if fraction > 0 and n_take == 0 and len(group) >= 2:
            n_take = 1
        taken.extend(group[:n_take])
        kept.extend(group[n_take:])
    return kept, taken


def split_sequences(
    seqs: Iterable[SeqInfo],
    strategy: str = "official",
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Return {"train": [...], "val": [...], "test": [...]} of sequence names."""
    seqs = list(seqs)
    rng = random.Random(seed)

    if strategy == "official":
        test = [s for s in seqs if s.official_split == "test"]
        pool = [s for s in seqs if s.official_split != "test"]
    elif strategy == "random":
        pool, test = _take_fraction(seqs, test_fraction, rng)
    else:
        raise ValueError(f"unknown split strategy: {strategy!r}")

    train, val = _take_fraction(pool, val_fraction, rng)
    out = {
        "train": sorted(s.name for s in train),
        "val": sorted(s.name for s in val),
        "test": sorted(s.name for s in test),
    }
    _assert_disjoint(out)
    return out


def _assert_disjoint(split: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for part, names in split.items():
        for n in names:
            if n in seen:
                raise AssertionError(f"sequence {n} in both {seen[n]} and {part}")
            seen[n] = part
