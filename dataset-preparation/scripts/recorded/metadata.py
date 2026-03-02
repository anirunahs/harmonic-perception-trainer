"""
Metadata management and split assignment for the recorded dataset.

Produces a CSV compatible with the synthetic dataset format, enabling
unified data loading downstream.
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import SPLIT_STRATEGIES, INTERVAL_SEMITONES

logger = logging.getLogger(__name__)


@dataclass
class SegmentRecord:
    """Metadata for one processed segment (original or augmented)."""
    sample_id: int
    path: str
    instrument: str
    interval: str
    semitones: int
    dynamics: str
    position: str
    source_file: str
    onset_index: int
    pitch_shift: int
    split: str
    duration: float
    rms_db: float
    is_augmented: bool

    @property
    def csv_row(self) -> list:
        return [
            self.sample_id, self.path, self.instrument, self.interval,
            self.semitones, self.dynamics, self.position, self.source_file,
            self.onset_index, self.pitch_shift, self.split, self.duration,
            f"{self.rms_db:.1f}", self.is_augmented,
        ]


CSV_HEADER = [
    'sample_id', 'path', 'instrument', 'interval', 'semitones',
    'dynamics', 'position', 'source_file', 'onset_index', 'pitch_shift',
    'split', 'duration', 'rms_db', 'is_augmented',
]


def assign_split(interval: str, position: str,
                 strategy: str = 'class') -> str:
    """Determine train/val/test split for a recorded segment.

    Strategies:
      - 'position': split by microphone position (NF train, MF test)
      - 'class': split by interval class (6 train / 2 val / 4 test)

    Note: augmented segments inherit the split of their parent.
    """
    mapping = SPLIT_STRATEGIES.get(strategy)
    if mapping is None:
        raise ValueError(f"Unknown split strategy: {strategy}")

    if strategy == 'position':
        for split_name, positions in mapping.items():
            if position in positions:
                return split_name
        return 'train'

    if strategy == 'class':
        for split_name, intervals in mapping.items():
            if interval in intervals:
                return split_name
        return 'train'

    return 'train'


def save_metadata(records: List[SegmentRecord], output_dir: Path):
    """Save metadata CSV and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'metadata.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in records:
            writer.writerow(r.csv_row)

    summary = _build_summary(records)
    json_path = output_dir / 'dataset_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata saved: {csv_path}")
    logger.info(f"Summary saved: {json_path}")


def _build_summary(records: List[SegmentRecord]) -> dict:
    by_split: Dict[str, int] = {}
    by_interval: Dict[str, int] = {}
    by_position: Dict[str, int] = {}
    by_dynamics: Dict[str, int] = {}
    originals = 0
    augmented = 0

    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_interval[r.interval] = by_interval.get(r.interval, 0) + 1
        by_position[r.position] = by_position.get(r.position, 0) + 1
        by_dynamics[r.dynamics] = by_dynamics.get(r.dynamics, 0) + 1
        if r.is_augmented:
            augmented += 1
        else:
            originals += 1

    return {
        'total_samples': len(records),
        'originals': originals,
        'augmented': augmented,
        'by_split': by_split,
        'by_interval': by_interval,
        'by_position': by_position,
        'by_dynamics': by_dynamics,
    }
