"""
Synthetic dataset generator: orchestrates interval generation across instruments.

Workflow:
  1. For each instrument — preload note samples from CDN.
  2. For each (instrument × interval × base_note) build a 2-second WAV clip.
  3. Assign each sample to a train / val / test split.
  4. Save audio files and a metadata CSV + summary JSON.

"""

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from .config import (
    INSTRUMENTS,
    INTERVALS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_DIR,
    get_split,
    midi_to_note,
    midi_to_safe_filename,
    InstrumentConfig,
)
from .sample_loader import SampleLoader
from .interval_builder import IntervalBuilder

logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """Metadata for a single generated interval sample."""
    sample_id: int
    path: str
    instrument: str
    interval: str
    semitones: int
    base_midi: int
    base_note: str
    upper_midi: int
    upper_note: str
    interval_type: str
    split: str
    duration: float

    @property
    def csv_row(self) -> list:
        return [
            self.sample_id, self.path, self.instrument, self.interval,
            self.semitones, self.base_midi, self.base_note,
            self.upper_midi, self.upper_note,
            self.interval_type, self.split, self.duration,
        ]


CSV_HEADER = [
    'sample_id', 'path', 'instrument', 'interval', 'semitones',
    'base_midi', 'base_note', 'upper_midi', 'upper_note',
    'type', 'split', 'duration',
]


class SyntheticDatasetGenerator:
    """Generates the full synthetic interval dataset."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        cache_dir: Optional[Path] = None,
        split_strategy: str = 'instrument',
        interval_types: Tuple[str, ...] = ('harmonic',),
        instruments: Optional[List[str]] = None,
        sr: int = SAMPLE_RATE,
        duration: float = SEGMENT_DURATION,
    ):
        self.output_dir = Path(output_dir)
        self.split_strategy = split_strategy
        self.interval_types = interval_types
        self.sr = sr
        self.duration = duration

        self.loader = SampleLoader(
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
            sample_rate=sr,
        )
        self.builder = IntervalBuilder(sr=sr, duration=duration)

        self.instrument_ids = instruments or list(INSTRUMENTS.keys())
        self.records: List[SampleRecord] = []
        self._counter = 0

    # Main entry point

    def generate(self) -> List[SampleRecord]:
        """Generate the full dataset and return metadata records."""
        est = self._estimate_total()
        logger.info(
            f"Generating synthetic dataset: ~{est} samples across "
            f"{len(self.instrument_ids)} instruments"
        )
        logger.info(f"Split strategy: {self.split_strategy}")
        logger.info(f"Interval types: {', '.join(self.interval_types)}")
        logger.info(f"Output directory: {self.output_dir}")

        self.records = []
        self._counter = 0
        generated = 0
        errors = 0
        t0 = time.time()

        for inst_id in self.instrument_ids:
            config = INSTRUMENTS[inst_id]
            logger.info(f"── {config.name} ({config.range_display}) ──")

            logger.info("  Downloading / caching samples...")
            self.loader.preload_instrument(inst_id)

            inst_generated = 0
            for interval_name, semitones in INTERVALS.items():
                base_notes = self._valid_base_notes(config, semitones)
                if not base_notes:
                    continue

                for base_midi in base_notes:
                    upper_midi = base_midi + semitones

                    for itype in self.interval_types:
                        try:
                            record = self._generate_one(
                                inst_id, interval_name, semitones,
                                base_midi, upper_midi, itype,
                            )
                            if record:
                                self.records.append(record)
                                generated += 1
                                inst_generated += 1
                        except Exception as e:
                            errors += 1
                            note = midi_to_note(base_midi)
                            logger.debug(
                                f"  Skip {interval_name} @ {note}: {e}"
                            )

            self.loader.clear_memory_cache()
            logger.info(f"  → {inst_generated} samples generated")

        self._save_metadata()

        elapsed = time.time() - t0
        logger.info(
            f"Done: {generated} samples, {errors} errors, {elapsed:.0f}s"
        )
        self._log_split_stats()
        return self.records

    # Per-sample generation

    def _generate_one(
        self,
        inst_id: str,
        interval_name: str,
        semitones: int,
        base_midi: int,
        upper_midi: int,
        interval_type: str,
    ) -> Optional[SampleRecord]:
        note1 = self.loader.get_note(inst_id, base_midi)
        note2 = self.loader.get_note(inst_id, upper_midi)
        if note1 is None or note2 is None:
            return None

        if interval_type == 'harmonic':
            audio = self.builder.build_harmonic(note1, note2)
        else:
            audio = self.builder.build_melodic(note1, note2)

        split = get_split(inst_id, base_midi, self.split_strategy)
        base_note = midi_to_note(base_midi)
        upper_note = midi_to_note(upper_midi)

        safe_base = midi_to_safe_filename(base_midi)
        rel_path = f"audio/{inst_id}/{interval_name}/{safe_base}_{interval_type}.wav"
        abs_path = self.output_dir / rel_path

        self._save_wav(audio, abs_path)

        self._counter += 1
        return SampleRecord(
            sample_id=self._counter,
            path=rel_path,
            instrument=inst_id,
            interval=interval_name,
            semitones=semitones,
            base_midi=base_midi,
            base_note=base_note,
            upper_midi=upper_midi,
            upper_note=upper_note,
            interval_type=interval_type,
            split=split,
            duration=self.duration,
        )

    # Helpers

    @staticmethod
    def _valid_base_notes(config: InstrumentConfig, semitones: int) -> List[int]:
        """All valid base MIDI notes for a given instrument and interval size."""
        lo, hi = config.midi_range
        max_base = hi - semitones
        if max_base < lo:
            return []
        return list(range(lo, max_base + 1))

    def _save_wav(self, audio: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, self.sr, subtype='PCM_16')

    def _save_metadata(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.output_dir / 'metadata.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for r in self.records:
                writer.writerow(r.csv_row)

        summary = self._build_summary()
        json_path = self.output_dir / 'dataset_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata: {csv_path}")
        logger.info(f"Summary:  {json_path}")

    def _build_summary(self) -> dict:
        by_split: dict = {}
        by_instrument: dict = {}
        by_interval: dict = {}

        for r in self.records:
            by_split[r.split] = by_split.get(r.split, 0) + 1
            by_instrument[r.instrument] = by_instrument.get(r.instrument, 0) + 1
            by_interval[r.interval] = by_interval.get(r.interval, 0) + 1

        return {
            'total_samples': len(self.records),
            'split_strategy': self.split_strategy,
            'interval_types': list(self.interval_types),
            'sample_rate': self.sr,
            'duration_sec': self.duration,
            'by_split': by_split,
            'by_instrument': by_instrument,
            'by_interval': by_interval,
            'instruments': [
                {
                    'id': iid,
                    'name': INSTRUMENTS[iid].name,
                    'range': INSTRUMENTS[iid].range_display,
                    'source': INSTRUMENTS[iid].source_type,
                }
                for iid in self.instrument_ids
            ],
        }

    def _log_split_stats(self):
        by_split: dict = {}
        for r in self.records:
            by_split[r.split] = by_split.get(r.split, 0) + 1
        total = len(self.records) or 1
        for split, count in sorted(by_split.items()):
            logger.info(f"  {split}: {count} ({count / total * 100:.1f}%)")

    def _estimate_total(self) -> int:
        total = 0
        for inst_id in self.instrument_ids:
            config = INSTRUMENTS[inst_id]
            for semitones in INTERVALS.values():
                total += len(self._valid_base_notes(config, semitones))
        return total * len(self.interval_types)
