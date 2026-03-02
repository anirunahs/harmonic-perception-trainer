"""
Main orchestrator for the recorded dataset processing pipeline.

Workflow per recording file:
  1. Load and normalize (LUFS, bandpass, noise reduction)
  2. Detect onsets and extract 2-second segments
  3. Analyze quality and reject defective segments
  4. Save accepted segments as WAV files
  5. (Optionally) create pitch-shifted augmentations for train split
  6. Write metadata CSV + summary JSON

"""

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .config import (
    SAMPLE_RATE,
    SEGMENT_DURATION,
    EXPECTED_INTERVALS_PER_FILE,
    INTERVAL_SEMITONES,
    DEFAULT_RAW_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSITIONS,
    POSITIONS,
    parse_recording_filename,
)
from .normalizer import normalize_recording
from .segmenter import segment_recording, Segment
from .quality_filter import filter_segments, QualityReport
from .augmenter import augment_segments
from .metadata import SegmentRecord, assign_split, save_metadata

logger = logging.getLogger(__name__)


class RecordedDatasetPipeline:
    """Processes raw recorded audio files into a clean, segmented dataset."""

    def __init__(
        self,
        raw_dir: Path = DEFAULT_RAW_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        positions: Optional[List[str]] = None,
        split_strategy: str = 'class',
        augment_train: bool = True,
        apply_noise_reduction: Optional[bool] = None,
        sr: int = SAMPLE_RATE,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.positions = positions or list(DEFAULT_POSITIONS)
        self.split_strategy = split_strategy
        self.augment_train = augment_train
        self.force_nr = apply_noise_reduction
        self.sr = sr

        self.records: List[SegmentRecord] = []
        self._counter = 0

    def run(self) -> List[SegmentRecord]:
        """Execute the full pipeline across all positions and files."""
        t0 = time.time()
        self.records = []
        self._counter = 0

        total_accepted = 0
        total_rejected = 0
        total_augmented = 0

        for pos_name in self.positions:
            pos = POSITIONS.get(pos_name)
            if pos is None:
                logger.error(f"Unknown position: {pos_name}")
                continue

            pos_dir = self.raw_dir / pos.folder
            if not pos_dir.exists():
                logger.warning(f"Directory not found: {pos_dir} — skipping position '{pos_name}'")
                continue

            wav_files = sorted(pos_dir.glob('*.wav'))
            if not wav_files:
                logger.warning(f"No WAV files in {pos_dir}")
                continue

            logger.info(f"═══ Position: {pos.name} ({pos.description}) — {len(wav_files)} files ═══")

            for wav_path in wav_files:
                try:
                    accepted, rejected, augmented = self._process_one_file(
                        wav_path, pos_name
                    )
                    total_accepted += accepted
                    total_rejected += rejected
                    total_augmented += augmented
                except Exception as e:
                    logger.error(f"Failed to process {wav_path.name}: {e}", exc_info=True)
                finally:
                    gc.collect()

        save_metadata(self.records, self.output_dir)

        elapsed = time.time() - t0
        logger.info(f"{'=' * 60}")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"  Accepted originals: {total_accepted}")
        logger.info(f"  Rejected:           {total_rejected}")
        logger.info(f"  Augmented copies:   {total_augmented}")
        logger.info(f"  Total records:      {len(self.records)}")
        logger.info(f"  Output directory:   {self.output_dir}")
        logger.info(f"{'=' * 60}")

        return self.records

    def _process_one_file(self, wav_path: Path, position: str):
        """Process a single long recording file."""
        logger.info(f"── {wav_path.name} ──")

        interval, dynamics = parse_recording_filename(wav_path.name)
        logger.info(f"  Interval: {interval}, dynamics: {dynamics}")

        pos_cfg = POSITIONS[position]
        use_nr = pos_cfg.needs_noise_reduction if self.force_nr is None else self.force_nr
        logger.info(f"  Noise reduction: {'yes' if use_nr else 'skip (already cleaned)'}")

        audio = normalize_recording(str(wav_path), self.sr, apply_nr=use_nr)

        segments = segment_recording(
            audio, self.sr, expected_count=EXPECTED_INTERVALS_PER_FILE
        )

        accepted, rejected = filter_segments(segments, self.sr)

        split = assign_split(interval, position, self.split_strategy)

        n_saved = self._save_accepted(
            accepted, interval, dynamics, position,
            wav_path.name, split, pitch_shift=0
        )

        n_augmented = 0
        if self.augment_train and split == 'train' and accepted:
            aug_list = augment_segments(accepted, self.sr)
            for aug_seg, parent_seg, shift in aug_list:
                shift_label = f"up{abs(shift)}" if shift > 0 else f"down{abs(shift)}"
                rel_path = self._segment_path(
                    interval, dynamics, position,
                    parent_seg.onset_index, shift_label
                )
                abs_path = self.output_dir / rel_path
                self._save_wav(aug_seg.audio, abs_path)

                self._counter += 1
                self.records.append(SegmentRecord(
                    sample_id=self._counter,
                    path=str(rel_path),
                    instrument='piano_recorded',
                    interval=interval,
                    semitones=INTERVAL_SEMITONES[interval],
                    dynamics=dynamics,
                    position=position,
                    source_file=wav_path.name,
                    onset_index=parent_seg.onset_index,
                    pitch_shift=shift,
                    split=split,
                    duration=SEGMENT_DURATION,
                    rms_db=parent_seg.rms_db,
                    is_augmented=True,
                ))
                n_augmented += 1

        logger.info(
            f"  Result: {n_saved} original, {n_augmented} augmented, "
            f"{len(rejected)} rejected"
        )
        return n_saved, len(rejected), n_augmented

    def _save_accepted(
        self,
        accepted: list,
        interval: str,
        dynamics: str,
        position: str,
        source_file: str,
        split: str,
        pitch_shift: int,
    ) -> int:
        """Save accepted segments as WAV files and append metadata records."""
        count = 0
        for seg, report in accepted:
            rel_path = self._segment_path(
                interval, dynamics, position, seg.onset_index, 'original'
            )
            abs_path = self.output_dir / rel_path
            self._save_wav(seg.audio, abs_path)

            self._counter += 1
            self.records.append(SegmentRecord(
                sample_id=self._counter,
                path=str(rel_path),
                instrument='piano_recorded',
                interval=interval,
                semitones=INTERVAL_SEMITONES[interval],
                dynamics=dynamics,
                position=position,
                source_file=source_file,
                onset_index=seg.onset_index,
                pitch_shift=pitch_shift,
                split=split,
                duration=SEGMENT_DURATION,
                rms_db=report.rms_db,
                is_augmented=False,
            ))
            count += 1
        return count

    @staticmethod
    def _segment_path(interval: str, dynamics: str, position: str,
                      onset_idx: int, variant: str) -> Path:
        """Build a relative path for a segment WAV file."""
        return Path(
            f"audio/{position}/{interval}/{dynamics}/"
            f"{interval}_{onset_idx:03d}_{dynamics}_{variant}.wav"
        )

    @staticmethod
    def _save_wav(audio: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype='PCM_16')
