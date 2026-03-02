"""
CLI entry point for recorded (microphone) dataset processing.

Processes raw audio recordings of musical intervals into a clean,
segmented dataset with quality filtering and optional pitch-shift
augmentation.

Usage:
    # Process recordings from lid position (default)
    python process_recorded_dataset.py

    # Process specific positions
    python process_recorded_dataset.py --positions lid table couch

    # Without noise reduction
    python process_recorded_dataset.py --no-noise-reduction

    # Without augmentation
    python process_recorded_dataset.py --no-augment

    # Use position-based split strategy
    python process_recorded_dataset.py --split-strategy position --positions lid table couch

    # Custom directories
    python process_recorded_dataset.py -i ./raw-recordings -o ./processed-segments

    # Verbose output
    python process_recorded_dataset.py -v
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recorded.config import DEFAULT_RAW_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_POSITIONS
from recorded.pipeline import RecordedDatasetPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Process recorded musical interval audio into a clean dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Positions (folder names inside raw-recordings/):
  NF    Near Field  — on the piano lid (~10–20 cm).  Already noise-reduced.
  MF    Mid Field   — on a table ~80 cm from piano.  Noise reduction applied.
  FF    Far Field   — on a couch ~1.5 m from piano.  Not recommended.

Default: NF + MF (recommended).  FF is skipped because it adds mostly
room reverb without new musical content (same performances as NF/MF).

Split strategies:
  class      Split by interval class (6 train / 2 val / 4 test).
             Both positions contribute to every split — no content leakage.
  position   Split by microphone position (NF train, MF test).
             Same content in train and test — only tests acoustic robustness.

Noise reduction:
  By default, per-position: NF is skipped (already cleaned), MF/FF applied.
  Use --force-nr or --no-noise-reduction to override for ALL positions.

Pipeline steps:
  1. Load audio at 22050 Hz (mono)
  2. DC offset removal + bandpass (80 Hz – 8 kHz)
  3. Noise reduction — per-position (NF skip, MF/FF apply) or forced
  4. LUFS normalization (EBU R128, -23 LUFS) — same level for all positions
  5. Onset detection → 2-second segment extraction
  6. Quality analysis (reject clicks, silence, noise)
  7. Pitch-shift augmentation ±1 semitone (train split only)
  8. Save WAV files + metadata CSV + summary JSON
        """,
    )
    parser.add_argument(
        '--input', '-i', type=Path, default=DEFAULT_RAW_DIR,
        help='Raw recordings root directory (default: %(default)s)',
    )
    parser.add_argument(
        '--output', '-o', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Output directory for processed segments (default: %(default)s)',
    )
    parser.add_argument(
        '--positions', '-p', nargs='+',
        choices=['NF', 'MF', 'FF'], default=list(DEFAULT_POSITIONS),
        help='Microphone positions to process (default: NF MF)',
    )
    parser.add_argument(
        '--split-strategy', '-s', choices=['class', 'position'],
        default='class',
        help='Train/val/test split strategy (default: class)',
    )
    parser.add_argument(
        '--no-augment', action='store_true',
        help='Skip pitch-shift augmentation',
    )
    nr_group = parser.add_mutually_exclusive_group()
    nr_group.add_argument(
        '--no-noise-reduction', action='store_true',
        help='Skip noise reduction for ALL positions',
    )
    nr_group.add_argument(
        '--force-nr', action='store_true',
        help='Force noise reduction for ALL positions (including NF)',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable debug-level logging',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    nr_flag = None
    if args.no_noise_reduction:
        nr_flag = False
    elif args.force_nr:
        nr_flag = True

    pipeline = RecordedDatasetPipeline(
        raw_dir=args.input,
        output_dir=args.output,
        positions=args.positions,
        split_strategy=args.split_strategy,
        augment_train=not args.no_augment,
        apply_noise_reduction=nr_flag,
    )

    t0 = time.time()
    records = pipeline.run()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Recorded dataset processed: {len(records)} samples in {elapsed:.1f}s")
    print(f"Output: {args.output.resolve()}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
