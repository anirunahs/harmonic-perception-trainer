#!/usr/bin/env python3
"""
CLI entry point for synthetic interval dataset generation.

Generates 2-second WAV clips of musical intervals from real instrument
samples (Salamander Grand Piano, FluidR3_GM SoundFonts).

Usage:
    # Generate full dataset (all 10 instruments, harmonic intervals)
    python generate_synthetic_dataset.py

    # Specific instruments only
    python generate_synthetic_dataset.py -i piano violin guitar_nylon

    # Include melodic intervals
    python generate_synthetic_dataset.py --melodic

    # Use register-based split instead of instrument-based
    python generate_synthetic_dataset.py -s register

    # Custom output directory
    python generate_synthetic_dataset.py -o ./my-dataset

    # Quick test with one instrument
    python generate_synthetic_dataset.py -i piano -v
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthetic.config import INSTRUMENTS, DEFAULT_OUTPUT_DIR, DEFAULT_CACHE_DIR
from synthetic.dataset_generator import SyntheticDatasetGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic musical interval dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available instruments:
  piano           Salamander Grand Piano    (C2–C6)
  guitar_nylon    Acoustic Guitar Nylon     (E2–E5)
  guitar_steel    Acoustic Guitar Steel     (E2–E5)
  violin          Violin                    (G3–E6)
  cello           Cello                     (C2–C5)
  flute           Flute                     (C4–C7)
  clarinet        Clarinet                  (D3–C6)
  trumpet         Trumpet                   (G3–C6)
  church_organ    Church Organ              (C2–C7)
  marimba         Marimba                   (C3–C6)

Split strategies:
  instrument    Disjoint instrument sets (default, strongest generalisation)
                  train: piano, guitar_nylon, guitar_steel, cello, church_organ
                  val:   clarinet, marimba
                  test:  violin, flute, trumpet
  register      Disjoint pitch ranges (all instruments in each split)
                  train: C2–G4  |  val: G#4–C#5  |  test: D5–C7

Requirements:
  ffmpeg must be installed for mp3 decoding (conda install -c conda-forge ffmpeg)
        """,
    )
    parser.add_argument(
        '--output', '-o', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Output directory (default: %(default)s)',
    )
    parser.add_argument(
        '--strategy', '-s', choices=['instrument', 'register'],
        default='instrument',
        help='Train/val/test split strategy (default: instrument)',
    )
    parser.add_argument(
        '--instruments', '-i', nargs='+',
        choices=list(INSTRUMENTS.keys()),
        help='Instruments to include (default: all 10)',
    )
    parser.add_argument(
        '--melodic', action='store_true',
        help='Also generate melodic (sequential) intervals',
    )
    parser.add_argument(
        '--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
        help='Sample cache directory (default: %(default)s)',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable debug-level logging',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    interval_types = ('harmonic',)
    if args.melodic:
        interval_types = ('harmonic', 'melodic')

    generator = SyntheticDatasetGenerator(
        output_dir=args.output,
        cache_dir=args.cache_dir,
        split_strategy=args.strategy,
        interval_types=interval_types,
        instruments=args.instruments,
    )

    t0 = time.time()
    records = generator.generate()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Dataset generated: {len(records)} samples in {elapsed:.1f}s")
    print(f"Output: {args.output.resolve()}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
