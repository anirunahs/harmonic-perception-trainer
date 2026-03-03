"""
Dataset loading, splitting, and augmentation.

Workflow
--------
1.  Read metadata.csv, assign each sample to train / val / test by instrument.
2.  Further split train into train-core (85 %) and train-dev (15 %),
    stratified by interval class.
3.  Extract (or load cached) features for every sample with the chosen
    representation extractor.
4.  Wrap in torch.utils.data.Dataset, apply SpecAugment during training.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    CACHE_DIR,
    DATASET_DIR,
    FREQ_MASK_PARAM,
    LABEL_TO_IDX,
    METADATA_CSV,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
    SR,
    SPLIT_INSTRUMENTS,
    TIME_MASK_PARAM,
    TRAIN_DEV_FRACTION,
)
from .representations import get_extractor

log = logging.getLogger(__name__)


# SpecAugment

class SpecAugment:
    """Frequency + time masking for 2-D spectrograms (C, F, T)."""

    def __init__(
        self,
        freq_mask: int = FREQ_MASK_PARAM,
        time_mask: int = TIME_MASK_PARAM,
        n_freq: int = NUM_FREQ_MASKS,
        n_time: int = NUM_TIME_MASKS,
    ):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        _, F, T = x.shape
        for _ in range(self.n_freq):
            f = np.random.randint(0, self.freq_mask + 1)
            f0 = np.random.randint(0, max(F - f, 1))
            x[:, f0:f0 + f, :] = 0.0
        for _ in range(self.n_time):
            t = np.random.randint(0, self.time_mask + 1)
            t0 = np.random.randint(0, max(T - t, 1))
            x[:, :, t0:t0 + t] = 0.0
        return x


# PyTorch Dataset

class IntervalDataset(Dataset):
    """Wraps pre-extracted features + labels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: Optional[SpecAugment] = None,
    ):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)


# Split logic

def _assign_instrument_split(df: pd.DataFrame) -> pd.DataFrame:
    """Override the dataset's original split column with our 6/2/2 split."""
    inv = {}
    for split, instruments in SPLIT_INSTRUMENTS.items():
        for inst in instruments:
            inv[inst] = split
    df = df.copy()
    df["split"] = df["instrument"].map(inv)
    unknown = df["split"].isna()
    if unknown.any():
        log.warning("Instruments not in split map: %s",
                    df.loc[unknown, "instrument"].unique().tolist())
        df.loc[unknown, "split"] = "train"
    return df


def _split_train_dev(
    df: pd.DataFrame, fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of *train* rows into train-core and train-dev."""
    train_df = df[df["split"] == "train"].copy()
    idx_core, idx_dev = train_test_split(
        train_df.index,
        test_size=fraction,
        stratify=train_df["interval"],
        random_state=seed,
    )
    return df.loc[idx_core], df.loc[idx_dev]


# Feature extraction with disk cache

def _extract_features(
    df: pd.DataFrame,
    repr_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio files and extract features for every row in *df*."""
    extractor = get_extractor(repr_name)
    features: List[np.ndarray] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        audio_path = DATASET_DIR / row["path"]
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        feat = extractor(y)
        features.append(feat)
        labels.append(LABEL_TO_IDX[row["interval"]])

    return np.stack(features), np.array(labels, dtype=np.int64)


def _load_or_extract(
    df: pd.DataFrame,
    repr_name: str,
    split_name: str,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features, with optional .npz caching to disk."""
    cache_file = CACHE_DIR / f"{repr_name}_{split_name}.npz"

    if use_cache and cache_file.exists():
        log.info("Loading cached features: %s", cache_file)
        data = np.load(cache_file)
        return data["features"], data["labels"]

    log.info("Extracting %s features for %d samples (%s)...",
             repr_name, len(df), split_name)
    features, labels = _extract_features(df, repr_name)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels)
        log.info("Cached → %s", cache_file)

    return features, labels


# Public API

SplitData = Dict[str, Tuple[np.ndarray, np.ndarray]]


def prepare_splits(
    repr_name: str,
    seed: int = 42,
    use_cache: bool = True,
) -> SplitData:
    """Build all four splits: train, train_dev, val, test.

    Returns
    -------
    dict  {split_name: (features, labels)}
    """
    df = pd.read_csv(METADATA_CSV)
    df = _assign_instrument_split(df)

    train_core_df, train_dev_df = _split_train_dev(df, TRAIN_DEV_FRACTION, seed)
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    log.info("Split sizes — train: %d, train_dev: %d, val: %d, test: %d",
             len(train_core_df), len(train_dev_df), len(val_df), len(test_df))

    splits: SplitData = {}
    for name, sub_df in [
        ("train", train_core_df),
        ("train_dev", train_dev_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        splits[name] = _load_or_extract(sub_df, repr_name, name, use_cache)

    return splits


def create_dataloaders(
    splits: SplitData,
    batch_size: int = 32,
    repr_name: str = "mel",
) -> Dict[str, DataLoader]:
    """Wrap split arrays in DataLoaders.

    SpecAugment is applied only to the *train* loader and only for
    2-D representations (mel / cqt / hcqt).
    """
    augment = SpecAugment() if repr_name != "raw" else None

    loaders: Dict[str, DataLoader] = {}
    for name, (feats, labels) in splits.items():
        aug = augment if name == "train" else None
        ds = IntervalDataset(feats, labels, augment=aug)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
