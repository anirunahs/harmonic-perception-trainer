"""
Dataset for Experiment 3: recorded (microphone) segments.

Loads metadata from processed-segments, filters by position (NF = train/val,
MF = test). Splits NF into train/val by source_file to avoid leakage.
Extracts HCQT on the fly or from cache.
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
    CQT_PARAMS,
    HCQT_PARAMS,
    LABEL_TO_IDX,
    MEL_PARAMS,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
    RECORDED_METADATA_CSV,
    RECORDED_SEGMENTS_DIR,
    RECORDED_SPLIT_BY_FILE,
    RECORDED_VAL_FRACTION,
    SR,
    TEST_POSITIONS,
    TIME_MASK_PARAM,
    TRAIN_POSITIONS,
    FREQ_MASK_PARAM,
)
log = logging.getLogger(__name__)


def _safe_cqt(audio: np.ndarray, fmin: float, n_bins: int, bins_per_octave: int, hop_length: int) -> np.ndarray:
    """CQT with n_bins clipped to stay below Nyquist."""
    nyquist = SR / 2.0
    max_bins = int(bins_per_octave * np.log2(nyquist / fmin))
    actual = min(n_bins, max(1, max_bins))
    C = librosa.cqt(
        audio,
        sr=SR,
        fmin=fmin,
        n_bins=actual,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
    )
    C = np.abs(C)
    if actual < n_bins:
        pad = np.zeros((n_bins - actual, C.shape[1]), dtype=C.dtype)
        C = np.concatenate([C, pad], axis=0)
    return C


def extract_mel_local(audio: np.ndarray) -> np.ndarray:
    """Mel spectrogram (1, n_mels, T)."""
    p = MEL_PARAMS
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR,
        n_fft=p["n_fft"],
        hop_length=p["hop_length"],
        n_mels=p["n_mels"],
        fmin=p["fmin"],
        fmax=p["fmax"],
    )
    return np.log(S + 1e-6).astype(np.float32)[np.newaxis, ...]


def extract_cqt_local(audio: np.ndarray) -> np.ndarray:
    """CQT (1, n_bins, T)."""
    p = CQT_PARAMS
    nyquist = SR / 2.0
    max_bins = int(p["bins_per_octave"] * np.log2(nyquist / p["fmin"]))
    actual = min(p["n_bins"], max(1, max_bins))
    C = np.abs(librosa.cqt(
        audio, sr=SR,
        hop_length=p["hop_length"],
        fmin=p["fmin"],
        n_bins=actual,
        bins_per_octave=p["bins_per_octave"],
    ))
    if actual < p["n_bins"]:
        pad = np.zeros((p["n_bins"] - actual, C.shape[1]), dtype=C.dtype)
        C = np.concatenate([C, pad], axis=0)
    return np.log(C + 1e-6).astype(np.float32)[np.newaxis, ...]


def extract_hcqt_local(audio: np.ndarray) -> np.ndarray:
    """HCQT with config from exp3."""
    p = HCQT_PARAMS
    layers = []
    for h in p["harmonics"]:
        C = _safe_cqt(
            audio,
            fmin=p["fmin"] * h,
            n_bins=p["n_bins"],
            bins_per_octave=p["bins_per_octave"],
            hop_length=p["hop_length"],
        )
        layers.append(C)
    hcqt = np.stack(layers, axis=0)
    hcqt = np.log(hcqt + 1e-6).astype(np.float32)
    return hcqt


def get_extractor(representation: str):
    """Return feature extractor for representation: hcqt, mel, cqt."""
    if representation == "hcqt":
        return extract_hcqt_local
    if representation == "mel":
        return extract_mel_local
    if representation == "cqt":
        return extract_cqt_local
    raise ValueError("representation must be hcqt, mel, or cqt")


class SpecAugment:
    def __init__(self, freq_mask: int = FREQ_MASK_PARAM, time_mask: int = TIME_MASK_PARAM,
                 n_freq: int = NUM_FREQ_MASKS, n_time: int = NUM_TIME_MASKS):
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


class RecordedDataset(Dataset):
    def __init__(
        self,
        paths: List[Path],
        labels: List[int],
        segments_dir: Path,
        cache_dir: Optional[Path],
        representation: str = "hcqt",
        augment: Optional[SpecAugment] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.segments_dir = segments_dir
        self.cache_dir = cache_dir
        self.representation = representation
        self.extract = get_extractor(representation)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def _load_and_extract(self, path: Path) -> np.ndarray:
        full_path = self.segments_dir / path if not path.is_absolute() else path
        if not full_path.exists():
            full_path = path
        audio, _ = librosa.load(str(full_path), sr=SR, mono=True)
        return self.extract(audio)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        if self.cache_dir:
            h = hashlib.sha256(str(path).encode()).hexdigest()[:12]
            cache_path = self.cache_dir / f"rec_{self.representation}_{h}.npy"
            if cache_path.exists():
                feat = np.load(cache_path)
            else:
                feat = self._load_and_extract(path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, feat)
        else:
            feat = self._load_and_extract(path)
        if self.augment is not None:
            feat = self.augment(feat)
        return torch.from_numpy(feat), torch.tensor(self.labels[idx], dtype=torch.long)


def prepare_recorded_splits(
    metadata_path: Path,
    segments_dir: Path,
    train_positions: List[str],
    test_positions: List[str],
    val_fraction: float,
    split_by_file: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load metadata and split: train/val from train_positions (NF), test from test_positions (MF)."""
    df = pd.read_csv(metadata_path)
    if "path" not in df.columns or "position" not in df.columns or "interval" not in df.columns:
        raise ValueError(f"Recorded metadata must have columns path, position, interval. Got: {list(df.columns)}")

    df_train_pos = df[df["position"].isin(train_positions)].copy()
    df_test = df[df["position"].isin(test_positions)].copy()

    if df_train_pos.empty:
        raise ValueError(f"No rows with position in {train_positions}. Check metadata.")
    if df_test.empty:
        raise ValueError(f"No rows with position in {test_positions}. Check metadata.")

    if split_by_file and "source_file" in df.columns:
        files = df_train_pos["source_file"].unique()
        train_files, val_files = train_test_split(
            list(files), test_size=val_fraction, random_state=seed, shuffle=True
        )
        df_train = df_train_pos[df_train_pos["source_file"].isin(train_files)]
        df_val = df_train_pos[df_train_pos["source_file"].isin(val_files)]
    else:
        df_train, df_val = train_test_split(
            df_train_pos, test_size=val_fraction, random_state=seed, stratify=df_train_pos["interval"]
        )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def prepare_recorded_splits_by_class(
    metadata_path: Path,
    train_positions: List[str],
    test_positions: List[str],
    train_intervals: List[str],
    val_intervals: List[str],
    test_intervals: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by interval class."""
    df = pd.read_csv(metadata_path)
    if "path" not in df.columns or "position" not in df.columns or "interval" not in df.columns:
        raise ValueError(f"Recorded metadata must have columns path, position, interval. Got: {list(df.columns)}")

    df_nf = df[df["position"].isin(train_positions)]
    df_mf = df[df["position"].isin(test_positions)]

    df_train = df_nf[df_nf["interval"].isin(train_intervals)].copy()
    df_val = df_nf[df_nf["interval"].isin(val_intervals)].copy()
    df_test = df_mf[df_mf["interval"].isin(test_intervals)].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError(
            "Class split produced empty split. Check CLASS_SPLIT_* and metadata."
        )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def create_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    segments_dir: Path,
    cache_dir: Optional[Path],
    batch_size: int,
    representation: str = "hcqt",
    num_workers: int = 0,
    use_augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def to_paths_labels(d: pd.DataFrame) -> Tuple[List[Path], List[int]]:
        paths = [Path(p) for p in d["path"].tolist()]
        labels = [LABEL_TO_IDX[interval] for interval in d["interval"].tolist()]
        return paths, labels

    pt, lt = to_paths_labels(df_train)
    pv, lv = to_paths_labels(df_val)
    pte, lte = to_paths_labels(df_test)

    aug = SpecAugment() if use_augment else None
    ds_train = RecordedDataset(pt, lt, segments_dir, cache_dir, representation=representation, augment=aug)
    ds_val = RecordedDataset(pv, lv, segments_dir, cache_dir, representation=representation, augment=None)
    ds_test = RecordedDataset(pte, lte, segments_dir, cache_dir, representation=representation, augment=None)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_synthetic_test_loader(
    metadata_path: Path,
    dataset_dir: Path,
    cache_dir: Optional[Path],
    batch_size: int,
    test_instruments: List[str],
) -> DataLoader:
    """Build DataLoader for synthetic test set for --eval-synthetic."""
    df = pd.read_csv(metadata_path)
    if "path" not in df.columns or "interval" not in df.columns or "instrument" not in df.columns:
        raise ValueError(f"Synthetic metadata must have path, interval, instrument. Got: {list(df.columns)}")
    df_test = df[df["instrument"].isin(test_instruments)]
    if df_test.empty:
        raise ValueError(f"No rows with instrument in {test_instruments}.")
    paths = [Path(p) for p in df_test["path"].tolist()]
    labels = [LABEL_TO_IDX[interval] for interval in df_test["interval"].tolist()]
    ds = RecordedDataset(paths, labels, dataset_dir, cache_dir, augment=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
