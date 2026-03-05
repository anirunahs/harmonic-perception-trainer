"""
Dataset preparation for Experiment 2.

Handles feature-extraction pipelines per model family:

  1. PANNs:         log₁₀-mel, 64 bands, 32 kHz
  2. AST:           normalised fbank, 128 bands, 16 kHz
  3. Scratch Mel:   log-mel, 128 bands, 22 050 Hz
  4. Scratch CQT:   log-CQT (1, 252, T), 22 050 Hz
  5. Scratch HCQT:  log-HCQT (5, 252, T), 22 050 Hz

"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    AST_FBANK_MEAN,
    AST_FBANK_STD,
    AST_N_MELS,
    AST_SR,
    AST_TARGET_LENGTH,
    BASELINE_FMAX,
    BASELINE_FMIN,
    BASELINE_HOP,
    BASELINE_N_FFT,
    BASELINE_N_MELS,
    CACHE_DIR,
    CQT_BINS_PER_OCTAVE,
    CQT_FMIN,
    CQT_HOP,
    CQT_N_BINS,
    DATASET_DIR,
    FREQ_MASK_PARAM,
    HCQT_HARMONICS,
    LABEL_TO_IDX,
    METADATA_CSV,
    NUM_FREQ_MASKS,
    NUM_TIME_MASKS,
    PANNS_FMAX,
    PANNS_FMIN,
    PANNS_HOP,
    PANNS_N_FFT,
    PANNS_N_MELS,
    PANNS_SR,
    SR,
    SPLIT_INSTRUMENTS,
    TIME_MASK_PARAM,
    TRAIN_DEV_FRACTION,
)

log = logging.getLogger(__name__)


#  Feature extraction

def extract_panns_mel(audio: np.ndarray) -> np.ndarray:
    """Log₁₀-mel for PANNs CNN14 (1, T, 64)."""
    audio_32k = librosa.resample(audio, orig_sr=SR, target_sr=PANNS_SR)
    S = librosa.feature.melspectrogram(
        y=audio_32k, sr=PANNS_SR, n_fft=PANNS_N_FFT,
        hop_length=PANNS_HOP, n_mels=PANNS_N_MELS,
        fmin=PANNS_FMIN, fmax=PANNS_FMAX, power=2.0,
    )
    S_log = np.log10(np.maximum(S, 1e-10)).astype(np.float32)
    return S_log.T[np.newaxis, ...]                      # (1, T, 64)


def extract_ast_fbank(audio: np.ndarray) -> np.ndarray:
    """Normalised fbank for AST (target_length, 128)."""
    audio_16k = librosa.resample(audio, orig_sr=SR, target_sr=AST_SR)
    waveform = torch.FloatTensor(audio_16k).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=AST_SR,
        use_energy=False, window_type="hanning",
        num_mel_bins=AST_N_MELS, dither=0.0, frame_shift=10.0,
    )                                                    # (T, 128)

    fbank = (fbank - AST_FBANK_MEAN) / (AST_FBANK_STD * 2)

    n = fbank.shape[0]
    if n < AST_TARGET_LENGTH:
        pad = torch.zeros(AST_TARGET_LENGTH - n, AST_N_MELS)
        fbank = torch.cat([fbank, pad], dim=0)
    elif n > AST_TARGET_LENGTH:
        fbank = fbank[:AST_TARGET_LENGTH]

    return fbank.numpy().astype(np.float32)              # (T_fix, 128)


def extract_baseline_mel(audio: np.ndarray) -> np.ndarray:
    """Log-mel for baseline CNN (1, 128, T)."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=BASELINE_N_FFT,
        hop_length=BASELINE_HOP, n_mels=BASELINE_N_MELS,
        fmin=BASELINE_FMIN, fmax=BASELINE_FMAX,
    )
    S_db = np.log(S + 1e-6).astype(np.float32)
    return S_db[np.newaxis, ...]                         # (1, 128, T)


def _safe_cqt(audio: np.ndarray, fmin: float, n_bins: int,
              bins_per_octave: int, hop_length: int) -> np.ndarray:
    """CQT with n_bins clipped to stay below Nyquist."""
    max_bins = int(bins_per_octave * np.log2((SR / 2) / fmin))
    actual = min(n_bins, max(1, max_bins))
    C = np.abs(librosa.cqt(
        y=audio, sr=SR, hop_length=hop_length,
        fmin=fmin, n_bins=actual, bins_per_octave=bins_per_octave,
    ))
    if actual < n_bins:
        pad = np.zeros((n_bins - actual, C.shape[1]), dtype=C.dtype)
        C = np.concatenate([C, pad], axis=0)
    return C


def extract_scratch_cqt(audio: np.ndarray) -> np.ndarray:
    """Log-CQT for from-scratch CQT CNN (1, 252, T)."""
    C = np.abs(librosa.cqt(
        y=audio, sr=SR, hop_length=CQT_HOP,
        fmin=CQT_FMIN, n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    ))
    C_db = np.log(C + 1e-6).astype(np.float32)
    return C_db[np.newaxis, ...]                         # (1, 252, T)


def extract_scratch_hcqt(audio: np.ndarray) -> np.ndarray:
    """Log-HCQT for from-scratch HCQT CNN (5, 252, T)."""
    layers = []
    for h in HCQT_HARMONICS:
        C = _safe_cqt(
            audio, fmin=CQT_FMIN * h, n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE, hop_length=CQT_HOP,
        )
        layers.append(C)
    hcqt = np.stack(layers, axis=0).astype(np.float32)
    hcqt = np.log(hcqt + 1e-6).astype(np.float32)
    return hcqt                                          # (5, 252, T)


_EXTRACTORS = {
    "panns":        extract_panns_mel,
    "ast":          extract_ast_fbank,
    "scratch_mel":  extract_baseline_mel,
    "scratch_cqt":  extract_scratch_cqt,
    "scratch_hcqt": extract_scratch_hcqt,
}


#  SpecAugment (scratch baseline only)

class SpecAugment:
    """Frequency + time masking for spectrograms shaped (C, F, T)."""

    def __init__(self, freq_mask: int = FREQ_MASK_PARAM,
                 time_mask: int = TIME_MASK_PARAM,
                 n_freq: int = NUM_FREQ_MASKS,
                 n_time: int = NUM_TIME_MASKS):
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


#  PyTorch Dataset

class IntervalDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 augment: Optional[SpecAugment] = None):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.augment is not None:
            x = self.augment(x)
        return (torch.from_numpy(x).float(),
                torch.tensor(self.labels[idx], dtype=torch.long))


#  Split logic

def _assign_instrument_split(df: pd.DataFrame) -> pd.DataFrame:
    inv = {}
    for split, instruments in SPLIT_INSTRUMENTS.items():
        for inst in instruments:
            inv[inst] = split
    df = df.copy()
    df["split"] = df["instrument"].map(inv)
    unknown = df["split"].isna()
    if unknown.any():
        log.warning("Unmapped instruments: %s",
                    df.loc[unknown, "instrument"].unique().tolist())
        df.loc[unknown, "split"] = "train"
    return df


def _split_train_dev(df: pd.DataFrame, fraction: float, seed: int):
    train_df = df[df["split"] == "train"].copy()
    idx_core, idx_dev = train_test_split(
        train_df.index, test_size=fraction,
        stratify=train_df["interval"], random_state=seed,
    )
    return df.loc[idx_core], df.loc[idx_dev]


#  Feature extraction with disk cache

def _extract_features(df: pd.DataFrame, model_type: str):
    extractor = _EXTRACTORS[model_type]
    features, labels = [], []
    for _, row in df.iterrows():
        audio_path = DATASET_DIR / row["path"]
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        feat = extractor(y)
        features.append(feat)
        labels.append(LABEL_TO_IDX[row["interval"]])
    return np.stack(features), np.array(labels, dtype=np.int64)


def _load_or_extract(df, model_type, split_name, use_cache=True):
    cache_file = CACHE_DIR / f"{model_type}_{split_name}.npz"
    if use_cache and cache_file.exists():
        log.info("Loading cached: %s", cache_file)
        data = np.load(cache_file)
        return data["features"], data["labels"]

    log.info("Extracting %s features for %d samples (%s) ...",
             model_type, len(df), split_name)
    features, labels = _extract_features(df, model_type)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=features, labels=labels)
        log.info("Cached → %s", cache_file)
    return features, labels


#  Public API

SplitData = Dict[str, Tuple[np.ndarray, np.ndarray]]


def prepare_splits(
    model_type: str,
    seed: int = 42,
    use_cache: bool = True,
) -> SplitData:
    """Build four splits: train, train_dev, val, test."""
    df = pd.read_csv(METADATA_CSV)
    df = _assign_instrument_split(df)

    train_core_df, train_dev_df = _split_train_dev(
        df, TRAIN_DEV_FRACTION, seed)
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    log.info("Splits — train: %d  train_dev: %d  val: %d  test: %d",
             len(train_core_df), len(train_dev_df),
             len(val_df), len(test_df))

    splits: SplitData = {}
    for name, sub_df in [
        ("train", train_core_df), ("train_dev", train_dev_df),
        ("val", val_df), ("test", test_df),
    ]:
        splits[name] = _load_or_extract(sub_df, model_type, name, use_cache)
    return splits


def create_dataloaders(
    splits: SplitData,
    batch_size: int = 32,
    model_type: str = "panns",
) -> Dict[str, DataLoader]:
    """Wrap feature arrays in DataLoaders.  SpecAugment is applied
    to the train loader for all from-scratch representations (mel, CQT, HCQT)."""
    use_augment = model_type in ("scratch_mel", "scratch_cqt", "scratch_hcqt")
    augment = SpecAugment() if use_augment else None
    loaders: Dict[str, DataLoader] = {}
    for name, (feats, labels) in splits.items():
        aug = augment if name == "train" else None
        ds = IntervalDataset(feats, labels, augment=aug)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(name == "train"), drop_last=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
    return loaders
