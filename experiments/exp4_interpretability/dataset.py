"""
Datasets for Experiment 4: HCQT and FFT numerical features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import (
    LABEL_TO_IDX,
    RECORDED_METADATA,
    RECORDED_DIR,
    RECORDED_VAL_FRACTION,
    RECORDED_SPLIT_BY_FILE,
    SYNTHETIC_METADATA,
    SYNTHETIC_DIR,
    SPLIT_INSTRUMENTS,
    TRAIN_POSITIONS,
    TEST_POSITIONS,
    SR,
    BATCH_SIZE,
    TRAIN_DEV_FRACTION,
    FFT_FEATURE_DIM,
)
from .fft_features import extract_fft_features

log = logging.getLogger(__name__)


# FFT feature extraction and splits

def _assign_synthetic_split(df: pd.DataFrame) -> pd.DataFrame:
    inv = {}
    for split, instruments in SPLIT_INSTRUMENTS.items():
        for inst in instruments:
            inv[inst] = split
    df = df.copy()
    df["split"] = df["instrument"].map(inv)
    df.loc[df["split"].isna(), "split"] = "train"
    return df


def _load_audio(path: Path, base_dir: Path) -> np.ndarray:
    full = base_dir / path if not path.is_absolute() else path
    if not full.exists():
        full = path
    y, _ = __import__("librosa").load(str(full), sr=SR, mono=True)
    return y


def extract_fft_splits_synthetic(
    metadata_path: Path,
    dataset_dir: Path,
    seed: int,
    train_dev_fraction: float = TRAIN_DEV_FRACTION,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Extract FFT splits for synthetic dataset."""
    df = pd.read_csv(metadata_path)
    df = _assign_synthetic_split(df)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    train_core, train_dev = train_test_split(
        train_df.index, test_size=train_dev_fraction, stratify=train_df["interval"], random_state=seed
    )
    train_df = df.loc[train_core]

    def extract_for(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for _, row in sub.iterrows():
            audio = _load_audio(Path(row["path"]), dataset_dir)
            feat = extract_fft_features(audio, sr=SR)
            X_list.append(feat)
            y_list.append(LABEL_TO_IDX[row["interval"]])
        return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)

    return extract_for(train_df), extract_for(df.loc[val_df.index]), extract_for(test_df)


def extract_fft_splits_recorded(
    metadata_path: Path,
    segments_dir: Path,
    val_fraction: float,
    split_by_file: bool,
    seed: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Extract FFT splits for recorded dataset."""
    df = pd.read_csv(metadata_path)
    df_train_pos = df[df["position"].isin(TRAIN_POSITIONS)]
    df_test = df[df["position"].isin(TEST_POSITIONS)]
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

    def extract_for(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for _, row in sub.iterrows():
            audio = _load_audio(Path(row["path"]), segments_dir)
            feat = extract_fft_features(audio, sr=SR)
            X_list.append(feat)
            y_list.append(LABEL_TO_IDX[row["interval"]])
        return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)

    return extract_for(df_train), extract_for(df_val), extract_for(df_test)


def create_fft_dataloaders(
    train: Tuple[np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    batch_size: int = BATCH_SIZE,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create FFT dataloaders."""
    ds_t = TensorDataset(torch.from_numpy(train[0]), torch.from_numpy(train[1]))
    ds_v = TensorDataset(torch.from_numpy(val[0]), torch.from_numpy(val[1]))
    ds_te = TensorDataset(torch.from_numpy(test[0]), torch.from_numpy(test[1]))
    train_loader = DataLoader(ds_t, batch_size=batch_size, shuffle=shuffle_train, drop_last=False)
    val_loader = DataLoader(ds_v, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# HCQT loaders

def get_synthetic_hcqt_splits(seed: int = 42, use_cache: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get synthetic HCQT splits."""
    from experiments.exp1_representations.dataset import prepare_splits
    return prepare_splits("hcqt", seed=seed, use_cache=use_cache)


def get_recorded_hcqt_loaders(
    seed: int = 42,
    batch_size: int = BATCH_SIZE,
    representation: str = "hcqt",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get recorded HCQT loaders."""
    from experiments.exp3_microphone.dataset import (
        prepare_recorded_splits,
        create_dataloaders,
    )
    df_train, df_val, df_test = prepare_recorded_splits(
        RECORDED_METADATA,
        RECORDED_DIR,
        TRAIN_POSITIONS,
        TEST_POSITIONS,
        RECORDED_VAL_FRACTION,
        RECORDED_SPLIT_BY_FILE,
        seed,
    )
    cache_dir = Path(__file__).resolve().parent / ".cache" / "recorded"
    return create_dataloaders(
        df_train, df_val, df_test,
        RECORDED_DIR,
        cache_dir,
        batch_size=batch_size,
        representation=representation,
        num_workers=0,
        use_augment=False,
    )


def get_synthetic_hcqt_test_loader(
    seed: int = 42,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Get synthetic HCQT test loader."""
    splits = get_synthetic_hcqt_splits(seed=seed)
    feats, labels = splits["test"]
    ds = TensorDataset(torch.from_numpy(feats), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
