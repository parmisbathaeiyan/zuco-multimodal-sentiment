"""Load cached EEG features and build leakage-safe sentence datasets."""

import glob
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from .config import (
    LABEL_TO_ID,
    ZUCO_REFERENCE_CHANNEL_INDEX,
    ZUCO_REFERENCE_CHANNEL_LABEL,
)
from .features import N_FAMILIES
from .labels import load_labels


@dataclass
class MultimodalData:
    sentence_ids: np.ndarray
    sentences: list
    labels: np.ndarray
    eeg: np.ndarray
    subject_mask: np.ndarray
    subjects: list
    feature_names: list
    n_channels: int
    dropped_reference_features: list

    def summary(self):
        counts = np.bincount(self.labels, minlength=3)
        per_sentence = self.subject_mask.sum(axis=1)
        usable_rows = self.eeg[self.subject_mask]
        observed_features = np.isfinite(usable_rows).any(axis=0)
        all_missing_features = [
            name
            for name, observed in zip(self.feature_names, observed_features)
            if not observed
        ]
        return {
            "n_sentences": int(len(self.labels)),
            "n_subjects": int(len(self.subjects)),
            "n_features": int(self.eeg.shape[-1]),
            "dropped_reference_channel": (
                {
                    "index": ZUCO_REFERENCE_CHANNEL_INDEX,
                    "label": ZUCO_REFERENCE_CHANNEL_LABEL,
                    "n_features": len(self.dropped_reference_features),
                }
                if self.dropped_reference_features
                else None
            ),
            "n_globally_all_missing_features": len(all_missing_features),
            "globally_all_missing_features": all_missing_features,
            "n_channels": int(self.n_channels),
            "n_feature_families": int(N_FAMILIES),
            "usable_subject_sentence_rows": int(self.subject_mask.sum()),
            "subjects_per_sentence_min": int(per_sentence.min()),
            "subjects_per_sentence_max": int(per_sentence.max()),
            "subjects_per_sentence_mean": float(per_sentence.mean()),
            "class_counts": {
                "negative": int(counts[0]),
                "neutral": int(counts[1]),
                "positive": int(counts[2]),
            },
            "subjects": self.subjects,
        }


def load_multimodal_data(labels_csv, features_dir):
    labels = load_labels(labels_csv)
    with open(os.path.join(features_dir, "feature_names.json")) as handle:
        cached_names = json.load(handle)
    if len(cached_names) % N_FAMILIES:
        raise ValueError(
            f"{len(cached_names)} features cannot be reshaped into {N_FAMILIES} families"
        )

    reference_suffix = f"_ch{ZUCO_REFERENCE_CHANNEL_INDEX}"
    keep_features = np.asarray(
        [not name.endswith(reference_suffix) for name in cached_names],
        dtype=bool,
    )
    dropped_reference_features = [
        name for name, keep in zip(cached_names, keep_features) if not keep
    ]
    if len(dropped_reference_features) not in (0, N_FAMILIES):
        raise ValueError(
            "reference-channel feature block is incomplete: "
            f"expected {N_FAMILIES}, found {len(dropped_reference_features)}"
        )
    names = [name for name, keep in zip(cached_names, keep_features) if keep]
    n_channels = len(names) // N_FAMILIES

    files = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no subject .npz files found in {features_dir}")
    subjects = [os.path.basename(path).rsplit(".", 1)[0] for path in files]

    sentence_ids = labels["sentence_id"].to_numpy()
    id_to_row = {sentence_id: row for row, sentence_id in enumerate(sentence_ids)}
    eeg = np.full((len(labels), len(files), len(names)), np.nan, dtype=np.float32)
    mask = np.zeros((len(labels), len(files)), dtype=bool)

    for subject_index, path in enumerate(files):
        cached = np.load(path, allow_pickle=True)
        if cached["X"].shape[1] != len(cached_names):
            raise ValueError(f"feature width mismatch in {path}")
        for cache_row, sentence_id in enumerate(cached["sentence_id"].astype(int)):
            row = id_to_row.get(int(sentence_id))
            if row is None:
                continue
            expected = int(labels.iloc[row]["sentiment_label"])
            observed = int(cached["label"][cache_row])
            if expected != observed:
                raise ValueError(
                    f"label mismatch for sentence {sentence_id} in {path}: "
                    f"{observed} != {expected}"
                )
            values = cached["X"][cache_row, keep_features].astype(np.float32)
            if np.isfinite(values).any():
                eeg[row, subject_index] = values
                mask[row, subject_index] = True

    mapped_labels = labels["sentiment_label"].map(LABEL_TO_ID)
    if mapped_labels.isna().any():
        raise ValueError("labels must be -1, 0, or 1")
    if not mask.any(axis=1).all():
        missing = sentence_ids[~mask.any(axis=1)]
        raise ValueError(f"sentences with no EEG recordings: {missing.tolist()}")

    return MultimodalData(
        sentence_ids=sentence_ids,
        sentences=labels["sentence"].astype(str).tolist(),
        labels=mapped_labels.astype(int).to_numpy(),
        eeg=eeg,
        subject_mask=mask,
        subjects=subjects,
        feature_names=names,
        n_channels=n_channels,
        dropped_reference_features=dropped_reference_features,
    )


class FoldPreprocessor:
    """Median imputation and scaling fitted on training sentences only."""

    def fit(self, eeg, subject_mask, train_indices):
        rows = eeg[train_indices][subject_mask[train_indices]]
        if not len(rows):
            raise ValueError("training fold has no EEG rows")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(rows, axis=0)
        self.observed_features = np.isfinite(median)
        self.n_all_missing_features = int((~self.observed_features).sum())
        self.median = np.where(self.observed_features, median, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(rows), rows, self.median)
        self.mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[~np.isfinite(self.std) | (self.std < 1e-8)] = 1.0
        return self

    def transform(self, eeg, subject_mask):
        filled = np.where(np.isfinite(eeg), eeg, self.median)
        scaled = ((filled - self.mean) / self.std).astype(np.float32)
        # A feature unseen in the training fold carries no estimable signal.
        scaled[..., ~self.observed_features] = 0.0
        scaled[~subject_mask] = 0.0
        return scaled


def fold_indices(labels, n_folds, val_size, seed):
    """Yield sentence-level train, validation, and test indices."""
    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_indices, test_indices in splitter.split(np.zeros(len(labels)), labels):
        train_indices, val_indices = train_test_split(
            fit_indices,
            test_size=val_size,
            stratify=labels[fit_indices],
            random_state=seed,
        )
        yield np.asarray(train_indices), np.asarray(val_indices), np.asarray(test_indices)


def apply_eeg_control(eeg, subject_mask, split_indices, control, seed):
    """Destroy EEG alignment without moving data across fold boundaries."""
    controlled = eeg.copy()
    controlled_mask = subject_mask.copy()
    if control == "aligned":
        return controlled, controlled_mask
    rng = np.random.default_rng(seed)
    if control == "shuffled":
        for indices in split_indices:
            permutation = indices[rng.permutation(len(indices))]
            controlled[indices] = eeg[permutation]
            controlled_mask[indices] = subject_mask[permutation]
        return controlled, controlled_mask
    if control == "noise":
        for indices in split_indices:
            controlled[indices] = rng.standard_normal(controlled[indices].shape).astype(np.float32)
        return controlled, controlled_mask
    raise ValueError(f"unknown EEG control {control!r}")


class SentenceDataset(Dataset):
    def __init__(self, encodings, eeg, subject_mask, labels, indices):
        self.encodings = encodings
        self.eeg = torch.from_numpy(eeg[indices])
        self.subject_mask = torch.from_numpy(subject_mask[indices])
        self.labels = torch.from_numpy(labels[indices]).long()
        self.indices = torch.from_numpy(np.asarray(indices)).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, row):
        item = {key: value[self.indices[row]] for key, value in self.encodings.items()}
        item["eeg"] = self.eeg[row]
        item["subject_mask"] = self.subject_mask[row]
        item["labels"] = self.labels[row]
        item["sentence_index"] = self.indices[row]
        return item
