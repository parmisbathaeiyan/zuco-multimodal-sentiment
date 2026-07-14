"""Cross-validation orchestration and resumable result files."""

import os

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import parse_setup
from .data import FoldPreprocessor, apply_eeg_control, fold_indices
from .engine import pick_device, train_fold
from .metrics import classification_metrics
from .utils import load_json, save_json, set_seed


def result_path(run_dir, setup_name, seed):
    return os.path.join(run_dir, setup_name, f"seed_{seed}.json")


def run_one_seed(data, setup_name, seed, cfg, run_dir, overwrite=False):
    path = result_path(run_dir, setup_name, seed)
    if os.path.exists(path) and not overwrite:
        print(f"skip {setup_name}, seed {seed} (already complete)")
        return load_json(path)

    set_seed(seed)
    setup = parse_setup(setup_name)
    if setup.uses_text:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        encodings = tokenizer(
            data.sentences,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
    else:
        # Keep one harmless token so the shared dataset/collation path stays the same.
        encodings = {
            "input_ids": torch.zeros((len(data.labels), 1), dtype=torch.long),
            "attention_mask": torch.ones((len(data.labels), 1), dtype=torch.long),
        }
    splits = list(fold_indices(data.labels, cfg.n_folds, cfg.val_size, seed))
    oof_predictions = np.full(len(data.labels), -1, dtype=int)
    oof_targets = np.full(len(data.labels), -1, dtype=int)
    fold_results = []
    device = pick_device()

    print(f"{setup_name}, seed {seed}, device={device}")
    for fold, split in enumerate(splits, start=1):
        fold_seed = seed + fold * 1000
        set_seed(fold_seed)
        controlled_eeg, controlled_mask = apply_eeg_control(
            data.eeg,
            data.subject_mask,
            split,
            setup.eeg_control,
            fold_seed,
        )
        preprocessor = FoldPreprocessor().fit(controlled_eeg, controlled_mask, split[0])
        prepared_eeg = preprocessor.transform(controlled_eeg, controlled_mask)
        fold_result, predictions, targets, indices = train_fold(
            setup=setup,
            cfg=cfg,
            encodings=encodings,
            eeg=prepared_eeg,
            subject_mask=controlled_mask,
            labels=data.labels,
            split_indices=split,
            device=device,
        )
        fold_result["fold"] = fold
        fold_result["n_all_missing_train_features"] = (
            preprocessor.n_all_missing_features
        )
        fold_results.append(fold_result)
        oof_predictions[indices] = predictions
        oof_targets[indices] = targets
        print(
            f"  fold {fold}/{cfg.n_folds}: "
            f"acc {fold_result['test']['accuracy']:.3f}, "
            f"macro-F1 {fold_result['test']['macro_f1']:.3f}, "
            f"epoch {fold_result['best_epoch']}, "
            f"all-missing train features {preprocessor.n_all_missing_features}"
        )

    if (oof_predictions < 0).any() or (oof_targets < 0).any():
        raise RuntimeError("out-of-fold predictions are incomplete")
    pooled = classification_metrics(oof_targets, oof_predictions)
    accuracies = [fold["test"]["accuracy"] for fold in fold_results]
    macro_f1s = [fold["test"]["macro_f1"] for fold in fold_results]
    result = {
        "setup": setup_name,
        "seed": seed,
        "model_name": cfg.model_name,
        "fusion": setup.fusion,
        "text_mode": setup.text_mode,
        "eeg_control": setup.eeg_control,
        "n_folds": cfg.n_folds,
        "fold_accuracy_mean": float(np.mean(accuracies)),
        "fold_accuracy_std": float(np.std(accuracies)),
        "fold_macro_f1_mean": float(np.mean(macro_f1s)),
        "fold_macro_f1_std": float(np.std(macro_f1s)),
        "oof": pooled,
        "folds": fold_results,
        "predictions": [
            {
                "sentence_id": int(sentence_id),
                "target": int(target),
                "prediction": int(prediction),
            }
            for sentence_id, target, prediction in zip(
                data.sentence_ids, oof_targets, oof_predictions
            )
        ],
    }
    save_json(result, path)
    print(
        f"  => OOF acc {pooled['accuracy']:.3f}, "
        f"macro-F1 {pooled['macro_f1']:.3f} -> {path}"
    )
    return result


def run_experiments(data, setup_names, seeds, cfg, run_dir, overwrite=False):
    completed = []
    for setup_name in setup_names:
        for seed in seeds:
            completed.append(
                run_one_seed(
                    data=data,
                    setup_name=setup_name,
                    seed=seed,
                    cfg=cfg,
                    run_dir=run_dir,
                    overwrite=overwrite,
                )
            )
    return completed
