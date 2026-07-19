"""Training and evaluation for one sentence-level fold."""

import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import SentenceDataset
from .features import N_FAMILIES
from .metrics import classification_metrics
from .model import MultimodalClassifier
from .utils import set_seed


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_optimizer(model, cfg):
    encoder, task = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if name.startswith("text_encoder.") else task).append(parameter)
    groups = [{"params": task, "lr": cfg.head_lr}]
    if encoder:
        groups.append({"params": encoder, "lr": cfg.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _grad_scaler(use_amp):
    """Use the current AMP API while retaining compatibility with older torch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast(device, use_amp):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def task_initialization_fingerprint(model):
    """Hash the randomly initialized task modules without copying LaBSE."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("text_encoder."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _summarize_diagnostics(diagnostics):
    if not diagnostics:
        return None
    summary = {}
    for name in [
        "text_embedding_norm",
        "eeg_embedding_norm",
        "candidate_eeg_contribution_norm",
        "gated_eeg_contribution_norm",
    ]:
        values = diagnostics[name]
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    logit_delta = (
        diagnostics["logits_with_eeg"] - diagnostics["logits_without_eeg"]
    )
    changed = (
        diagnostics["logits_with_eeg"].argmax(axis=-1)
        != diagnostics["logits_without_eeg"].argmax(axis=-1)
    )
    summary["logit_delta_l2"] = {
        "mean": float(np.linalg.norm(logit_delta, axis=-1).mean()),
        "std": float(np.linalg.norm(logit_delta, axis=-1).std()),
        "min": float(np.linalg.norm(logit_delta, axis=-1).min()),
        "max": float(np.linalg.norm(logit_delta, axis=-1).max()),
    }
    summary["prediction_changed_without_eeg_count"] = int(changed.sum())
    summary["prediction_changed_without_eeg_rate"] = float(changed.mean())
    return summary


@torch.no_grad()
def evaluate(model, loader, device, criterion, collect_diagnostics=False):
    model.eval()
    total_loss = 0.0
    predictions, targets, indices = [], [], []
    diagnostic_batches = {}
    for batch in loader:
        batch = _move(batch, device)
        labels = batch.pop("labels")
        sentence_index = batch.pop("sentence_index")
        if collect_diagnostics:
            logits, _, batch_diagnostics = model(
                **batch, return_diagnostics=True
            )
            for name, values in batch_diagnostics.items():
                diagnostic_batches.setdefault(name, []).append(
                    values.detach().float().cpu().numpy()
                )
        else:
            logits, _ = model(**batch)
        total_loss += criterion(logits, labels).item() * len(labels)
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(labels.cpu().numpy())
        indices.append(sentence_index.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    indices = np.concatenate(indices)
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(targets)
    if collect_diagnostics:
        diagnostics = {
            name: np.concatenate(values, axis=0)
            for name, values in diagnostic_batches.items()
        }
        return metrics, predictions, targets, indices, diagnostics
    return metrics, predictions, targets, indices


def train_fold(
    setup,
    cfg,
    encodings,
    eeg,
    subject_mask,
    labels,
    split_indices,
    device=None,
    initialization_seed=None,
):
    train_indices, val_indices, test_indices = split_indices
    device = device or pick_device()
    datasets = [
        SentenceDataset(encodings, eeg, subject_mask, labels, indices)
        for indices in split_indices
    ]
    train_loader = _loader(datasets[0], cfg.batch_size, True, cfg.num_workers)
    val_loader = _loader(datasets[1], cfg.batch_size, False, cfg.num_workers)
    test_loader = _loader(datasets[2], cfg.batch_size, False, cfg.num_workers)

    if initialization_seed is not None:
        set_seed(initialization_seed)
    model = MultimodalClassifier(
        model_name=cfg.model_name,
        fusion=setup.fusion,
        text_mode=setup.text_mode,
        n_channels=eeg.shape[-1] // N_FAMILIES,
        text_dim=cfg.text_dim,
        channel_dim=cfg.channel_dim,
        eeg_dim=cfg.eeg_dim,
        dropout=cfg.dropout,
        zero_eeg_contribution=setup.zero_eeg_contribution,
    ).to(device)
    initialization_fingerprint = task_initialization_fingerprint(model)
    optimizer = build_optimizer(model, cfg)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = _grad_scaler(use_amp)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = _move(batch, device)
            labels_batch = batch.pop("labels")
            batch.pop("sentence_index")
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits, _ = model(**batch)
                loss = criterion(logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * len(labels_batch)
            seen += len(labels_batch)

        val_metrics, _, _, _ = evaluate(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_val_f1 + 1e-8:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    model.load_state_dict(best_state)
    if setup.fusion == "gated":
        test_metrics, predictions, targets, indices, diagnostics = evaluate(
            model,
            test_loader,
            device,
            criterion,
            collect_diagnostics=True,
        )
    else:
        test_metrics, predictions, targets, indices = evaluate(
            model, test_loader, device, criterion
        )
        diagnostics = None
    result = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test": test_metrics,
        "history": history,
        "gate_mean": model.gate_mean(),
        "gate_values": model.gate_values(),
        "diagnostics": _summarize_diagnostics(diagnostics),
        "initialization": {
            "seed": initialization_seed,
            "task_sha256": initialization_fingerprint,
        },
        "n_train": int(len(train_indices)),
        "n_val": int(len(val_indices)),
        "n_test": int(len(test_indices)),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, predictions, targets, indices, diagnostics
