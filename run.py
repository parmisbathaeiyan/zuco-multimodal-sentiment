"""Run the sentence-level ZuCo multimodal experiment suite."""

import argparse
import os

from src.config import DEFAULT_SETUPS, VALID_SETUPS, TrainConfig
from src.data import load_multimodal_data
from src.experiment import run_experiments
from src.reporting import build_summary
from src.utils import auto_run_tag, load_json, save_json


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="LaBSE + classical EEG sentiment experiments.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--results-base", required=True)
    parser.add_argument("--run-tag", default=None, help="defaults to a timestamped run folder")
    parser.add_argument("--setups", nargs="+", choices=VALID_SETUPS, default=DEFAULT_SETUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--n-folds", type=int, default=defaults.n_folds)
    parser.add_argument("--val-size", type=float, default=defaults.val_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--encoder-lr", type=float, default=defaults.encoder_lr)
    parser.add_argument("--head-lr", type=float, default=defaults.head_lr)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.run_tag or auto_run_tag()
    run_dir = os.path.join(args.results_base, tag)
    os.makedirs(run_dir, exist_ok=True)
    cfg = TrainConfig(
        model_name=args.model_name,
        n_folds=args.n_folds,
        val_size=args.val_size,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        dropout=args.dropout,
        num_workers=args.num_workers,
    )
    data = load_multimodal_data(args.labels_csv, args.features_dir)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = {
        "run_tag": tag,
        "run_dir": run_dir,
        "setups": args.setups,
        "seeds": args.seeds,
        "training": cfg.to_dict(),
        "data": data.summary(),
        "paths": {
            "labels_csv": args.labels_csv,
            "features_dir": args.features_dir,
        },
    }
    if os.path.exists(manifest_path):
        existing = load_json(manifest_path)
        if (
            existing.get("training") != manifest["training"]
            or existing.get("paths") != manifest["paths"]
            or existing.get("data") != manifest["data"]
        ):
            raise SystemExit(
                "this run tag already uses different training settings or data; "
                "choose a new --run-tag"
            )
        manifest["setups"] = sorted(set(existing.get("setups", [])) | set(args.setups))
        manifest["seeds"] = sorted(set(existing.get("seeds", [])) | set(args.seeds))
    save_json(manifest, manifest_path)
    print(f"run directory: {run_dir}")
    print(f"setups: {', '.join(args.setups)}")
    print(f"seeds: {args.seeds}")
    run_experiments(
        data=data,
        setup_names=args.setups,
        seeds=args.seeds,
        cfg=cfg,
        run_dir=run_dir,
        overwrite=args.overwrite,
    )
    summary = build_summary(run_dir, bootstrap_samples=args.bootstrap_samples)
    print("\n" + summary.to_string(index=False))
    print(f"\ncomplete -> {run_dir}")


if __name__ == "__main__":
    main()
