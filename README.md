# ZuCo multimodal sentiment classification

Can sentence-level EEG features improve a strong text sentiment classifier?
This project combines LaBSE sentence representations with the classical EEG
features extracted from ZuCo Task 1. It keeps the text-only, EEG-only, fusion,
and control experiments in one evaluation pipeline so their scores are directly
comparable.

The main model uses LaBSE's `[CLS]` representation and a compact EEG set encoder.
Each subject's 2,520 classical features are reshaped into 105 electrodes x 24
feature families. A shared MLP encodes the electrodes, attention pools them into
one subject representation, and a masked mean combines the available subjects
for the sentence. The text and EEG representations are then fused for ternary
sentiment classification.

## Experiment setups

| setup | text branch | EEG branch | purpose |
| --- | --- | --- | --- |
| `text_frozen` | frozen LaBSE + CLS | no | frozen text baseline |
| `text_finetune` | fine-tuned LaBSE + CLS | no | primary text baseline |
| `eeg_only` | no | aligned EEG | EEG-only baseline |
| `concat_finetune` | fine-tuned | aligned EEG | late hidden-state concatenation |
| `gated_finetune` | fine-tuned | aligned EEG | gated residual fusion |
| `gated_shuffled_finetune` | fine-tuned | shuffled inside each split | alignment control |
| `gated_noise_finetune` | fine-tuned | matched random noise | regularization control |

Frozen variants of both fusion models are also available from the command line.
The gated model starts with a small EEG contribution, so it behaves approximately
like the text model until the validation data supports using the second modality.

## Evaluation protocol

The 400 unique sentences are the units of evaluation. The outer stratified folds
hold out complete sentences; every subject's EEG recording of a sentence follows
that sentence into the same train, validation, or test partition. The test fold
is evaluated once, after the best epoch has been selected by validation macro-F1.

All EEG preprocessing is fitted inside each fold using training sentences only:

1. median imputation;
2. per-feature standardization;
3. transformation of validation and test EEG with the stored training values.

The shuffled-EEG control permutes recordings independently inside train,
validation, and test. It destroys text-EEG alignment without moving any EEG row
across a split boundary.

Accuracy, macro-F1, weighted F1, per-class F1, confusion matrices, fold histories,
and out-of-fold predictions are saved. Fusion models are compared with the
matching text baseline using a paired sentence-level bootstrap interval.

## Classical feature cache

`extract_features.py` reads the original `results*_SR.mat` files and writes one
compressed `.npz` per subject. Each EEG row contains:

- 16 raw-signal statistics for each of 105 electrodes;
- ZuCo's eight sentence-level band means (`t1`, `t2`, `a1`, `a2`, `b1`, `b2`,
  `g1`, `g2`) for each electrode.

The default line-length feature is mean absolute successive difference rather
than cumulative path length. This removes its direct dependence on the number of
time samples in a sentence. Pass `--line-length sum` to reproduce the earlier
cumulative definition as a sensitivity analysis.

The raw files, sentence labels, and cache belong on Drive rather than in the
repository. The Colab notebook reads the fixed labels CSV from the main Thesis
data folder and the subject files from its `zuco_og_raw` subfolder. It uses a
versioned feature folder, so extraction is performed once and reused.

## Colab

Open `notebooks/zuco_multimodal_colab.ipynb` in a GPU runtime and run it from top
to bottom. Each section begins with a short description of what it will read,
run, and save. The notebook:

1. mounts Drive and checks the runtime;
2. clones or updates this repository and installs its requirements;
3. defines the data, cache, result, and run-version paths;
4. builds or resumes the classical feature cache;
5. validates text/EEG alignment;
6. runs a short EEG-only smoke test;
7. launches the complete experiment suite;
8. displays the saved summary and plots.

`RUN_TAG` is the version of an experiment. Re-running the same tag skips completed
setup/seed files and resumes the missing work. Change the tag when a model or
training setting changes. A run tag cannot silently mix incompatible training
settings.

## Command-line use

Install the dependencies:

```bash
pip install -r requirements.txt
```

Build the reusable feature cache:

```bash
python extract_features.py \
  --mat-dir /path/to/zuco_og_raw \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --out-dir /path/to/drive/zuco_classical_features/classical_v2_normalized
```

Validate the cache:

```bash
python inspect_data.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/drive/zuco_classical_features/classical_v2_normalized
```

Run the default three-seed suite:

```bash
python run.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/drive/zuco_classical_features/classical_v2_normalized \
  --results-base /path/to/drive/zuco_multimodal_results \
  --run-tag v1_full
```

A narrower experiment is selected with `--setups` and `--seeds`:

```bash
python run.py \
  --labels-csv /path/to/Thesis/Data/zuco_sentiment_labels_task1_fixed.csv \
  --features-dir /path/to/features \
  --results-base /path/to/results \
  --run-tag v1_gated \
  --setups text_finetune gated_finetune gated_shuffled_finetune \
  --seeds 42 52 62
```

## Drive result layout

Every setup and seed is saved immediately and independently:

```text
<results-base>/
  v1_full/
    run_manifest.json
    text_finetune/
      seed_42.json
      seed_52.json
      seed_62.json
    gated_finetune/
      seed_42.json
      ...
    tables/
      summary.csv
      summary.json
      summary.md
    plots/
      scores.png
      confusions.png
```

The JSON files contain the complete fold histories and out-of-fold predictions,
so tables and plots can be regenerated without retraining:

```bash
python plot_results.py --run-dir /path/to/results/v1_full
```

## Repository layout

```text
extract_features.py       original .mat files -> reusable subject caches
inspect_data.py           cache alignment and data summary
run.py                    resumable cross-validation suite
plot_results.py           rebuild tables and plots from saved JSON
src/features.py           classical EEG statistics
src/zuco_io.py            MATLAB/HDF5 reader
src/data.py               sentence grouping, fold preprocessing, controls
src/model.py              LaBSE, EEG set encoder, and fusion heads
src/engine.py             fold training and best-epoch evaluation
src/experiment.py         setup/seed orchestration
src/reporting.py          summaries, bootstrap deltas, and plots
notebooks/                minimal Colab runner
tests/                    feature, split, preprocessing, and model checks
```
