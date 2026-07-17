# Project log

This log is chronological. It records the motivation, design decisions,
implementation changes, trial runs, observed problems, and their outcomes.
Future entries should be appended at the bottom so the order of work remains
visible.

## 2026-07-14 — Define the first ZuCo multimodal experiment

### Starting point

The text-only LaBSE project and the classical EEG-feature project were used as
the starting baselines. The classical features alone had not produced useful
sentiment classification performance, so the next question became whether they
could add information when used as a second modality beside text.

The first scope was deliberately limited to ZuCo Task 1. TeCo was left for a
later stage. LaBSE `[CLS]` remained the text representation, while both frozen
and fine-tuned text baselines were kept because the better choice was not yet
known.

### Data decisions

- The unit of prediction is one of the 400 unique ZuCo sentences.
- The existing labels file is used directly from
  `Thesis/Data/zuco_sentiment_labels_task1_fixed.csv`.
- The original subject `results*_SR.mat` files stay in
  `Thesis/Data/zuco_og_raw`.
- Raw data, extracted data, models, and results are not committed to GitHub.
- A duplicate labels copy created during setup was removed; the notebook uses
  the existing file rather than maintaining another copy.

### Classical EEG representation

The reusable cache initially contains 2,520 features per subject-sentence row:
105 channels x 24 feature families.

The 24 families are:

- 16 raw-signal statistics: mean, standard deviation, variance, minimum,
  maximum, peak-to-peak range, median, interquartile range, skewness, kurtosis,
  RMS, mean absolute value, line length, zero-crossing rate, Hjorth mobility,
  and Hjorth complexity;
- eight ZuCo sentence-level band means: `t1`, `t2`, `a1`, `a2`, `b1`, `b2`,
  `g1`, and `g2`.

Line length was defined as mean absolute successive difference instead of the
cumulative path length. This makes it less directly dependent on sentence
duration. The old cumulative definition remains available as a sensitivity
analysis.

### Architecture version 1

The first multimodal architecture was implemented as follows:

1. LaBSE produces the sentence's first-token (`[CLS]`) representation.
2. Layer normalization, dropout, and a projection reduce the text branch to
   128 dimensions.
3. Each subject's flat EEG vector is reshaped into electrode tokens with 24
   feature families per electrode.
4. A shared MLP encodes every electrode. A learned electrode embedding supplies
   channel identity.
5. Learned attention pools the electrodes into one representation per subject.
6. A masked mean pools the available subjects for the sentence, so missing
   subject recordings do not become artificial zeros in the subject average.
7. The EEG representation is projected to 64 dimensions.

Two fusion approaches were included:

- `concat`: concatenate the projected text and EEG representations, then apply
  a fusion MLP;
- `gated`: project EEG into the text space and add it through a learned
  elementwise residual gate. The gate starts near 0.12, so the initial model is
  close to the text baseline and must learn to use EEG.

The default comparison suite was:

- `text_frozen`;
- `text_finetune`;
- `eeg_only`;
- `concat_finetune`;
- `gated_finetune`;
- `gated_shuffled_finetune`;
- `gated_noise_finetune`.

Frozen fusion variants were also made available from the command line.

### Controls and evaluation protocol

- Outer stratified folds split complete sentences, not subject-sentence rows.
  Every subject recording for a sentence stays in the same partition.
- Fifteen percent of each training portion is used for validation.
- Median imputation and standardization are fitted only on training-fold EEG.
- The best epoch is selected by validation macro-F1, and the test fold is then
  evaluated once.
- Shuffled EEG is permuted separately inside train, validation, and test, which
  destroys text–EEG pairing without crossing split boundaries.
- Random-noise EEG checks whether gains come from a second learned branch or
  regularization rather than aligned neural information.
- The default full evaluation uses five folds and seeds 42, 52, and 62.
- Saved metrics include accuracy, macro-F1, weighted F1, per-class F1,
  confusion matrices, fold histories, and out-of-fold predictions.
- Fusion results are compared with the matching text baseline using paired
  sentence-level bootstrap intervals.

### Repository and Colab workflow

The experiment logic was placed in `.py` files, with a minimal Colab notebook
as the runner. The notebook mounts Drive, updates the repository, defines paths,
extracts or resumes features, validates alignment, runs smoke tests, launches
the suite, and displays saved reports.

Each setup and seed is saved independently under a versioned `RUN_TAG`, allowing
an interrupted Colab run to resume. A manifest prevents a tag from silently
mixing incompatible configurations. The project is Colab-first; heavyweight
libraries, model downloads, and training are not run on the Mac.

## 2026-07-14 — Correct the labels path

The notebook was changed to use the labels CSV that already existed under the
main Thesis data folder. No labels file is stored in the repository, and no
second Drive copy is required.

## 2026-07-15 — Validate the extracted ZuCo cache

The first complete data inspection reported:

```text
400 sentences
12 subjects
2,520 cached features
105 channels
24 feature families
4,537 usable subject-sentence rows
8–12 subjects per sentence (mean 11.3425)
123 negative / 137 neutral / 140 positive sentences
```

The available subjects were ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW,
ZMG, and ZPH. Every sentence had at least one usable EEG recording, so the cache
and labels were aligned well enough to train.

## 2026-07-15 — Run the first EEG-only smoke test

The first smoke run used `eeg_only`, seed 42, two sentence-level folds, and one
epoch. It completed on CUDA and saved correctly under `v1_full_smoke`.

```text
fold 1: accuracy 0.345, macro-F1 0.187
fold 2: accuracy 0.380, macro-F1 0.312
OOF:    accuracy 0.3625, macro-F1 0.267378
```

These scores were treated only as an end-to-end check. One epoch of EEG-only
training was not interpreted as a final performance result.

Two warnings appeared:

1. NumPy reported an all-NaN slice during fold median calculation.
2. PyTorch reported that the older CUDA AMP scaler and autocast calls were
   deprecated.

## 2026-07-15 — Make reusable artifacts persistent

A simple Drive artifact layout was introduced:

```text
Thesis/CachedArtifacts/zuco_multimodal_sentiment/
  eeg_features/classical_v2_normalized_line_length/
  models/LaBSE/

Thesis/Results/zuco_multimodal_sentiment/
  <run tag>/
```

The already-extracted EEG cache is moved into `CachedArtifacts` rather than
being copied or recomputed. LaBSE is stored once in Drive and copied to Colab's
temporary disk once per runtime for faster repeated fold initialization. Python
environments are not stored in Drive.

## 2026-07-15 — Handle entirely unavailable fold features

The preprocessing warning was investigated rather than only suppressed. For
each fold, features with no finite training value are now recorded. They are
forced to zero in training, validation, and test because that fold has no basis
for estimating their distribution. This also prevents a value that appears only
outside training from passing through without meaningful scaling.

The number of such features is saved in every fold result as
`n_all_missing_train_features`. The CUDA AMP calls were updated to the current
PyTorch API with compatibility fallbacks for older versions.

A second smoke run, `v1_full_smoke_cached`, completed with the same metrics as
the first. Both folds reported four all-missing training features, and the
deprecation warnings were gone.

## 2026-07-15 — Report globally unavailable feature names

Data inspection was extended to distinguish globally unavailable features from
features absent only within one training fold. The four global columns were:

```text
raw_skew_ch104
raw_kurtosis_ch104
raw_hjorth_mobility_ch104
raw_hjorth_complexity_ch104
```

This showed that the issue was confined to four variance-dependent statistics
on the final channel rather than four missing sentences, subjects, recordings,
or electrodes.

## 2026-07-17 — Cache LaBSE in Drive

The public `sentence-transformers/LaBSE` files were downloaded in Colab. The
network transfer was approximately 1.81 GB and reconstructed to approximately
1.90 GB in Drive. Six required model/tokenizer files completed successfully.
The unauthenticated Hugging Face warning was harmless because the model is
public.

The persistent copy is under
`CachedArtifacts/zuco_multimodal_sentiment/models/LaBSE`, and the runtime copy
was prepared at `/content/cached_artifacts/LaBSE`.

## 2026-07-17 — Identify and exclude the Cz reference channel

The canonical 105-channel ZuCo montage maps `ch104` to Cz. Earlier feature code
also identified the final channel as the flat reference channel. A constant
reference signal has zero variance, which explains the exact missing pattern:

- skewness and kurtosis are undefined for a constant signal;
- Hjorth mobility contains signal variance in its denominator;
- Hjorth complexity depends on the undefined mobility.

This was therefore expected reference-channel behavior, not corrupted or
missing EEG.

The loader now excludes all 24 Cz features before constructing the EEG tensor.
The original cache remains unchanged and does not need to be rebuilt. The model
will now receive:

```text
104 electrode tokens x 24 feature families = 2,496 features
```

Because removal happens before model construction, the encoder will create 104
learned electrode embeddings; Cz cannot contribute a meaningless embedding or
attention token. Run manifests now compare the full data summary as well as
training settings and paths, preventing 105-channel and 104-channel results
from being combined under one tag.

### Status: implemented but not yet run

The Cz exclusion is the latest architecture/input change. It has been committed,
but no Colab smoke result has yet been produced with the 104-channel input. The
next validation run is `v1_full_smoke_no_cz`.

Expected section 5 values before that smoke test are:

```text
n_features: 2496
n_channels: 104
dropped reference: Cz / ch104 / 24 features
n_globally_all_missing_features: 0
```

After this smoke test succeeds, the next step is the full versioned experiment
suite under `v1_full`.
