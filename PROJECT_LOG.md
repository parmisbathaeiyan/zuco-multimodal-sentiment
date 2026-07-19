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

### Planned first experiment table

The first experiment table was defined around six conceptual comparisons. They
map to the implemented setup names as follows:

| Planned comparison | Implemented setup |
| --- | --- |
| LaBSE-CLS text only | `text_frozen` and `text_finetune` |
| EEG only | `eeg_only` |
| Text + EEG concatenation | `concat_finetune` |
| Text + EEG gated fusion | `gated_finetune` |
| Text + shuffled EEG | `gated_shuffled_finetune` |
| Text + random-noise modality | `gated_noise_finetune` |

The table has seven result rows because the text-only comparison is evaluated
in both frozen and fine-tuned forms. The shuffled and noise rows use the same
gated architecture as `gated_finetune`, changing only the EEG input. This keeps
them as direct controls for whether any gated-fusion gain depends on correctly
aligned EEG rather than extra parameters or regularization.

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

## 2026-07-18 — Validate the Cz exclusion in Colab

Section 5 was rerun after updating the loader. It reported:

```text
400 sentences
12 subjects
2,496 model features
104 model channels
24 feature families
4,537 usable subject-sentence rows
0 globally all-missing features
dropped reference: Cz / ch104 / 24 cached features
```

The subject coverage, class counts, and number of usable subject-sentence rows
were unchanged from the 105-channel inspection. This confirms that the loader
removed only the 24 Cz reference features; it did not remove any sentences,
subjects, or usable recordings. The 104-channel smoke run
`v1_full_smoke_no_cz` was started after this validation; its results were still
pending at the time of this entry.

## 2026-07-18 — Complete the 104-channel EEG smoke test

The `v1_full_smoke_no_cz` run completed on CUDA with no all-missing training
features:

```text
fold 1: accuracy 0.350, macro-F1 0.173, all-missing train features 0
fold 2: accuracy 0.310, macro-F1 0.158, all-missing train features 0
OOF:    accuracy 0.330, macro-F1 0.265222
```

The result was saved under
`Results/zuco_multimodal_sentiment/v1_full_smoke_no_cz/eeg_only/seed_42.json`.
The lower one-epoch accuracy relative to the earlier smoke runs was not treated
as a performance comparison. Macro-F1 was nearly unchanged, and all three runs
used only one training epoch. The purpose of this run was to confirm that the
104-channel tensor, preprocessing, model forward/backward pass, evaluation, and
result saving all work without unavailable features.

## 2026-07-18 — Add a text + gated-fusion smoke test

The completed smoke tests had exercised only the EEG branch. A second short
test was added before the full suite to cover the remaining high-risk path:

```text
setups: text_finetune, gated_finetune
seed: 42
folds: 2
epochs: 1
run tag: v1_full_smoke_multimodal
```

This test checks loading LaBSE from the persistent artifact, fine-tuning the
text encoder, gated text–EEG fusion, CUDA memory use, and paired text/fusion
report generation. It must pass before starting the full experiment suite.

## 2026-07-18 — Complete the text + gated-fusion smoke test

Section 7 successfully reused the persistent LaBSE copy from Drive and copied it
to `/content/cached_artifacts/LaBSE`. No model network download was needed for
the new runtime.

The `v1_full_smoke_multimodal` run then completed on CUDA without an
out-of-memory error, shape error, unavailable training feature, or result-saving
error.

```text
text_finetune
  fold 1: accuracy 0.440, macro-F1 0.350
  fold 2: accuracy 0.445, macro-F1 0.394
  OOF:    accuracy 0.4425, macro-F1 0.373680

gated_finetune
  fold 1: accuracy 0.510, macro-F1 0.470
  fold 2: accuracy 0.395, macro-F1 0.333
  OOF:    accuracy 0.4525, macro-F1 0.423656
```

For this smoke run, gated fusion had a paired macro-F1 difference of `+0.049975`
relative to `text_finetune`. The 200-sample bootstrap interval was
`[0.003591, 0.115736]`. The mean gate was `0.119214`, still close to its initial
value because training lasted only one epoch.

These numbers were recorded as a successful integration check, not as evidence
that EEG improves classification. The run used one seed, two folds, one epoch,
and a small bootstrap. The full multi-seed experiment is required before
interpreting modality effects.

After this smoke test passed, section 9 was started with the full comparison
suite under `v1_full`.

## 2026-07-18 — Keep model execution on Colab

Local training on the 24 GB Apple-silicon MacBook Air was considered and then
left out of scope. Supporting its GPU would require an MPS execution path,
additional local dependencies and model storage, and separate backend
validation. Mixing MPS and CUDA runs could also make the main experiment less
consistent.

All official smoke tests and full experiments will therefore continue on Colab
CUDA. The Mac will remain limited to lightweight source editing and checks that
require no project dependency installation, model download, dataset download,
or training.

## 2026-07-18 — Full run interrupted and audited from Drive

The `v1_full` Colab runtime disconnected before the suite completed. Drive was
inspected to identify the last durable checkpoint. Seventeen of the planned 21
setup/seed JSON files had been saved:

```text
text_frozen:                 seeds 42, 52, 62 complete
text_finetune:               seeds 42, 52, 62 complete
eeg_only:                    seeds 42, 52, 62 complete
concat_finetune:             seeds 42, 52, 62 complete
gated_finetune:              seeds 42, 52, 62 complete
gated_shuffled_finetune:     seeds 42, 52 complete
```

The last saved file was `gated_shuffled_finetune/seed_52.json`. It was written
at `2026-07-18 11:41:06 UTC`; no result file appeared during the following
three-plus hours. The runtime therefore most likely disconnected while running
`gated_shuffled_finetune` seed 62.

The remaining jobs are:

```text
gated_shuffled_finetune: seed 62
gated_noise_finetune:    seeds 42, 52, 62
```

No final `tables` or `plots` folder exists yet because report generation runs
only after all setup/seed jobs finish. Resuming section 9 with the same
`RUN_TAG=v1_full` will skip the 17 completed files. Any unfinished folds from
the interrupted seed were not saved separately, so seed 62 of the shuffled
setup will restart from its first fold.

## 2026-07-19 — Full run interrupted a second time

The resumed run skipped the 17 existing setup/seed files and successfully
completed:

```text
gated_shuffled_finetune: seed 62
gated_noise_finetune:    seeds 42, 52
```

Their saved out-of-fold results were:

```text
gated_shuffled_finetune seed 62: accuracy 0.698, macro-F1 0.692
gated_noise_finetune seed 42:    accuracy 0.715, macro-F1 0.710
gated_noise_finetune seed 52:    accuracy 0.705, macro-F1 0.700
```

Drive now contains 20 of the planned 21 result JSON files. The only remaining
job is:

```text
gated_noise_finetune: seed 62
```

The console reached the model-weight load for that seed but printed no completed
fold. Drive also contains no `seed_62.json` in the noise folder. The last durable
result, `gated_noise_finetune/seed_52.json`, was saved at
`2026-07-18 22:50:04 UTC`, more than eight hours before this audit. The runtime
therefore disconnected while starting the final seed.

There are still no final tables or plots. Resuming section 9 with
`RUN_TAG=v1_full` will skip the 20 completed jobs, rerun all five folds of the
remaining noise seed, and then build the final reports.

## 2026-07-19 — Complete the full experiment suite and generate reports

Section 9 was resumed again with the same `RUN_TAG=v1_full`. The runner skipped
the 20 durable setup/seed results, completed `gated_noise_finetune` seed 62, and
then generated the final reports. Drive now contains all 21 planned result JSON
files: seven setups with seeds 42, 52, and 62.

The reporting stage completed successfully and created:

```text
tables/summary.csv
tables/summary.json
tables/summary.md
plots/scores.png
plots/confusions.png
```

The final three-seed summary is:

| setup | accuracy mean | accuracy SD | macro-F1 mean | macro-F1 SD | mean gate |
|---|---:|---:|---:|---:|---:|
| `text_frozen` | 0.6033 | 0.0077 | 0.5971 | 0.0093 | — |
| `text_finetune` | 0.6792 | 0.0246 | 0.6766 | 0.0237 | — |
| `eeg_only` | 0.3392 | 0.0319 | 0.2809 | 0.0156 | — |
| `concat_finetune` | 0.6850 | 0.0256 | 0.6801 | 0.0260 | — |
| `gated_finetune` | 0.7067 | 0.0083 | 0.7019 | 0.0088 | 0.1193 |
| `gated_shuffled_finetune` | 0.7058 | 0.0072 | 0.7006 | 0.0072 | 0.1193 |
| `gated_noise_finetune` | 0.7067 | 0.0062 | 0.7015 | 0.0062 | 0.1193 |

Relative to fine-tuned text only, the paired macro-F1 results were:

```text
concat_finetune:          +0.0035, 95% CI [-0.0368, 0.0439]
gated_finetune:           +0.0253, 95% CI [-0.0124, 0.0630]
gated_shuffled_finetune:  +0.0240, 95% CI [-0.0140, 0.0619]
gated_noise_finetune:     +0.0249, 95% CI [-0.0131, 0.0628]
```

The aligned gated model is numerically the best setup, but its result is almost
identical to the shuffled-EEG and random-noise controls. Every paired confidence
interval includes zero. This experiment therefore does not provide evidence
that correctly aligned EEG information improved sentiment classification. The
small gated-model gain is more consistent with an architectural or
regularization effect from the added branch. Simple concatenation also did not
meaningfully improve over the fine-tuned text baseline. EEG alone remained
weak, and fine-tuning LaBSE clearly outperformed freezing it.

The learned gate stayed near its approximately `0.119` initialization in all
three gated conditions. This and the negative-control results should be
considered when choosing the next architecture or diagnostic experiment.

## 2026-07-19 — Analyze the completed results in detail

The 21 seed files were inspected directly rather than relying only on the
aggregate table. The main conclusions are:

1. Fine-tuning LaBSE produced the clearest real improvement in this run.
   `text_finetune` improved mean macro-F1 from `0.5971` to `0.6766` relative to
   `text_frozen`, a gain of about `0.0795`. The fine-tuned result was higher for
   all three seeds.
2. Simple concatenation did not reliably improve the fine-tuned text baseline.
   Its per-seed macro-F1 differences were `+0.0336`, `+0.0024`, and `-0.0254`;
   the mean paired difference was only `+0.0035`.
3. Aligned gated fusion had per-seed differences of `+0.0343`, `+0.0538`, and
   `-0.0122` against fine-tuned text. It improved two seeds but hurt the third,
   and its paired interval included zero.
4. The apparent gated gain was not specific to aligned EEG. Mean macro-F1 was
   `0.7019` with aligned EEG, `0.7006` with shuffled EEG, and `0.7015` with
   random noise.

The saved sentence predictions made the negative-control result especially
clear:

```text
aligned vs shuffled EEG prediction agreement
  seed 42: 98.25%
  seed 52: 100.00%
  seed 62: 99.75%

aligned EEG vs random noise prediction agreement
  seed 42: 98.25%
  seed 52: 100.00%
  seed 62: 99.50%
```

For seed 52, all 400 class predictions were identical across aligned, shuffled,
and noise gated models. For seed 62, aligned and shuffled differed on only one
prediction without changing accuracy, while aligned and noise differed on two
predictions and noise was correct on one additional sentence. For seed 42,
aligned EEG corrected four predictions that shuffled/noise missed but lost
three that they got right: a net difference of one sentence.

The gated model therefore showed almost no sensitivity to EEG–sentence
alignment at the final decision level. The additional branch may alter
optimization, regularization, or random initialization relative to the
text-only model, but the current results do not support attributing its score
to informative aligned EEG. Identical predicted classes do not prove that all
logits were identical, because logits were not saved, but any modality effect
was too small to materially change the decisions.

The gate is a 128-dimensional sigmoid vector initialized from logits of `-2`,
which gives an initial mean of approximately `0.11920`. Its final mean across
the gated runs was approximately `0.11925`–`0.11926`, with only tiny
fold-to-fold changes. A stable mean alone cannot prove that every gate
coordinate or the EEG projection was unused, but it agrees with the
negative-control evidence that the model did not learn a meaningful
alignment-dependent contribution.

Class-level behavior was:

| setup | negative F1 | neutral F1 | positive F1 |
|---|---:|---:|---:|
| `text_frozen` | 0.5524 | 0.5196 | 0.7193 |
| `text_finetune` | 0.6392 | 0.5842 | 0.8064 |
| `eeg_only` | 0.0541 | 0.3646 | 0.4239 |
| `concat_finetune` | 0.6532 | 0.5898 | 0.7974 |
| `gated_finetune` | 0.6841 | 0.6078 | 0.8139 |
| `gated_shuffled_finetune` | 0.6822 | 0.6042 | 0.8156 |
| `gated_noise_finetune` | 0.6822 | 0.6060 | 0.8165 |

Positive sentiment was consistently easiest and neutral sentiment remained the
hardest for the text and multimodal models. The small class-level changes from
gated fusion were reproduced by shuffled EEG and noise. The EEG-only model
nearly stopped predicting the negative class: across three seeds its aggregate
negative recall was only `3.25%`. With balanced class counts, its mean accuracy
of `0.3392` was approximately chance-level and slightly below the `0.35`
majority-class baseline.

The gated variants had lower seed-to-seed macro-F1 standard deviation
(`0.0061`–`0.0088`) than fine-tuned text (`0.0237`), but three seeds are too few
to treat this as strong evidence of greater robustness. Fold macro-F1 also
varied substantially because each fold contained only about 80 test sentences.

The current confidence intervals are paired sentence-level bootstrap intervals
against the matching text-only result. The report averages the three
seed-specific deltas and interval endpoints. They are useful diagnostics, but
they are not a formal estimate of uncertainty across model seeds, subjects, or
new datasets. Any thesis claim should therefore remain limited to this
sentence-level ZuCo experiment.

The appropriate conclusion for this first full experiment is a negative but
useful one: the implemented handcrafted EEG modality did not add demonstrated
alignment-specific sentiment information beyond fine-tuned LaBSE. This does
not establish that ZuCo EEG is intrinsically uninformative. The result could
also reflect the handcrafted representation, equal averaging across readers,
text dominance during joint fine-tuning, or the current fusion mechanism.

Before trying a substantially larger architecture, the next diagnostic run
should preserve identical initialization across controls, add a zero-EEG gated
control, save modality and logit norms, and compare frozen-text gated fusion
against frozen text only. These checks can distinguish a genuine EEG effect
from initialization and optimization effects and test whether fine-tuned text
is simply overpowering the EEG branch.
