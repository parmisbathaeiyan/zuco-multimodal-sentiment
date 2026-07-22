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

## 2026-07-19 — Record interpretation details for future architecture work

Two observations from the first result analysis are important for later
feature-by-architecture experiments:

1. The poor EEG-only result does not prove that ZuCo EEG contains no sentiment
   information. Equal averaging across readers may wash out subject-specific
   patterns; 2,496 handcrafted summary features may be noisy or redundant; 400
   sentences provide a small training set; the present channel-attention plus
   subject-mean encoder may be unsuitable; and useful sentiment information may
   require temporal or word-level EEG rather than sentence-level statistics.
2. Positive sentiment was consistently the easiest class. Neutral sentiment was
   the hardest for the text and multimodal systems and was often confused with
   negative sentiment. Future error analysis and architecture comparisons
   should retain per-class results instead of relying only on overall accuracy.

The future architecture matrix should explicitly include both:

- pooled-across-subjects models, beginning with the present equal mean and then
  learned subject attention;
- subject-specific models, trained and evaluated per reader before aggregating
  conclusions across readers.

The same aligned, shuffled, noise, and zero controls should be applied to both
forms. Subject-specific and learned-pooling implementations remain a later
stage, after the controlled diagnostic experiment determines whether the
current gated branch is being used at all.

## 2026-07-19 — Audit the epoch explanation

Every setup in `v1_full` had the same maximum of 12 epochs and patience of four.
The gated setup was not assigned a larger explicit epoch budget. Early stopping
did, however, allow it to train longer on average:

```text
text_finetune
  mean selected best epoch: 5.93
  mean epochs actually run: 9.33

gated_finetune
  mean selected best epoch: 7.27
  mean epochs actually run: 10.93
```

The fine-tuned text encoder in the gated model therefore received more optimizer
updates on average. This could contribute to the higher gated score. It is not
an unfair command-line setting—the validation procedure selected checkpoints
independently—but it is an optimization-path difference that prevents the
original score gap from being interpreted as an EEG effect. The forced-zero
gated control is intended to isolate this: it keeps the gated architecture,
initialization, dropout computations, optimizer schedule, and early-stopping
behavior while making the EEG contribution exactly zero.

## 2026-07-19 — Implement the controlled diagnostic experiment

The next experiment is versioned as `v2_controlled_diagnostics`. The
implementation adds:

- `gated_zero_finetune` and `gated_zero_frozen`;
- explicit resetting of the initialization seed immediately before model
  construction;
- a SHA-256 fingerprint of all randomly initialized task modules, excluding the
  shared pretrained LaBSE checkpoint;
- report-time verification that aligned, shuffled, noise, and zero controls
  have identical task fingerprints for each text mode, seed, and fold;
- complete final sigmoid gate values for every fold rather than only their mean;
- held-out text embedding norms;
- held-out raw EEG embedding norms;
- candidate and effective gated EEG-contribution norms;
- held-out logits with and without the EEG contribution;
- the rate at which removing EEG changes the predicted class;
- direct aligned-versus-shuffled, aligned-versus-noise, and
  aligned-versus-zero tables.

The zero model still constructs and executes the same EEG encoder and gated
modules so its random-number and optimization path matches the other gated
controls. Its contribution is replaced by an exact zero tensor before fusion,
so it cannot pass EEG information to the classifier.

The Colab notebook now contains:

```text
section 12: short matched-control smoke test
section 13: fine-tuned text and four gated controls
section 14: frozen text and four gated controls
section 15: score, control-comparison, diagnostic, and fingerprint outputs
```

Sections 13 and 14 share the same resumable Drive run folder. This allows the
fine-tuned and frozen phases to be run separately without losing or repeating
completed setup/seed files.

The section 12 smoke cell displays its direct aligned-versus-zero comparison,
modality diagnostics, and initialization-verification metadata immediately
after completion so the new path can be checked before starting the full run.

## 2026-07-19 — Clarify seeds, folds, and bootstrap uncertainty

The five folds and three seeds are part of model training and evaluation:

- Within one seed, five-fold cross-validation partitions the 400 sentences.
  Each sentence is a test example exactly once, producing 400 out-of-fold
  predictions for that seed.
- Three seeds repeat the complete five-fold process with different splits,
  model initialization, data order, and stochastic training. This produces
  three macro-F1 values and three sets of 400 paired predictions.

The bootstrap is a separate analysis performed after training. For one seed,
the reporter repeatedly samples 400 sentence positions with replacement from
the paired baseline and candidate predictions. It recalculates both macro-F1
scores and their difference for every resample. The 2.5th and 97.5th
percentiles form the reported sentence-level 95% bootstrap interval.

The current summary computes this bootstrap separately for each seed and then
averages the three observed differences and the three interval endpoints. It
answers a limited question: how uncertain is the paired score difference under
resampling of these ZuCo sentences for these trained models? It is not a formal
confidence interval over arbitrary training seeds, new readers, or a new
dataset. Five folds do not provide five independent datasets, because together
they form one out-of-fold prediction set over the same 400 sentences. Three
seeds are also too few to estimate the full distribution of training
randomness. Thesis reporting should state this scope explicitly.

## 2026-07-19 — Complete and verify the controlled diagnostic smoke test

The `v2_controlled_diagnostics_smoke` run completed and was audited directly
from Drive. Both expected seed files are present:

```text
gated_finetune/seed_42.json
gated_zero_finetune/seed_42.json
```

The summary, control-comparison, diagnostic, metadata, and plot artifacts were
also generated successfully. The report verified matching task-module
initialization fingerprints for both seed/fold groups:

```text
matched_initialization_groups_verified: 2
```

The one-epoch, two-fold integration scores were:

```text
gated_finetune:      accuracy 0.4525, macro-F1 0.4237
gated_zero_finetune: accuracy 0.4550, macro-F1 0.4279
```

Aligned gated fusion was `-0.0043` macro-F1 below the forced-zero control, with
a smoke bootstrap interval of `[-0.0138, 0.0047]`. Their class predictions
agreed on `98.75%` of sentences. These scores are not substantive experiment
results because the smoke test used one seed, two folds, and one epoch.

The diagnostic path behaved as designed:

```text
gated_finetune
  candidate/effective contribution-to-text norm ratio: 0.0648
  mean logit-delta L2:                              0.0322
  predictions changed when EEG was removed:        6.0%

gated_zero_finetune
  candidate contribution-to-text norm ratio:       0.0634
  effective contribution-to-text norm ratio:       0.0000
  mean logit-delta L2:                              0.0000
  predictions changed when EEG was removed:        0.0%
```

The zero setup therefore preserved and measured the candidate EEG branch while
blocking its effective contribution exactly. This validates the new control and
saved diagnostics before the expensive run.

No `v2_controlled_diagnostics` full-run folder exists yet. Sections 13 and 14
remain to be run, followed by section 15 for the complete controlled results.

## 2026-07-22 — Audit section 13 progress after interruption

The `v2_controlled_diagnostics` Drive folder now contains 14 of the 15 planned
fine-tuned controlled setup/seed files:

```text
text_finetune:                 seeds 42, 52, 62 complete
gated_finetune:                seeds 42, 52, 62 complete
gated_shuffled_finetune:       seeds 42, 52, 62 complete
gated_noise_finetune:          seeds 42, 52, 62 complete
gated_zero_finetune:           seeds 42, 52 complete
```

Section 13 is therefore `93.3%` complete. The only missing result is:

```text
gated_zero_finetune: seed 62
```

The last durable file was `gated_zero_finetune/seed_52.json`, saved at
`2026-07-20 10:54:12 UTC` (`14:24:12` Tehran time). There are no section 13
summary tables or plots yet because report generation starts only after every
requested setup/seed job finishes. Rerunning section 13 will skip the 14 saved
files and restart the five folds of the missing zero-control seed 62.

Section 14 has not started. Across both planned full diagnostic phases, 14 of
30 setup/seed jobs are currently complete.

## 2026-07-22 — Complete the fine-tuned controlled diagnostic phase

Section 13 completed after resuming the missing `gated_zero_finetune` seed 62.
Drive now contains all 15 planned fine-tuned setup/seed JSON files, along with
summary, direct-control, diagnostic, metadata, score-plot, and confusion-plot
artifacts.

The matched-initialization check passed for all 15 seed/fold groups:

```text
matched_initialization_groups_verified: 15
```

The three-seed results were:

| setup | accuracy mean | accuracy SD | macro-F1 mean | macro-F1 SD |
|---|---:|---:|---:|---:|
| `text_finetune` | 0.6792 | 0.0246 | 0.6766 | 0.0237 |
| `gated_finetune` | 0.7067 | 0.0082 | 0.7019 | 0.0088 |
| `gated_shuffled_finetune` | 0.7058 | 0.0072 | 0.7006 | 0.0072 |
| `gated_noise_finetune` | 0.7067 | 0.0062 | 0.7015 | 0.0061 |
| `gated_zero_finetune` | 0.7075 | 0.0061 | 0.7028 | 0.0065 |

All four gated forms scored about `0.024`–`0.026` macro-F1 above the original
text-only model, and every interval against text included zero. More
importantly, the direct aligned-control comparisons were:

```text
aligned minus shuffled: +0.00128, CI [-0.00304, 0.00571], agreement 99.33%
aligned minus noise:    +0.00038, CI [-0.00583, 0.00564], agreement 99.25%
aligned minus zero:     -0.00089, CI [-0.00564, 0.00283], agreement 99.75%
```

The forced-zero model was numerically the best fine-tuned setup. Its per-seed
macro-F1 values were `0.7107`, `0.7030`, and `0.6948`. The aligned and zero
models also selected the same best epochs in every fold and ran the same number
of epochs, providing a direct check on the earlier concern that longer gated
training might explain the score gap.

Held-out branch diagnostics showed:

```text
aligned EEG contribution / text norm: 1.78%
aligned mean logit-delta L2:          0.0360
aligned predictions changed if EEG removed: 0.0833% (1 of 1,200)

shuffled predictions changed if EEG removed: 0.1667% (2 of 1,200)
noise predictions changed if EEG removed:    0.2500% (3 of 1,200)
zero predictions changed if EEG removed:     0.0000%
```

The full gate mean remained almost exactly at its `0.1192` initialization. The
aligned branch was only about 1.8% of the text-vector norm and altered one final
decision across three complete out-of-fold prediction sets. Shuffled EEG and
noise affected at least as many decisions, while forced-zero EEG achieved the
highest score. This controlled phase therefore provides strong evidence that
the fine-tuned classifier is not using alignment-specific EEG information.

The shared gain and lower seed variability of all gated forms are attributable
to the gated model's construction and stochastic optimization path rather than
to EEG content. Because the zero model still constructs and executes the EEG
branch but blocks its contribution, differences from the separately built
text-only model can reflect module-initialization order and random-number usage.

Section 14 has not been run: no `text_frozen`, `gated_frozen`,
`gated_shuffled_frozen`, `gated_noise_frozen`, or `gated_zero_frozen` folders
exist in the diagnostic run. The frozen-text phase remains necessary to test
whether EEG becomes usable when LaBSE cannot adapt and dominate the task.

## 2026-07-22 — Complete the frozen controlled phase and full diagnostic suite

Section 14 completed successfully. Drive contains all three seeds for each of
the five frozen setups, bringing `v2_controlled_diagnostics` to all 30 planned
setup/seed results. The combined summary, direct-control comparisons,
diagnostics, metadata, score plot, and confusion plot were regenerated after
the final seed.

Initialization fingerprints matched for all seed/fold groups across both text
modes:

```text
matched_initialization_groups_verified: 30
```

The frozen-text results were:

| setup | accuracy mean | accuracy SD | macro-F1 mean | macro-F1 SD |
|---|---:|---:|---:|---:|
| `text_frozen` | 0.6033 | 0.0077 | 0.5971 | 0.0093 |
| `gated_frozen` | 0.6075 | 0.0074 | 0.6018 | 0.0092 |
| `gated_shuffled_frozen` | 0.6083 | 0.0077 | 0.6029 | 0.0097 |
| `gated_noise_frozen` | 0.6075 | 0.0108 | 0.6023 | 0.0130 |
| `gated_zero_frozen` | 0.6075 | 0.0094 | 0.6020 | 0.0113 |

All gated frozen models were only about `0.0047`–`0.0058` macro-F1 above
frozen text, and every interval against frozen text included zero. Direct
aligned-control comparisons were:

```text
aligned minus shuffled: -0.00113, CI [-0.00628, 0.00265], agreement 99.42%
aligned minus noise:    -0.00053, CI [-0.01240, 0.01019], agreement 97.42%
aligned minus zero:     -0.00020, CI [-0.00708, 0.00578], agreement 99.17%
```

Shuffled EEG was numerically the best frozen multimodal setup. Freezing LaBSE
therefore did not expose a useful alignment-specific EEG contribution.

The aligned frozen EEG contribution was larger relative to text than in the
fine-tuned condition because the frozen text embeddings had a smaller norm:

```text
aligned frozen EEG contribution / text norm: 3.65%
aligned frozen mean logit-delta L2:          0.0289
aligned frozen predictions changed without EEG: 1.42% (17 of 1,200)

shuffled frozen predictions changed without EEG: 1.33% (16 of 1,200)
noise frozen predictions changed without EEG:    2.33% (28 of 1,200)
zero frozen predictions changed without EEG:     0.00%
```

Although the frozen EEG branch changed more decisions than the fine-tuned
branch, shuffled EEG and noise changed a similar or larger number and matched or
outperformed aligned EEG. The gate again stayed near its `0.1192`
initialization. Across both text modes, the controlled evidence therefore does
not support an alignment-specific benefit from the current handcrafted EEG
features and pooling/fusion architecture.

Section 15 initially raised `NameError: Markdown is not defined`. This occurred
only while displaying already generated files and did not affect training,
saving, or reporting. The notebook was updated so section 15 imports `Image`,
`Markdown`, and `display` itself and defines the diagnostic result path
directly, making the display section safe after a runtime restart.

## 2026-07-22 — Roadmap and stopping rules after the controlled diagnostics

The controlled results are sufficient to stop tuning the present pipeline:
sentence-level handcrafted EEG features, equal averaging across readers, and
the current concatenation or gated-residual fusion architectures. Aligned EEG
did not outperform shuffled, random-noise, or forced-zero controls under either
fine-tuned or frozen LaBSE. More epochs, gate initializations, wider MLPs, or
additional seeds are therefore not justified as attempts to make this exact
pipeline succeed. The result is retained as a useful negative finding rather
than treated as a failed run.

The next work is divided into bounded phases.

### Phase 0 — Close and analyze the present experiment

Use the saved out-of-fold predictions and diagnostics without retraining to:

1. analyze errors by class, text confidence, sentence length, reader coverage,
   and especially the low-confidence text subset;
2. determine whether aligned EEG helps any reproducible text-hard subset more
   than shuffled, noise, and zero controls;
3. report prediction flips and EEG logit contributions for those subsets; and
4. improve uncertainty reporting where useful, distinguishing the present
   sentence bootstrap diagnostics from variation across seeds and subjects.

If no alignment-specific subset effect appears, the current fusion pipeline is
closed permanently.

### Phase 1 — One final viability test for the classical EEG features

Before attaching EEG to text again, test whether the classical features can
carry sentiment information on their own. Compare:

- subject-specific models, evaluated separately for each reader;
- pooled models using equal subject means;
- a small learned subject-pooling model, such as attention or DeepSets; and
- simple regularized baselines before larger neural models, including
  multinomial logistic regression and a linear SVM.

Retain aligned, shuffled-alignment, label-permutation, and chance or majority
controls. Report both within-subject and cross-subject or leave-one-subject-out
evaluation where the available sentence coverage permits it. A positive versus
negative binary analysis may be included as a clearly labelled sensitivity
analysis because neutral was the hardest class, but it must not replace the
three-class result.

**Stop gate A:** continue using these 2,496 sentence-level classical features
only if aligned EEG beats its controls by a predeclared practically meaningful
margin (provisionally at least `0.015` macro-F1), the direction is stable across
seeds or a meaningful share of subjects, and a direct paired uncertainty
interval supports a positive effect. If both subject-specific and learned
pooled analyses fail this gate, stop using these features for sentiment fusion.

### Phase 2 — One richer EEG-representation attempt

This phase is a separate representation study, not another modification of the
current MLP. Move from sentence summary statistics to the word/fixation-aligned
or temporal EEG available in ZuCo. Limit the search to at most two compact EEG
encoders, for example a small temporal CNN and a compact Transformer, and at
most two fusion approaches, such as late logit fusion and word-aligned
cross-attention. A self-supervised masked-EEG or contrastive EEG-text objective
may be used on training-fold data to reduce dependence on the 400 sentiment
labels. All representation learning must remain inside the training folds.

**Stop gate B:** an aligned representation must outperform shuffled, noise,
and zero controls by the predeclared margin with a positive direct paired
interval. If this fails for the bounded representation families, stop pursuing
the claim that ZuCo EEG improves sentiment classification. Do not begin a third
rescue cycle without new independent evidence.

### Phase 3 — Multimodal confirmation or a deliberate pivot

If either EEG viability gate passes, return to multimodal classification and
compare the successful EEG representation against text-only using identical
initialization and training controls. Preserve an untouched final evaluation or
use nested cross-validation to avoid selecting architectures on the same 400
sentences.

If the gates fail, pivot to one of the following research questions rather than
continuing to tune sentiment fusion:

- combine text with ZuCo eye-tracking features, which are directly aligned to
  fixations and words;
- study reading-task, difficulty, predictability, surprisal, or comprehension
  signals instead of sentiment;
- study subject variability, subject adaptation, and within-subject versus
  cross-subject transfer;
- analyze representational alignment between EEG and LaBSE with RSA, CKA, or
  canonical-correlation methods rather than requiring a classification gain;
- use EEG as privileged training information through an auxiliary or
  contrastive objective, with text-only inference, but only after demonstrating
  an alignment-specific EEG signal; or
- apply the same controlled protocol to TECO or another dataset with more
  trials or better temporal and label support.

An acceptable final thesis direction is also the controlled negative result
itself: apparent multimodal gains can be reproduced by shuffled, noise, and
zero modalities, showing why modality-specific controls are necessary. This is
more defensible than selecting a superficially better multimodal score after a
large uncontrolled architecture search.
