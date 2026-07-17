# Project log

## 2026-07-17 — Exclude the Cz reference channel

Data inspection found four globally unavailable columns, all on `ch104`:
`raw_skew`, `raw_kurtosis`, `raw_hjorth_mobility`, and
`raw_hjorth_complexity`. The canonical ZuCo montage maps `ch104` to Cz. Cz is
the flat reference channel, so these variance-dependent statistics are
mathematically undefined rather than missing because of corrupted recordings.

All 24 features belonging to Cz are now excluded when the cached subject files
are loaded. The cache remains unchanged and does not need to be rebuilt. The EEG
encoder receives 104 channels x 24 feature families = 2,496 features. Run
manifests also compare the data summary so results made with different channel
sets cannot be mixed under one run tag.

## 2026-07-15 — Persist reusable Colab artifacts

Reusable EEG feature files and the LaBSE model are kept in the project's Drive
`CachedArtifacts` folder. Experiment outputs remain separately versioned under
`Thesis/Results/zuco_multimodal_sentiment`.

## 2026-07-14 — Initial multimodal pipeline

Implemented sentence-level evaluation for frozen and fine-tuned LaBSE `[CLS]`
text representations, an electrode-set EEG encoder, concatenation and gated
fusion, and shuffled/noise EEG controls. Splits are sentence-level and EEG
preprocessing is fitted inside each training fold.
