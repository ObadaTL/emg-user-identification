# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Biometric user identification from EMG (electromyography) signals, using the UCI EMG dataset (36 subjects, 8 channels, 7 hand-gesture classes, 200Hz). The pipeline loads raw per-subject CSVs, segments the continuous signal into windows, extracts Root Sum Square (RSS) features per channel, optionally reduces dimensionality with a custom Kernel Fisher Discriminant (KFD), and trains an MLP classifier to predict a "biometric ID" from the feature vector.

Two competing approaches to defining what a "biometric identity" is are implemented side by side, selected by which pipeline/loader class is instantiated:

- **Per-subject** (`SubjectPipeline` + `SubjectLoader`, default): both arms of a person map to the same biometric ID. This is the more meaningful setup for a biometric-identification use case.
- **Per-arm** (`UCIPipeline` + `UCIDataLoader`, via `--per-arm`): each arm/session is a distinct biometric ID (user_id, arm) → id.

README.md quotes current, honest benchmark numbers (~68-75% mean test accuracy depending on user count) reproduced with `experiment_results/` + `analyze_experiments.py`. Older figures (e.g. ~94%/~87% in commit history) predate a fix and are wrong — an earlier version of this pipeline applied data augmentation before the train/test split and fit KFD once across the whole training set before cross-validating it, both of which leak near-duplicate/label information into the "held-out" data and inflate accuracy. Both are now fixed (augmentation and KFD are applied strictly inside `model.train()`, after splitting) — see "Architecture" below. If you change anything in the train/test split, augmentation, or KFD/CV ordering, re-run the benchmark commands in "Commands" above and regenerate `experiment_results/` before trusting any number.

`SubjectLoader` subclasses `UCIDataLoader` and only overrides `_create_biometric_id` to collapse arm/session into the parent subject ID. Everything else (loading, preprocessing, segmentation, feature extraction) is shared.

## Commands

There is no test suite, linter, or build step in this repo — it's a research/experimentation codebase. Environment setup and pipeline execution:

```bash
pip install -r requirements.txt

# Run the full pipeline (default: per-subject approach, all 36 users)
python run_pipeline.py

# Common flags (see run_pipeline.py or README.md for the full list)
python run_pipeline.py --experiment my_experiment --users 10
python run_pipeline.py --per-arm                          # per-arm identity instead of per-subject
python run_pipeline.py --no-kfd                            # disable KFD dimensionality reduction
python run_pipeline.py --kfd-kernel rbf --kfd-components 15
python run_pipeline.py --no-augment                        # disable data augmentation
python run_pipeline.py --selected-gestures 1 2 3            # restrict to specific gesture classes
python run_pipeline.py --users 8 --run-count 5 --random-selection   # multi-run statistical experiment
python run_pipeline.py --no-cv                              # skip 5-fold CV for faster iteration

# Real-time / streaming inference demos (src/inference/)
python src/inference/demo_real_time.py --users 6 --model sklearn [--per-arm]
python src/inference/benchmark_approach.py --users 6 --model sklearn   # compares per-subject vs per-arm
python src/inference/benchmark_windows.py --users 6 --model sklearn --min-window 250 --max-window 1500

# Post-hoc statistical analysis across saved experiment result CSVs.
# Reads hardcoded paths, so these four experiments must exist first:
python run_pipeline.py --users 5  --run-count 10 --random-selection --experiment 5users-10runs-rand
python run_pipeline.py --users 10 --run-count 10 --random-selection --experiment 10users-10runs-rand
python run_pipeline.py --users 15 --run-count 10 --random-selection --experiment 15users-10runs-rand
python run_pipeline.py --users 20 --run-count 10 --random-selection --experiment 20users-10runs-rand
python analyze_experiments.py
```

`--run-count > 1` writes each run to `experiments/<experiment>/run{N}/` plus a combined `experiments/<experiment>/results.csv`; a single run writes directly to `experiments/<experiment>/`.

## Architecture

The pipeline is a strict 5-stage sequence, implemented once per approach in `src/pipelines/{uci_pipeline,subject_pipeline}.py` (the two are near-duplicates — the only real difference is which loader class they instantiate):

1. **Load** — `data_loader.load_raw_data()`: reads `data/raw/UCI/<user_id>/<user_id>.csv`, assigns a `bio_id` per row via the loader's `_create_biometric_id(user_id, session_id)`.
2. **Preprocess** — `data_loader.preprocess_data()`: type coercion, missing-value fill, clipping/flat-signal checks.
3. **Segment** — `data_loader.segment_data()`: splits each subject's continuous signal into overlapping windows (`segment_samples` / `overlap_samples` from config), one row per window in the output DataFrame.
4. **Extract features** — `feature_extractor.extract_features()` (`src/features/uci_extractor.py`): per-channel RSS on each windowed segment only. No augmentation and no KFD happen here (see below). Output is a single `np.ndarray` of shape `(n_samples, n_channels + 1)` where **the last column is always the target `bio_id`** — every downstream consumer relies on this convention (`X = features[:, :-1]`, `y = features[:, -1]`).
5. **Train/evaluate** — `EMGUserIdentifier` (sklearn, `src/models/sklearn_mlp.py`) or `TFEMGUserIdentifier` (TensorFlow, `src/models/tensorflow_mlp.py`), selected via `config['model']['model_type']`. Both `train()` methods take the raw `features` array plus optional `class_labels`, and internally: split into train/test → (sklearn only) 5-fold CV on the raw, non-augmented training split → augment the training split only (`src/features/augmentation.py::augment_training_set`) → fit KFD + scaler + classifier on the augmented training split → evaluate once on the untouched test split. This ordering is deliberate — augmenting or fitting KFD before the split leaks near-duplicate/label information into "held-out" data. See README.md's "Data Integrity & Methodology" section.

**KFD is applied inside model training, not in the feature extractor.** `KernelFisherDiscriminant` (`src/features/kfd.py`) is a custom sklearn-style transformer (`fit`/`transform`) implementing kernelized Fisher discriminant analysis (poly/rbf/linear kernels). For the sklearn backend it's a step inside `self.model` (an `sklearn.pipeline.Pipeline` of `kfd → scaler → classifier`), so `cross_val_score` refits it from scratch per CV fold instead of once on the whole training set — that per-fold refit is what makes the CV score leak-free. Saving/loading the sklearn model (`save_model`/`load_model`) just joblib-dumps that one Pipeline object; there's no separate KFD artifact to track. The TensorFlow backend can't use a Pipeline (Keras isn't an sklearn step), so `TFEMGUserIdentifier` fits KFD manually on the training split and persists it as a sibling `.kfd` file next to the `.keras` model and `.scaler`/`.idmap` files (`save_model`/`load_model` in `tensorflow_mlp.py`).

Both loaders and both feature-extraction/model classes inherit from ABCs in `base_loader.py` / `base_extractor.py` that define the load/preprocess/segment and normalize/RSS contracts respectively — adding a new dataset or feature set means subclassing these rather than modifying the UCI-specific implementations.

`src/inference/` (`RealTimeEMGIdentifier` and friends) is a **proof-of-concept**, not a maintained feature — see `src/inference/README.md`. It reimplements the windowing + feature extraction + prediction path for a simulated streaming context, but was never updated to apply KFD to the features it extracts, so it errors out against a model trained with the default `use_kfd: true`. The main pipeline (`run_pipeline.py`) does not depend on anything under `src/inference/`.

## Configuration

All pipeline behavior is driven by `config/uci_config.yaml` (paths, data shape, feature-extraction/augmentation/KFD parameters — all still grouped under `feature_extraction:` even though augmentation/KFD now execute inside the model classes, not the feature extractor — model hyperparameters for both sklearn and TensorFlow backends, training split/CV settings). `run_pipeline.py` CLI flags mutate `pipeline.config` (a dict shared by reference with whatever model gets constructed later inside `pipeline.run()`) in-memory before running — they do not touch the YAML file. Some flags (e.g. `--no-kfd`, `--kfd-kernel`) also set a same-named attribute directly on `pipeline.feature_extractor`, which still holds `use_kfd`/`kfd_kernel`/`kfd_components` for informational/POC purposes even though it doesn't act on them itself. When adding a new config-driven option, wire it through both the YAML default and a corresponding `argparse` flag in `run_pipeline.py`, mutating `pipeline.config['feature_extraction'][...]` following the existing pattern.

## Data layout

- `data/raw/UCI/<user_id>/<user_id>.csv` — raw per-subject input (gitignored).
- `data/interim/`, `data/processed/` — intermediate/processed data (gitignored).
- `experiments/<name>/` — per-run outputs: trained model (`model.joblib`), confusion matrix, gesture-performance plots, `results_summary.txt`, `results.csv`. Entirely gitignored — nothing under `experiments/` is tracked in git, regardless of naming.
- `experiment_results/` — aggregate cross-experiment plots/statistics (boxplots, gesture heatmap/radar, Tukey HSD test) produced by `analyze_experiments.py`, reading from four hardcoded paths (`experiments/{5,10,15,20}users-10runs-rand/results.csv`) that must exist locally before running it. This directory *is* committed to git — regenerate it (`python analyze_experiments.py`) after any change that affects reported accuracy, so committed plots never go stale relative to the code.
