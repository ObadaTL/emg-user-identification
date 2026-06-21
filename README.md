# EMG-Based User Identification

This project implements a biometric identification system using Electromyography (EMG) signals from the UCI EMG dataset. The system identifies users based on their unique muscle activation patterns during various hand gestures.

## Overview

The system processes EMG data from multiple subjects performing different hand movements (gestures 1-7), extracts features, and trains a machine learning model to identify users based on these features.

### Key Features

- **Subject-based approach**: Each person is treated as a single biometric identity, regardless of which arm performed the gesture
- Feature extraction with Root Sum Square (RSS) computation for each EMG channel
- Kernel Fisher Discriminant (KFD) for dimensionality reduction and improved class separation
- Data augmentation for enhanced training and class balancing
- Support for both scikit-learn and TensorFlow neural network models

## Dataset

The system uses the UCI EMG dataset, which contains EMG recordings from 36 subjects performing different hand movements:
- Each subject performed gestures with both left and right arms
- 8 EMG channels per recording
- 7 gesture classes (plus class 0 for unmarked data)
- 200Hz sampling rate

## Biometric Identification Approach

Based on extensive experimentation, we found that treating each subject as a single biometric identity (the "per-subject" approach) yields significantly better accuracy than treating each arm as a separate identity (the "per-arm" approach):

- **Per-subject approach**: ~94% accuracy (default)
- **Per-arm approach**: ~87% accuracy

The per-subject approach combines data from both arms of the same user, allowing the model to learn more general patterns specific to each individual, resulting in more robust identification.

## Feature Processing

### Kernel Fisher Discriminant (KFD)

KFD is applied during model training to avoid data leakage:

1. Features are first extracted without KFD transformation
2. Data is split into training and test sets
3. KFD is fitted only on the training data
4. The fitted transformation is applied to both training and test sets
5. The model is trained on the transformed training data
6. For new predictions, the same KFD transformer is applied before prediction

This approach ensures that test data doesn't influence the KFD transformation, resulting in a more realistic evaluation of model performance.

## Usage

### Running the Pipeline

The main script to run the EMG user identification pipeline is `run_pipeline.py`. By default, it uses the per-subject approach.

```bash
# Basic usage with default parameters
python run_pipeline.py
```

### Command Reference

Below is a comprehensive list of available commands:

#### Basic Configuration

```bash
# Set experiment name (used for saving results)
python run_pipeline.py --experiment my_experiment

# Specify number of users to load
python run_pipeline.py --users 10

# Specify a custom configuration file
python run_pipeline.py --config config/my_custom_config.yaml
```

#### Multi-Run Analysis

```bash
# Run 5 iterations with random user selection
python run_pipeline.py --users 8 --run-count 5 --random-selection

# Run 10 iterations with fixed user selection
python run_pipeline.py --users 8 --run-count 10

# Speed up training by disabling cross-validation
python run_pipeline.py --run-count 5 --no-cv
```

When using `--run-count > 1`, all runs are organized in a single experiment directory:
- `experiments/experiment_name/run1/` - Results for first run
- `experiments/experiment_name/run2/` - Results for second run
- `experiments/experiment_name/results.csv` - Combined results from all runs

#### Pipeline Approach

```bash
# Use per-arm approach instead of per-subject
python run_pipeline.py --per-arm
```

#### Processing Options

```bash
# Skip saving intermediate outputs
python run_pipeline.py --no-save

# Skip model training (extract features only)
python run_pipeline.py --no-train

# Apply signal normalization
python run_pipeline.py --normalize
```

#### Feature Processing

```bash
# Disable Kernel Fisher Discriminant (KFD)
python run_pipeline.py --no-kfd

# Change KFD kernel type
python run_pipeline.py --kfd-kernel rbf

# Set number of KFD components to retain
python run_pipeline.py --kfd-components 15
```

#### Data Augmentation

```bash
# Disable data augmentation
python run_pipeline.py --no-augment

# Set maximum augmentations per sample
python run_pipeline.py --n-augmentations 5
```

#### Gesture Selection

```bash
# Use only specific gesture classes
python run_pipeline.py --selected-gestures 1 2 3
```

#### Combined Example

```bash
# Complex example combining multiple options
python run_pipeline.py --experiment custom_test --users 8 --selected-gestures 2 3 5 --kfd-kernel rbf --kfd-components 12 --n-augmentations 4
```

## Project Structure

```
.
├── config/               # Configuration files
├── data/                 # Data directory
│   ├── raw/              # Raw EMG data
│   ├── processed/        # Processed data
│   └── interim/          # Intermediate data files
├── src/                  # Source code
│   ├── data/             # Data loading modules
│   ├── features/         # Feature extraction modules
│   ├── models/           # ML models
│   └── pipelines/        # Processing pipelines
├── experiments/          # Experiment results
└── run_pipeline.py       # Main runner script
```