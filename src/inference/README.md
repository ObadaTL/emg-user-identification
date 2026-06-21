# EMG-Based Real-Time User Identification

This module provides tools for real-time user identification using EMG signals. It extends the existing EMG user identification system to work in a streaming context, allowing for continuous user identification with configurable parameters.

## Key Components

- `RealTimeEMGIdentifier`: Core class for real-time identification
- `demo_real_time.py`: Demo script showing basic usage
- `benchmark_windows.py`: Tool to find optimal window/step size configurations
- `benchmark_approach.py`: Tool to compare per-arm vs per-subject approaches

## Identification Approaches

The system supports two approaches to biometric identification:

1. **Per-Arm Approach**: Each arm is treated as a separate biometric identity
2. **Per-Subject Approach**: Each person is treated as a single biometric identity, regardless of which arm is used

## How It Works

The real-time identification system works by:

1. Processing EMG data in sliding windows
2. Extracting features from each window
3. Using a pre-trained model to identify the user
4. Aggregating multiple predictions for robust identification
5. Respecting re-identification periods to avoid constant classification

## Usage Examples

### Basic Demo

Run a basic real-time demo with default settings:

```bash
# Per-subject approach (default)
python src/inference/demo_real_time.py --users 6 --model sklearn

# Per-arm approach
python src/inference/demo_real_time.py --users 6 --model sklearn --per-arm
```

This will:
- Load data for 6 users (subjects)
- Train a model using the specified approach and classifier
- Run a real-time simulation with 5-second windows (1000 samples at 200Hz)
- Display visualization of the results

### Comparing Approaches

To compare the per-subject and per-arm approaches:

```bash
python src/inference/benchmark_approach.py --users 6 --model sklearn
```

This will:
- Train models for both approaches
- Run simulations with both approaches
- Compare accuracy, response time, and confidence
- Generate visualizations showing the differences

### Window Size Optimization

Find the optimal window size for your needs:

```bash
python src/inference/benchmark_windows.py --users 6 --model sklearn --min-window 250 --max-window 1500
```

This will:
- Test various window and step size combinations
- Measure accuracy, response time, and processing load for each
- Generate visualizations showing the tradeoffs
- Recommend configurations for:
  - Highest accuracy
  - Fastest response time
  - Balanced performance

## Key Parameters

### Window Size

The window size determines how many samples are collected before making an identification:
- Larger windows (e.g., 1000+ samples / 5+ seconds): Higher accuracy but slower response time
- Smaller windows (e.g., 250-500 samples / 1.25-2.5 seconds): Faster response but potentially lower accuracy

### Step Size

The step size controls how much the window slides after each identification:
- Larger steps: Fewer computations, more gaps between identifications
- Smaller steps: More frequent identifications, higher computational load

### Re-identification Period

The period (in seconds) between forced re-identifications:
- Shorter periods: More frequent identity checks, better for multi-user environments
- Longer periods: Fewer identity checks, more efficient for single-user sessions

## Performance Metrics

When evaluating real-time identification, consider these metrics:

1. **Accuracy**: How often is the correct user identified?
2. **Response Time**: How quickly can we identify a user?
3. **Re-identification Interval**: How frequently does the system make identifications?
4. **Processing Load**: How computationally expensive is the identification process?

## Integrating With Your Application

To integrate this system into your application:

```python
from src.inference.real_time_identifier import RealTimeEMGIdentifier
from src.pipelines.subject_pipeline import SubjectPipeline  # or UCIPipeline for per-arm approach

# First, train a model using your preferred approach
pipeline = SubjectPipeline(config_path='config/uci_config.yaml')
features, class_labels, model_results = pipeline.run(train_model=True)
model = pipeline._get_model()

# Initialize with your trained model and configuration
identifier = RealTimeEMGIdentifier(
    model=model,
    config=pipeline.config,
    window_size=500,  # 2.5 seconds at 200Hz
    step_size=100,    # 0.5 seconds at 200Hz
    reidentification_period=20.0,  # Seconds
    feature_extractor=pipeline.feature_extractor  # Pass feature extractor for consistent feature extraction
)

# Process streaming data
for emg_sample in your_emg_stream:
    result = identifier.process_sample(emg_sample)
    if result is not None:
        user_id, confidence = result
        print(f"Identified user: {user_id} with confidence {confidence:.2f}")
```

## Recommended Configurations

Based on benchmarking:

- **Highest Accuracy**: Window size of 1000-1500 samples (5-7.5 seconds), step size of 200-300 samples
- **Fastest Response**: Window size of 250-500 samples (1.25-2.5 seconds), step size of 50-100 samples
- **Balanced Performance**: Window size of 500-750 samples (2.5-3.75 seconds), step size of 100-200 samples 