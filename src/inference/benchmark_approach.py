#!/usr/bin/env python
"""
Benchmark for comparing per-subject vs per-arm identification approaches.

This script compares the two main approaches to EMG-based user identification:
1. Per-subject: Each person is treated as a single biometric identity
2. Per-arm: Each arm is treated as a separate biometric identity

The benchmark measures:
- Accuracy
- Response time
- Processing load
- Confidence scores

Example usage:
  python benchmark_approach.py --users 6 --model sklearn
  python benchmark_approach.py --window 500 --step 100
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.uci_pipeline import UCIPipeline
from src.pipelines.subject_pipeline import SubjectPipeline
from src.inference.real_time_identifier import RealTimeEMGIdentifier

def run_benchmark(pipeline, args, approach_name):
    """Run benchmark for a specific approach"""
    print(f"\n{'-'*50}")
    print(f"Benchmarking {approach_name} approach")
    print(f"{'-'*50}")
    
    # Run the pipeline with model training
    features, class_labels, model_results = pipeline.run(
        save_output=True,
        train_model=True,
        experiment_name=f"benchmark_{approach_name.lower().replace(' ', '_')}"
    )
    
    if not model_results:
        print(f"Error training model for {approach_name}. Skipping.")
        return None
    
    # Get the trained model
    model = pipeline._get_model()
    print(f"\nModel training completed with accuracy: {model_results['accuracy']:.4f}")
    
    # Extract the actual model to use
    if args.model == 'sklearn':
        # For sklearn models, we can use the full pipeline
        model_to_use = model.model
    else:
        # For tensorflow, we need the model attribute of our wrapper
        model_to_use = model.model
    
    # If after several attempts we still encounter errors, add this fallback option:
    # model_to_use = model  # Use the wrapped model directly instead of the pipeline
    
    # Get the preprocessed data for streaming simulation
    try:
        # For UCIPipeline, the processed_data_path is defined
        # For SubjectPipeline, we need to use the output_dir instead
        if hasattr(pipeline, 'processed_data_path'):
            data_path = pipeline.processed_data_path / "preprocessed_data.csv"
        else:
            data_path = pipeline.output_dir / "preprocessed_data.csv"
        
        # Check if the file exists
        if not os.path.exists(data_path):
            print(f"Preprocessing data file not found at {data_path}")
            print("Saving preprocessed data for future use...")
            
            # Get preprocessed data directly from the pipeline
            if hasattr(pipeline.data_loader, 'load_raw_data') and hasattr(pipeline.data_loader, 'preprocess_data'):
                raw_data = pipeline.data_loader.load_raw_data()
                preprocessed_data = pipeline.data_loader.preprocess_data(raw_data)
                
                # Save the preprocessed data
                preprocessed_data.to_csv(data_path, index=False)
                print(f"Preprocessed data saved to {data_path}")
            else:
                raise Exception("Cannot generate preprocessed data: data_loader missing required methods")
        else:
            # Load the preprocessed data from file
            preprocessed_data = pd.read_csv(data_path)
            
        print(f"Loaded preprocessed data: {preprocessed_data.shape}")
    except Exception as e:
        print(f"Error accessing preprocessed data: {e}")
        return None
    
    # Create biometric ID ground truth for evaluation
    true_ids = preprocessed_data['bio_id'].values if 'bio_id' in preprocessed_data.columns else None
    
    # Initialize the real-time EMG identifier
    identifier = RealTimeEMGIdentifier(
        model=model_to_use,
        config=pipeline.config,
        window_size=args.window,
        step_size=args.step,
        reidentification_period=args.period,
        confidence_threshold=args.threshold,
        feature_extractor=pipeline.feature_extractor
    )
    
    # Run the simulation
    max_samples = min(60 * 200, len(preprocessed_data))  # Use up to 60 seconds
    simulation_data = preprocessed_data.iloc[:max_samples]
    
    print(f"\nRunning simulation with {max_samples} samples ({max_samples/200:.1f} seconds)...")
    start_time = time.time()
    sim_results = identifier.simulate_stream(
        data=simulation_data,
        true_labels=true_ids[:max_samples] if true_ids is not None else None,
        simulated_rate_hz=200.0,
        real_time=False  # Run as fast as possible for benchmarking
    )
    total_time = time.time() - start_time
    
    # Add total processing time to results
    sim_results['total_processing_time'] = total_time
    sim_results['samples_per_second'] = max_samples / total_time
    
    # Add approach name to results
    sim_results['approach'] = approach_name
    
    return sim_results

def compare_results(subject_results, arm_results):
    """Compare and visualize the results from both approaches"""
    if subject_results is None or arm_results is None:
        print("Cannot compare results, one or both approaches failed.")
        return
    
    # Create comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    metrics = [
        ('Accuracy', 'accuracy', '{:.4f}'),
        ('First Identification Time (s)', 'first_identification_time', '{:.2f}'),
        ('Average Processing Time (ms)', 'avg_processing_time', '{:.2f}'),
        ('Average Confidence', 'avg_confidence', '{:.4f}'),
        ('Total Processing Time (s)', 'total_processing_time', '{:.2f}'),
        ('Processing Rate (samples/s)', 'samples_per_second', '{:.2f}'),
    ]
    
    print(f"{'Metric':<30} | {'Per-Subject':<15} | {'Per-Arm':<15} | Difference")
    print("-"*80)
    
    for name, key, fmt in metrics:
        if key in subject_results and key in arm_results:
            subject_val = subject_results[key]
            arm_val = arm_results[key]
            diff = subject_val - arm_val
            diff_str = fmt.format(diff)
            if key == 'accuracy' or key == 'avg_confidence' or key == 'samples_per_second':
                # For these metrics, higher is better
                if diff > 0:
                    diff_str = f"+{diff_str} (Subject better)"
                else:
                    diff_str = f"{diff_str} (Arm better)"
            elif key in ['first_identification_time', 'avg_processing_time', 'total_processing_time']:
                # For these metrics, lower is better
                if diff < 0:
                    diff_str = f"{diff_str} (Subject better)"
                else:
                    diff_str = f"+{diff_str} (Arm better)"
            
            print(f"{name:<30} | {fmt.format(subject_val):<15} | {fmt.format(arm_val):<15} | {diff_str}")
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    if 'accuracy' in subject_results and 'accuracy' in arm_results:
        axs[0, 0].bar(['Per-Subject', 'Per-Arm'], 
                      [subject_results['accuracy'], arm_results['accuracy']])
        axs[0, 0].set_title('Identification Accuracy')
        axs[0, 0].set_ylim(0, 1)
        for i, v in enumerate([subject_results['accuracy'], arm_results['accuracy']]):
            axs[0, 0].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # Response time comparison
    if 'first_identification_time' in subject_results and 'first_identification_time' in arm_results:
        axs[0, 1].bar(['Per-Subject', 'Per-Arm'], 
                      [subject_results['first_identification_time'], arm_results['first_identification_time']])
        axs[0, 1].set_title('First Identification Time (seconds)')
        for i, v in enumerate([subject_results['first_identification_time'], arm_results['first_identification_time']]):
            axs[0, 1].text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Processing time comparison
    if 'avg_processing_time' in subject_results and 'avg_processing_time' in arm_results:
        axs[1, 0].bar(['Per-Subject', 'Per-Arm'], 
                      [subject_results['avg_processing_time'], arm_results['avg_processing_time']])
        axs[1, 0].set_title('Average Processing Time (milliseconds)')
        for i, v in enumerate([subject_results['avg_processing_time'], arm_results['avg_processing_time']]):
            axs[1, 0].text(i, v + 0.1, f"{v:.2f}ms", ha='center')
    
    # Confidence comparison
    if 'avg_confidence' in subject_results and 'avg_confidence' in arm_results:
        axs[1, 1].bar(['Per-Subject', 'Per-Arm'], 
                      [subject_results['avg_confidence'], arm_results['avg_confidence']])
        axs[1, 1].set_title('Average Confidence Score')
        axs[1, 1].set_ylim(0, 1)
        for i, v in enumerate([subject_results['avg_confidence'], arm_results['avg_confidence']]):
            axs[1, 1].text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    
    # Save the comparison figure
    output_dir = Path("experiments/benchmark_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "approach_comparison.png")
    
    print(f"\nComparison chart saved to {output_dir}/approach_comparison.png")
    
    # Show the plot
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark Per-Subject vs Per-Arm Approaches')
    parser.add_argument('--config', type=str, default='config/uci_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--users', type=int, default=6,
                        help='Number of users to load (default: 6)')
    parser.add_argument('--model', type=str, choices=['sklearn', 'tensorflow'], default='sklearn',
                        help='Model type to use (default: sklearn)')
    parser.add_argument('--window', type=int, default=1000,
                        help='Window size in samples (default: 1000)')
    parser.add_argument('--step', type=int, default=200,
                        help='Step size in samples (default: 200)')
    parser.add_argument('--period', type=float, default=30.0,
                        help='Re-identification period in seconds (default: 30.0)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    
    print(f"===============================================")
    print(f"   EMG User Identification Approach Benchmark   ")
    print(f"===============================================")
    print(f"This benchmark compares per-subject vs per-arm approaches")
    print(f"Window size: {args.window} samples ({args.window/200:.1f} seconds at 200Hz)")
    print(f"Step size: {args.step} samples ({args.step/200:.1f} seconds at 200Hz)")
    print(f"Loading {args.users} users and using {args.model} model type")
    
    # Create pipelines for both approaches
    subject_pipeline = SubjectPipeline(config_path=args.config, users_to_load=args.users)
    arm_pipeline = UCIPipeline(config_path=args.config, users_to_load=args.users)
    
    # Set model type in configs
    subject_pipeline.config['model']['model_type'] = args.model
    arm_pipeline.config['model']['model_type'] = args.model
    
    # Run benchmarks
    subject_results = run_benchmark(subject_pipeline, args, "Per-Subject")
    arm_results = run_benchmark(arm_pipeline, args, "Per-Arm")
    
    # Compare and visualize results
    compare_results(subject_results, arm_results)
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main() 