#!/usr/bin/env python
"""
Benchmark window sizes for EMG user identification.

This script evaluates the trade-off between:
1. Response time (how quickly can we identify a user)
2. Accuracy (how reliably can we identify the correct user)
3. Processing load (how computationally expensive is the identification)

Results will help optimize the real-time identification system.

Example usage:
  python benchmark_windows.py --users 6 --model sklearn
  python benchmark_windows.py --min-window 250 --max-window 2000 --window-step 250
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.uci_pipeline import UCIPipeline
from src.inference.real_time_identifier import RealTimeEMGIdentifier

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark Window Sizes for EMG User Identification')
    parser.add_argument('--config', type=str, default='config/uci_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--users', type=int, default=6,
                        help='Number of users to load (default: 6)')
    parser.add_argument('--model', type=str, choices=['sklearn', 'tensorflow'], default='sklearn',
                        help='Model type to use (default: sklearn)')
    parser.add_argument('--min-window', type=int, default=250,
                        help='Minimum window size in samples (default: 250)')
    parser.add_argument('--max-window', type=int, default=2000,
                        help='Maximum window size in samples (default: 2000)')
    parser.add_argument('--window-step', type=int, default=250,
                        help='Step size between window sizes (default: 250)')
    parser.add_argument('--min-step', type=int, default=50,
                        help='Minimum step size in samples (default: 50)')
    parser.add_argument('--max-step', type=int, default=400,
                        help='Maximum step size in samples (default: 400)')
    parser.add_argument('--step-increment', type=int, default=50,
                        help='Increment between step sizes (default: 50)')
    parser.add_argument('--output', type=str, default='window_benchmark_results.csv',
                        help='Output CSV file for benchmark results')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    
    print(f"===============================================")
    print(f"   EMG User Identification Window Size Benchmark   ")
    print(f"===============================================")
    
    # Step 1: Run the pipeline to get preprocessed data and trained model
    print(f"Loading {args.users} users and training model...")
    pipeline = UCIPipeline(config_path=args.config, users_to_load=args.users)
    
    # Set model type in config
    pipeline.config['model']['model_type'] = args.model
    
    # Run the pipeline with model training
    features, class_labels, model_results = pipeline.run(
        save_output=True,
        train_model=True,
        experiment_name="window_benchmark"
    )
    
    if not model_results:
        print("Error training model. Aborting.")
        return
    
    # Get the trained model
    model = pipeline._get_model()
    
    print(f"\nModel training completed with accuracy: {model_results['accuracy']:.4f}")
    
    # Step 2: Get the raw preprocessed data for streaming simulation
    try:
        preprocessed_data = pd.read_csv(pipeline.processed_data_path / "preprocessed_data.csv")
        print(f"Loaded preprocessed data: {preprocessed_data.shape}")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return
    
    # Create biometric ID ground truth for evaluation
    true_ids = preprocessed_data['bio_id'].values if 'bio_id' in preprocessed_data.columns else None
    
    # Step 3: Generate window sizes and step sizes to test
    window_sizes = list(range(args.min_window, args.max_window + 1, args.window_step))
    step_sizes = list(range(args.min_step, args.max_step + 1, args.step_increment))
    
    print(f"\nBenchmarking window sizes: {window_sizes}")
    print(f"Benchmarking step sizes: {step_sizes}")
    
    # Step 4: Create a basic identifier with default settings
    identifier = RealTimeEMGIdentifier(
        model=model if args.model == 'sklearn' else model.model,
        config=pipeline.config,
        window_size=1000,  # Default, will be changed during benchmarking
        step_size=200      # Default, will be changed during benchmarking
    )
    
    # Step 5: Run benchmark for each window/step size combination
    print("\nRunning benchmark...")
    
    # For benchmark, use a subset of data to speed up testing
    max_samples = min(30 * 200, len(preprocessed_data))  # 30 seconds of data
    benchmark_data = preprocessed_data.iloc[:max_samples]
    
    # Store results
    results = []
    
    # Create progress bar
    total_combinations = sum(1 for w in window_sizes for s in step_sizes if s <= w)
    with tqdm(total=total_combinations) as pbar:
        for window_size in window_sizes:
            for step_size in step_sizes:
                # Skip invalid combinations
                if step_size > window_size:
                    continue
                
                # Update identifier settings
                identifier.window_size = window_size
                identifier.step_size = step_size
                identifier.reset()
                
                # Run simulation with these settings
                sim_results = identifier.simulate_stream(
                    data=benchmark_data,
                    true_labels=true_ids[:max_samples] if true_ids is not None else None,
                    simulated_rate_hz=200.0,
                    real_time=False  # Run as fast as possible for benchmarking
                )
                
                # Extract key metrics
                result = {
                    'window_size': window_size,
                    'window_time_sec': window_size / 200.0,  # Convert to seconds
                    'step_size': step_size,
                    'step_time_sec': step_size / 200.0,      # Convert to seconds
                    'accuracy': sim_results.get('accuracy', np.nan),
                    'first_identification_time': sim_results.get('first_identification_time', np.nan),
                    'average_processing_time': sim_results.get('average_processing_time', np.nan) * 1000,  # ms
                    'identification_count': sim_results.get('identification_count', 0),
                    'identification_interval': sim_results.get('identification_interval', np.nan)
                }
                
                results.append(result)
                pbar.update(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved benchmark results to {args.output}")
    
    # Step 6: Visualize results
    print("\nGenerating visualizations...")
    
    # Plot 1: Accuracy vs Window Size
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for step in sorted(results_df['step_size'].unique()):
        step_data = results_df[results_df['step_size'] == step]
        plt.plot(step_data['window_size'], step_data['accuracy'], 
                marker='o', label=f'Step: {step}')
    
    plt.xlabel('Window Size (samples)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Window Size')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Step Size')
    
    # Plot 2: First Identification Time vs Window Size
    plt.subplot(2, 2, 2)
    for step in sorted(results_df['step_size'].unique()):
        step_data = results_df[results_df['step_size'] == step]
        plt.plot(step_data['window_size'], step_data['first_identification_time'], 
                marker='o', label=f'Step: {step}')
    
    plt.xlabel('Window Size (samples)')
    plt.ylabel('First Identification Time (seconds)')
    plt.title('Response Time vs Window Size')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Step Size')
    
    # Plot 3: Processing Time vs Window Size
    plt.subplot(2, 2, 3)
    for step in sorted(results_df['step_size'].unique()):
        step_data = results_df[results_df['step_size'] == step]
        plt.plot(step_data['window_size'], step_data['average_processing_time'], 
                marker='o', label=f'Step: {step}')
    
    plt.xlabel('Window Size (samples)')
    plt.ylabel('Processing Time (ms)')
    plt.title('Processing Time vs Window Size')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Step Size')
    
    # Plot 4: Identification Count vs Window Size
    plt.subplot(2, 2, 4)
    for step in sorted(results_df['step_size'].unique()):
        step_data = results_df[results_df['step_size'] == step]
        plt.plot(step_data['window_size'], step_data['identification_count'], 
                marker='o', label=f'Step: {step}')
    
    plt.xlabel('Window Size (samples)')
    plt.ylabel('Number of Identifications')
    plt.title('Identification Count vs Window Size')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Step Size')
    
    plt.tight_layout()
    plt.savefig('window_benchmark_plots.png', dpi=300)
    plt.show()
    
    # Create additional visualization: heatmap of accuracy by window/step size
    plt.figure(figsize=(10, 8))
    
    # Reshape data for heatmap
    heatmap_data = results_df.pivot_table(
        index='window_size', 
        columns='step_size', 
        values='accuracy'
    )
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Accuracy by Window Size and Step Size')
    plt.xlabel('Step Size (samples)')
    plt.ylabel('Window Size (samples)')
    
    plt.tight_layout()
    plt.savefig('window_step_accuracy_heatmap.png', dpi=300)
    plt.show()
    
    # Create response time heatmap
    plt.figure(figsize=(10, 8))
    
    # Reshape data for heatmap
    heatmap_data = results_df.pivot_table(
        index='window_size', 
        columns='step_size', 
        values='first_identification_time'
    )
    
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', fmt='.2f')
    plt.title('Response Time by Window Size and Step Size (seconds)')
    plt.xlabel('Step Size (samples)')
    plt.ylabel('Window Size (samples)')
    
    plt.tight_layout()
    plt.savefig('window_step_response_heatmap.png', dpi=300)
    plt.show()
    
    # Print optimal configurations
    print("\nRecommended Configurations:")
    
    # 1. For highest accuracy
    if not results_df.empty and 'accuracy' in results_df.columns:
        max_acc_row = results_df.loc[results_df['accuracy'].idxmax()]
        print(f"\nHighest accuracy configuration:")
        print(f"  Window size: {int(max_acc_row['window_size'])} samples ({max_acc_row['window_time_sec']:.1f} seconds)")
        print(f"  Step size: {int(max_acc_row['step_size'])} samples ({max_acc_row['step_time_sec']:.1f} seconds)")
        print(f"  Accuracy: {max_acc_row['accuracy']:.4f}")
        print(f"  Response time: {max_acc_row['first_identification_time']:.2f} seconds")
    
    # 2. For fastest response
    if not results_df.empty and 'first_identification_time' in results_df.columns:
        valid_results = results_df.dropna(subset=['first_identification_time', 'accuracy'])
        if not valid_results.empty:
            min_time_row = valid_results.loc[valid_results['first_identification_time'].idxmin()]
            print(f"\nFastest response configuration:")
            print(f"  Window size: {int(min_time_row['window_size'])} samples ({min_time_row['window_time_sec']:.1f} seconds)")
            print(f"  Step size: {int(min_time_row['step_size'])} samples ({min_time_row['step_time_sec']:.1f} seconds)")
            print(f"  Accuracy: {min_time_row['accuracy']:.4f}")
            print(f"  Response time: {min_time_row['first_identification_time']:.2f} seconds")
    
    # 3. Balanced configuration (good accuracy with reasonable response time)
    if not results_df.empty and 'accuracy' in results_df.columns and 'first_identification_time' in results_df.columns:
        valid_results = results_df.dropna(subset=['first_identification_time', 'accuracy'])
        if not valid_results.empty:
            # Normalize metrics to 0-1 scale
            valid_results['norm_accuracy'] = (valid_results['accuracy'] - valid_results['accuracy'].min()) / \
                                          (valid_results['accuracy'].max() - valid_results['accuracy'].min())
            valid_results['norm_response'] = 1 - (valid_results['first_identification_time'] - valid_results['first_identification_time'].min()) / \
                                          (valid_results['first_identification_time'].max() - valid_results['first_identification_time'].min())
            
            # Calculate balanced score (equal weighting of accuracy and response time)
            valid_results['balanced_score'] = 0.6 * valid_results['norm_accuracy'] + 0.4 * valid_results['norm_response']
            
            # Get row with highest balanced score
            balanced_row = valid_results.loc[valid_results['balanced_score'].idxmax()]
            
            print(f"\nBalanced configuration (accuracy and response time):")
            print(f"  Window size: {int(balanced_row['window_size'])} samples ({balanced_row['window_time_sec']:.1f} seconds)")
            print(f"  Step size: {int(balanced_row['step_size'])} samples ({balanced_row['step_time_sec']:.1f} seconds)")
            print(f"  Accuracy: {balanced_row['accuracy']:.4f}")
            print(f"  Response time: {balanced_row['first_identification_time']:.2f} seconds")
    
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main() 