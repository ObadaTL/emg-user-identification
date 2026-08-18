#!/usr/bin/env python
"""
Demo of real-time EMG-based user identification.

PROOF OF CONCEPT - not a maintained feature. See src/inference/README.md.
Known limitation: this demo will error out with the default `use_kfd: true`
config, since RealTimeEMGIdentifier does not apply KFD to extracted features.
Pass `--no-kfd` to run_pipeline.py's config, or run this demo against a config
with `use_kfd: false`, to avoid the feature-dimension mismatch.

This script demonstrates how to:
1. Load and preprocess data using the existing pipeline
2. Train a model using the pipeline
3. Set up a real-time identification system
4. Run a simulated streaming session
5. Benchmark the system for optimal window size and step size
6. Visualize real-time performance metrics

Example usage:
  python demo_real_time.py --users 6 --model tensorflow
  python demo_real_time.py --window 500 --step 100 --period 10
  python demo_real_time.py --per-arm  # Use per-arm approach (each arm as separate identity)
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.uci_pipeline import UCIPipeline
from src.pipelines.subject_pipeline import SubjectPipeline
from src.inference.real_time_identifier import RealTimeEMGIdentifier

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo Real-Time EMG User Identification')
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
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking to find optimal window/step size')
    # Add per-arm option to match run_pipeline.py
    parser.add_argument('--per-arm', action='store_true',
                        help='Use per-arm approach instead of per-subject')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    
    print(f"===============================================")
    print(f"   EMG-Based Real-Time User Identification Demo   ")
    print(f"===============================================")
    print(f"Loading {args.users} users and training model...")
    
    # Step 1: Create pipeline based on the selected approach
    if args.per_arm:
        print("Using per-arm approach (each arm treated as a separate biometric identity)")
        pipeline = UCIPipeline(config_path=args.config, users_to_load=args.users)
    else:
        print("Using per-subject approach (each subject treated as a single biometric identity)")
        pipeline = SubjectPipeline(config_path=args.config, users_to_load=args.users)
    
    # Set model type in config
    pipeline.config['model']['model_type'] = args.model
    
    # Run the pipeline with model training
    features, class_labels, model_results = pipeline.run(
        save_output=True,
        train_model=True,
        experiment_name="realtime_demo"
    )
    
    if not model_results:
        print("Error training model. Aborting.")
        return
    
    # Get the trained model
    model = pipeline._get_model()
    
    print(f"\nModel training completed with accuracy: {model_results['accuracy']:.4f}")
    
    # Step 2: Get the raw preprocessed data for streaming simulation
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
        return
    
    # Create biometric ID ground truth for evaluation
    true_ids = preprocessed_data['bio_id'].values if 'bio_id' in preprocessed_data.columns else None
    
    # Step 3: Initialize the real-time EMG identifier
    print(f"\nInitializing real-time EMG identifier:")
    print(f"  Window size: {args.window} samples ({args.window/200:.1f} seconds at 200Hz)")
    print(f"  Step size: {args.step} samples ({args.step/200:.1f} seconds at 200Hz)")
    print(f"  Re-identification period: {args.period} seconds")
    
    # Extract the actual model to use
    if args.model == 'sklearn':
        # For sklearn models, we can use the full pipeline or just the classifier
        # The RealTimeEMGIdentifier will extract the classifier and scaler if needed
        model_to_use = model.model
    else:
        # For tensorflow, we need the model attribute of our wrapper
        model_to_use = model.model
    
    # If after several attempts we still encounter errors, add this fallback option:
    # model_to_use = model  # Use the wrapped model directly instead of the pipeline
    
    identifier = RealTimeEMGIdentifier(
        model=model_to_use,
        config=pipeline.config,
        window_size=args.window,
        step_size=args.step,
        reidentification_period=args.period,
        confidence_threshold=args.threshold,
        feature_extractor=pipeline.feature_extractor
    )
    
    # Step 4: If benchmarking is requested, run it first
    if args.benchmark:
        print("\n===============================================")
        print("Running benchmarking to find optimal parameters")
        print("===============================================")
        
        # Define window and step size ranges to test
        window_sizes = [500, 750, 1000, 1500]
        step_sizes = [100, 200, 300, 400]
        
        # Run benchmarking
        benchmark_results = identifier.benchmark_response_time(
            data=preprocessed_data,
            window_sizes=window_sizes,
            step_sizes=step_sizes
        )
        
        # Get the fastest configuration from benchmarking
        if (benchmark_results and 'results_df' in benchmark_results 
                and not benchmark_results['results_df'].empty):
            best_row = benchmark_results['results_df'].dropna(
                subset=['first_identification_time']
            ).loc[benchmark_results['results_df']['first_identification_time'].idxmin()]
            
            # Update identifier with optimal settings
            print(f"\nUpdating to optimal settings:")
            print(f"  Window size: {int(best_row['window_size'])} samples")
            print(f"  Step size: {int(best_row['step_size'])} samples")
            
            identifier.window_size = int(best_row['window_size'])
            identifier.step_size = int(best_row['step_size'])
            identifier.reset()
    
    # Step 5: Run a simulated streaming session
    print("\n===============================================")
    print("       Running Real-Time Simulation       ")
    print("===============================================")
    
    # For demo purposes, we'll use the first 30 seconds (6000 samples @ 200Hz)
    # of data to simulate a streaming session
    max_samples = min(30 * 200, len(preprocessed_data))
    simulation_data = preprocessed_data.iloc[:max_samples]
    
    # Run the simulation
    sim_results = identifier.simulate_stream(
        data=simulation_data,
        true_labels=true_ids[:max_samples] if true_ids is not None else None,
        simulated_rate_hz=200.0,
        real_time=True  # Run in real-time for demo purposes
    )
    
    # Step 6: Visualize the results
    print("\n===============================================")
    print("       Visualizing Performance Metrics       ")
    print("===============================================")
    
    identifier.plot_performance(sim_results)
    
    # Step 7: Demonstrate a live updating visualization (optional)
    print("\n===============================================")
    print("  Starting Live Streaming Demo (press Ctrl+C to exit)  ")
    print("===============================================")
    
    try:
        # Setup a callback for visualization
        def update_visualization(index, time_elapsed, result):
            user_id, confidence = result
            print(f"\rTime: {time_elapsed:.2f}s - Identified User: {user_id} (conf: {confidence:.2f})", end="")
        
        # Start background streaming
        identifier.start_streaming_simulation(
            data=simulation_data,
            simulated_rate_hz=200.0,
            callback=update_visualization
        )
        
        # Wait for streaming to complete
        while identifier.simulation_running:
            time.sleep(0.1)
            
        print("\nStreaming completed!")
        
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user")
        identifier.stop_streaming_simulation()
    
    print("\nDemo completed successfully!")
    print(f"Identification approach: {'Per-Arm' if args.per_arm else 'Per-Subject'}")
    print(f"Identification accuracy: {sim_results.get('accuracy', 'N/A')}")
    print(f"Average response time: {sim_results.get('first_identification_time', 'N/A')} seconds")
    
if __name__ == "__main__":
    main() 