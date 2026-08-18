#!/usr/bin/env python
"""
EMG User Identification Pipeline Runner

This script runs the complete EMG user identification pipeline:
1. Load and preprocess raw EMG data
2. Segment the data
3. Extract features
4. Apply optional processing (augmentation, dimensionality reduction)
5. Train and evaluate a machine learning model

Usage Examples:
  # Basic usage (subject-based approach)
  python run_pipeline.py

  # Load a specific number of users
  python run_pipeline.py --users 5

  # Switch to per-arm approach
  python run_pipeline.py --per-arm

  # Feature processing options
  python run_pipeline.py --no-kfd --no-augment

  # Use only specific gestures
  python run_pipeline.py --selected-gestures 1 2 3

  # Extract features without training
  python run_pipeline.py --no-train
  
  # Run multiple iterations with random user selection
  python run_pipeline.py --users 8 --run-count 5 --random-selection
  
  # Speed up training by disabling cross-validation
  python run_pipeline.py --no-cv
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipelines.uci_pipeline import UCIPipeline
from src.pipelines.subject_pipeline import SubjectPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EMG User Identification Pipeline')
    
    # Basic configuration
    parser.add_argument('--config', type=str, default='config/uci_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--experiment', type=str, default='emg_user_id',
                        help='Experiment name (for saving results)')
    parser.add_argument('--users', type=int, default=None,
                        help='Number of users to load (default: all available)')
    
    # Pipeline approach selection
    parser.add_argument('--per-arm', action='store_true',
                        help='Use per-arm approach instead of per-subject')
    
    # Processing options
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving intermediate outputs')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip model training (extract features only)')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply signal normalization')
    
    # Feature processing options
    parser.add_argument('--no-kfd', action='store_true',
                        help='Disable Kernel Fisher Discriminant dimensionality reduction')
    parser.add_argument('--kfd-kernel', type=str, choices=['poly', 'rbf', 'linear'], default='poly',
                        help='KFD kernel type (default: poly)')
    parser.add_argument('--kfd-components', type=int, default=10,
                        help='Number of KFD components to retain (default: 10)')
    
    # Data augmentation options
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--n-augmentations', type=int, default=3,
                        help='Maximum augmentations per sample (default: 3)')
    
    # Gesture selection
    parser.add_argument('--selected-gestures', type=int, nargs='+',
                        help='Specific gesture classes to use (default: all)')
    
    # Multi-run options
    parser.add_argument('--run-count', type=int, default=1,
                        help='Number of pipeline runs (default: 1)')
    parser.add_argument('--random-selection', action='store_true',
                        help='Use random user selection instead of sequential')
    
    # Performance options
    parser.add_argument('--no-cv', action='store_true',
                        help='Disable cross-validation for faster training')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    
    print(f"Using configuration from: {args.config}")
    
    # Create experiment directory
    exp_dir = Path(f"experiments/{args.experiment}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results tracking for multiple runs
    all_results = []
    
    # Run the pipeline multiple times if requested
    for run_idx in range(args.run_count):
        print(f"\n{'='*80}")
        print(f"PIPELINE RUN {run_idx + 1}/{args.run_count}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Create pipeline based on the selected approach
        if args.per_arm:
            print("Using per-arm approach (each arm treated as a separate biometric identity)")
            pipeline = UCIPipeline(
                config_path=args.config, 
                users_to_load=args.users,
                random_selection=args.random_selection
            )
        else:
            print("Using per-subject approach (each subject treated as a single biometric identity)")
            pipeline = SubjectPipeline(
                config_path=args.config, 
                users_to_load=args.users,
                random_selection=args.random_selection
            )
        
        # Update pipeline configuration
        # Feature processing settings
        if args.normalize:
            pipeline.config['feature_extraction']['apply_normalization'] = True
            pipeline.feature_extractor.apply_normalization = True
        
        # KFD settings
        pipeline.config['feature_extraction']['use_kfd'] = not args.no_kfd
        pipeline.feature_extractor.use_kfd = not args.no_kfd
        pipeline.config['feature_extraction']['kfd_kernel'] = args.kfd_kernel
        pipeline.feature_extractor.kfd_kernel = args.kfd_kernel
        pipeline.config['feature_extraction']['kfd_components'] = args.kfd_components
        pipeline.feature_extractor.kfd_components = args.kfd_components
        
        # Augmentation settings. NOTE: augmentation is applied inside model
        # training (train-split only), not by the feature extractor, so only
        # the config dict needs updating here - the model reads it directly.
        pipeline.config['feature_extraction']['apply_augmentation'] = not args.no_augment
        pipeline.config['feature_extraction']['n_augmentations'] = args.n_augmentations

        # Update selected gestures if provided
        if args.selected_gestures:
            pipeline.config['feature_extraction']['selected_gestures'] = args.selected_gestures
            pipeline.feature_extractor.selected_gestures = args.selected_gestures

        # Cross-validation settings
        pipeline.config['training']['perform_cv'] = not args.no_cv

        # Display updated feature extraction configuration
        feature_config = pipeline.config['feature_extraction']
        print("\nFinal Feature Extraction Configuration:")
        print(f"  Apply normalization: {pipeline.feature_extractor.apply_normalization}")
        print(f"  Apply data augmentation (training split only): {feature_config['apply_augmentation']}")
        if feature_config['apply_augmentation']:
            print(f"  Number of augmentations per sample: {feature_config['n_augmentations']}")
        print(f"  KFD enabled: {pipeline.feature_extractor.use_kfd}")
        if pipeline.feature_extractor.use_kfd:
            print(f"  KFD kernel: {pipeline.feature_extractor.kfd_kernel}")
            print(f"  KFD components: {pipeline.feature_extractor.kfd_components}")
        print(f"  Cross-validation enabled: {not args.no_cv}")
        
        # Create run-specific experiment name
        if args.run_count > 1:
            # New directory structure: all runs in the main experiment directory
            # Each run in a subdirectory: experiments/experiment_name/run1, run2, etc.
            run_dir_name = f"run{run_idx+1}"
            run_output_dir = exp_dir / run_dir_name
            run_output_dir.mkdir(parents=True, exist_ok=True)
            run_experiment_name = f"{args.experiment}/{run_dir_name}"
        else:
            run_experiment_name = args.experiment
        
        # Run the pipeline
        features, class_labels, model_results = pipeline.run(
            save_output=not args.no_save,
            train_model=not args.no_train,
            experiment_name=run_experiment_name
        )
        
        run_time = time.time() - start_time
        
        # Track results for this run if model training was performed
        if model_results:
            # Calculate number of biometric identities
            biometric_ids = features[:, -1]
            num_identities = len(set(biometric_ids))
            
            # Display results
            print(f"\nBiometric Identification Results:")
            print(f"Model accuracy: {model_results['accuracy']:.4f}")
            print(f"Training time: {model_results['training_time']:.2f} seconds")
            print(f"Number of biometric identities: {num_identities}")
            
            # Store results for this run
            run_result = {
                'run': run_idx + 1,
                'accuracy': model_results['accuracy'],
                'training_time': model_results['training_time'],
                'run_time': run_time,
                'num_identities': num_identities,
                'random_selection': args.random_selection
            }
            
            # Add cross-validation results if available
            if 'cv_mean' in model_results:
                run_result['cv_mean'] = model_results['cv_mean']
                run_result['cv_std'] = model_results['cv_std']
            else:
                run_result['cv_mean'] = model_results['accuracy']
                run_result['cv_std'] = 0
            
            # Add any loaded user IDs if available
            if hasattr(pipeline.data_loader, 'loaded_user_ids'):
                run_result['user_ids'] = ','.join(map(str, pipeline.data_loader.loaded_user_ids))
            
            # Add gesture-specific performance
            if 'gesture_performance' in model_results:
                for gesture, perf in model_results['gesture_performance'].items():
                    run_result[f'gesture_{gesture}_accuracy'] = perf['accuracy']
                    run_result[f'gesture_{gesture}_samples'] = perf['samples']
            
            all_results.append(run_result)
            
            # Configuration summary
            print("\nConfiguration summary:")
            print(f"  Neural network: {pipeline.config['model']['hidden_layer_sizes']}")
            
            if pipeline.config['feature_extraction']['use_kfd']:
                print(f"  KFD: enabled, kernel={pipeline.config['feature_extraction']['kfd_kernel']}, " +
                      f"components={pipeline.config['feature_extraction']['kfd_components']}")
            else:
                print("  KFD: disabled")
                
            if pipeline.config['feature_extraction']['apply_augmentation']:
                print(f"  Data augmentation: enabled, max augmentations={pipeline.config['feature_extraction']['n_augmentations']}")
            else:
                print("  Data augmentation: disabled")
                
            print(f"\nResults saved to: experiments/{run_experiment_name}/")
    
    # Save combined results from all runs to CSV
    if all_results and args.run_count > 1:
        results_df = pd.DataFrame(all_results)
        results_path = exp_dir / f"results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print(f"MULTI-RUN SUMMARY ({args.run_count} runs)")
        print(f"{'='*80}")
        print(f"Mean accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
        print(f"Min accuracy: {results_df['accuracy'].min():.4f}")
        print(f"Max accuracy: {results_df['accuracy'].max():.4f}")
        print(f"Mean training time: {results_df['training_time'].mean():.2f} seconds")
        print(f"Mean total run time: {results_df['run_time'].mean():.2f} seconds")
        print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main() 