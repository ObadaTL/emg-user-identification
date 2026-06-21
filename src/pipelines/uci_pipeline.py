from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import cross_val_score, KFold

from src.data.uci_loader import UCIDataLoader
from src.features.uci_extractor import UCIFeatureExtractor
from src.models.sklearn_mlp import EMGUserIdentifier
from src.models.tensorflow_mlp import TFEMGUserIdentifier

class UCIPipeline:
    """Pipeline for EMG-based user identification using UCI dataset
    
    Implements the complete pipeline:
    1. Data loading and preprocessing
    2. Segmentation
    3. Feature extraction with optional data augmentation
    4. Optional KFD dimensionality reduction
    5. Model training and evaluation
    """
    
    def __init__(self, config_path: str = "config/uci_config.yaml", users_to_load: int = None, random_selection: bool = False):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_loader = UCIDataLoader(config_path, users_to_load=users_to_load, random_selection=random_selection)
        self.feature_extractor = UCIFeatureExtractor(self.config)
        
        # Create output directory
        self.output_dir = Path(self.config['paths']['processed_data'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiments directory
        self.experiments_dir = Path(self.config['paths'].get('experiments', 'experiments'))
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_model(self):
        """Get the appropriate model based on configuration"""
        model_type = self.config.get('model', {}).get('model_type', 'sklearn')
        
        if model_type.lower() == 'tensorflow':
            print("Using TensorFlow MLP model")
            return TFEMGUserIdentifier(self.config)
        else:
            print("Using scikit-learn MLP model")
            return EMGUserIdentifier(self.config)
    
    def run(self, save_output: bool = True, train_model: bool = False, experiment_name: str = None) -> tuple:
        """Run the complete pipeline
        
        Args:
            save_output: Whether to save the processed data and features
            train_model: Whether to train a model on the extracted features
            experiment_name: Name for the experiment (used for saving model)
            
        Returns:
            tuple: (features, labels, model_results)
        """
        print(f"\n{'='*50}")
        print(f"Starting EMG User Identification Pipeline")
        print(f"{'='*50}")
        
        # Step 1: Load raw data
        print("\nStep 1: Loading raw data...")
        start_time = time.time()
        raw_data = self.data_loader.load_raw_data()
        load_time = time.time() - start_time
        print(f"Data loading completed in {load_time:.2f} seconds")
        
        # Step 2: Preprocess data
        print("\nStep 2: Preprocessing data...")
        start_time = time.time()
        preprocessed_data = self.data_loader.preprocess_data(raw_data)
        preprocess_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        
        # Step 3: Segment data
        print("\nStep 3: Segmenting data...")
        start_time = time.time()
        segmented_data = self.data_loader.segment_data(preprocessed_data)
        segment_time = time.time() - start_time
        print(f"Segmentation completed in {segment_time:.2f} seconds")
        
        # Step 4: Extract features
        print("\nStep 4: Extracting features...")
        start_time = time.time()
        features, class_labels, user_ids = self.feature_extractor.extract_features(segmented_data)
        feature_time = time.time() - start_time
        print(f"Feature extraction completed in {feature_time:.2f} seconds")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Pipeline Summary:")
        print(f"{'='*50}")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        print(f"Segmented data shape: {segmented_data.shape}")
        print(f"Features shape: {features.shape}")
        
        # Calculate and print feature dimensions
        feature_dim = features.shape[1] - 1  # Exclude biometric ID column
        channel_count = self.config['data']['channel_no']
        
        # Print feature information
        print("\nFeature Information:")
        print(f"  Base features (RSS per channel): {channel_count}")
        
        # Add KFD information
        use_kfd = self.config.get('feature_extraction', {}).get('use_kfd', True)
        if use_kfd:
            kfd_components = self.config.get('feature_extraction', {}).get('kfd_components', None)
            kfd_kernel = self.config.get('feature_extraction', {}).get('kfd_kernel', 'poly')
            print(f"  KFD transformation applied with {kfd_kernel} kernel")
            print(f"  KFD components: {kfd_components if kfd_components else 'auto'}")
        
        print(f"  Total feature dimensions: {feature_dim}")
        
        if class_labels is not None:
            print(f"Class labels shape: {class_labels.shape}")
            unique_classes = np.unique(class_labels)
            print(f"Unique classes: {unique_classes}")
            
            # Report whether filtering was done
            selected_gestures = self.config.get('feature_extraction', {}).get('selected_gestures', None)
            if selected_gestures:
                print(f"NOTE: Only using selected gestures: {selected_gestures}")
                # Verify which classes are actually present in the data
                selected_classes_present = np.isin(unique_classes, selected_gestures)
                filtered_classes = unique_classes[selected_classes_present]
                print(f"Selected classes present in data: {filtered_classes}")
        
        # Print biometric ID information
        biometric_ids = features[:, -1]
        print(f"\nBiometric ID Information:")
        print(f"  Unique biometric identities: {len(np.unique(biometric_ids))}")
        print(f"  Range: {int(np.min(biometric_ids))} to {int(np.max(biometric_ids))}")
        
        # Save outputs if requested
        #if save_output:
        #    self._save_outputs(preprocessed_data, segmented_data, features, class_labels, user_ids)
        
        # Train model if requested
        model_results = None
        if train_model:
            print(f"\n{'='*50}")
            print(f"Step 5: Training User Identification Model")
            print(f"{'='*50}")
            
            # Create model based on configuration
            model = self._get_model()
            
            # Get training parameters from config
            test_size = self.config.get('training', {}).get('test_size', 0.2)
            
            # Extract features and target
            X = features[:, :-1]  # All columns except the last
            y = features[:, -1]   # Last column is the target (user ID)
            
            # Perform cross-validation based on model type
            model_type = self.config.get('model', {}).get('model_type', 'sklearn')
            cv_results = None
            
            # Check if cross-validation is enabled
            perform_cv = self.config.get('training', {}).get('perform_cv', True)
            
            if model_type.lower() == 'sklearn' and perform_cv:
                print("\nPerforming 5-fold cross-validation...")
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model.model, X, y, cv=cv)
                print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"Individual fold scores: {cv_scores}")
                cv_results = {
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
            elif not perform_cv:
                print("Cross-validation disabled, skipping...")
            
            # Train model on full dataset
            model_results = model.train(features, test_size=test_size)
            
            # Add cross-validation results to model_results if available
            if cv_results:
                for key, value in cv_results.items():
                    model_results[key] = value
            
            # Evaluate by gesture if class labels are available
            if class_labels is not None:
                gesture_results = model.evaluate_with_gestures(features, class_labels)
                model_results['gesture_performance'] = gesture_results
            
            # Save model and results if experiment name is provided
            if experiment_name:
                exp_dir = self.experiments_dir / experiment_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                model.save_model(exp_dir / 'model.joblib')
                
                # Save visualizations
                model.visualize_confusion_matrix(exp_dir / 'confusion_matrix.png')
                
                # Save comprehensive results if the method exists, for all model types
                if hasattr(model, 'save_comprehensive_results'):
                    model.save_comprehensive_results(
                        model_results, 
                        model_results.get('gesture_performance', None),
                        exp_dir
                    )
                
                # If using TensorFlow model, also save additional visualizations
                if model_type.lower() == 'tensorflow':
                    if hasattr(model, 'visualize_training_history'):
                        model.visualize_training_history(exp_dir / 'training_history.png')
                
                print(f"\nExperiment results saved to {exp_dir}")
        
        return features, class_labels, model_results
    
    def _save_outputs(self, preprocessed_data, segmented_data, features, class_labels, user_ids=None):
        """Save pipeline outputs to files"""
        # Save preprocessed data
        preprocessed_path = self.output_dir / "preprocessed_data.csv"
        preprocessed_data.to_csv(preprocessed_path, index=False)
        print(f"Saved preprocessed data to {preprocessed_path}")
        
        # Save segmented data
        segmented_path = self.output_dir / "segmented_data.csv"
        segmented_data.to_csv(segmented_path, index=False)
        print(f"Saved segmented data to {segmented_path}")
        
        # Save features as numpy array
        features_path = self.output_dir / "features.npy"
        np.save(features_path, features)
        print(f"Saved features to {features_path}")
        
        # Save class labels if available
        if class_labels is not None:
            class_labels_path = self.output_dir / "class_labels.npy"
            np.save(class_labels_path, class_labels)
            print(f"Saved class labels to {class_labels_path}")
            
        # Save user IDs if available
        if user_ids is not None:
            user_ids_path = self.output_dir / "user_ids.npy"
            np.save(user_ids_path, user_ids)
            print(f"Saved user IDs to {user_ids_path}")


if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = UCIPipeline(users_to_load=5)  # Load only 5 users (10 biometric identities)
    features, class_labels, _ = pipeline.run(train_model=True, experiment_name="biometric_test") 