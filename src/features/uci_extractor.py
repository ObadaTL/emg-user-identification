import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_extractor import BaseFeatureExtractor
from tqdm import tqdm
import os
from ..features.kfd import KernelFisherDiscriminant

class UCIFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for EMG-based user identification
    
    Implements feature extraction with the following pipeline:
    1. Basic RSS feature extraction from EMG signals
    2. Optional data augmentation to balance class distribution
    3. Optional KFD transformation for dimensionality reduction and improved separation
    """
    
    def __init__(self, config: Dict[str, Any], print_config: bool = False):
        """Initialize the UCI dataset feature extractor with configuration."""
        # Set column names based on UCI dataset structure
        self.ts_col = 'time'
        self.emg_cols = [f'emg_{i}' for i in range(8)]
        self.class_col = 'class'
        self.user_id_col = 'id'
        
        # Extract configuration parameters
        self.config = config
        self.segment_window_size = config.get('segmentation', {}).get('window_size', 200)
        self.segment_step_size = config.get('segmentation', {}).get('step_size', 200)
        
        # Feature extraction parameters
        feature_config = config.get('feature_extraction', {})
        self.use_kfd = feature_config.get('use_kfd', True)
        self.kfd_kernel = feature_config.get('kfd_kernel', 'poly')
        self.kfd_components = feature_config.get('kfd_components', None)
        self.apply_augmentation = feature_config.get('apply_augmentation', True)
        self.n_augmentations = feature_config.get('n_augmentations', 3)
        self.apply_normalization = feature_config.get('apply_normalization', False)
        self.use_enhanced_features = feature_config.get('use_enhanced_features', False)
        
        # Selected gesture classes to include (None = use all)
        self.selected_gestures = feature_config.get('selected_gestures', None)
        if self.selected_gestures:
            print(f"Using only selected gesture classes: {self.selected_gestures}")
        
        # Basic settings
        self.channel_no = config['data']['channel_no']
        
        # Column names
        self.channel_cols = [f'c{i}' for i in range(1, self.channel_no + 1)]
        self.session_id_col = 'idx'
        self.biometric_id_col = 'bio_id'
        
        # Only print configuration if explicitly requested
        if print_config:
            print(f"\nFeature Extraction Configuration:")
            print(f"  Apply normalization: {self.apply_normalization}")
            print(f"  Use enhanced features: {self.use_enhanced_features}")
            print(f"  Apply data augmentation: {self.apply_augmentation}")
            if self.apply_augmentation:
                print(f"  Number of augmentations per sample: {self.n_augmentations}")
        
        # Parallel processing
        self.n_workers = config.get('processing', {}).get('n_workers', -1)
        if self.n_workers <= 0:
            self.n_workers = os.cpu_count()
    
    def extract_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from segmented data
        
        Args:
            data: Segmented EMG data
            
        Returns:
            tuple: (features, class_labels, user_ids)
                features: Feature matrix with shape (n_samples, n_features + 1)
                class_labels: Array of gesture class labels
                user_ids: Array of original user IDs
        """
        print("\nExtracting features from segmented data...")
        
        # Filter data based on selected gestures if specified
        if self.selected_gestures and self.class_col in data.columns:
            print(f"Filtering data to include only gestures: {self.selected_gestures}")
            original_len = len(data)
            data = data[data[self.class_col].isin(self.selected_gestures)]
            print(f"Filtered from {original_len} to {len(data)} segments")
            
            if len(data) == 0:
                raise ValueError(f"No data left after filtering for gestures {self.selected_gestures}")
        
        # Extract basic features from each segment
        features_array, class_labels_array, user_ids_array = self._extract_basic_features(data)
        
        # Apply data augmentation if configured
        if self.apply_augmentation:
            features_array, class_labels_array, user_ids_array = self._augment_data(
                features_array, class_labels_array, user_ids_array)
        
        # NOTE: KFD is now applied during model training to prevent data leakage
        # KFD will be fitted only on training data and then applied to test data
        
        # Get the biometric IDs column (last column)
        bio_ids = features_array[:, -1]
        # Convert to integer
        bio_ids = bio_ids.astype(int)
        # Replace the last column
        features_array[:, -1] = bio_ids
        
        print(f"Extracted features: {features_array.shape}")
        return features_array, class_labels_array, user_ids_array
    
    def _augment_data(self, features: np.ndarray, class_labels: np.ndarray, 
                      user_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment feature data to increase training dataset size and balance classes
        
        Args:
            features: Feature matrix (last column contains biometric IDs)
            class_labels: Array of class labels
            user_ids: Array of original user IDs
        """
        print(f"Applying data augmentation...")
        print(f"Original data shape: {features.shape}")
        
        # Extract features and target
        X = features[:, :-1]
        y = features[:, -1]
        
        # Create arrays to store augmented data
        augmented_features = [features]
        augmented_labels = [class_labels] if class_labels is not None else None
        augmented_ids = [user_ids] if user_ids is not None else None
        
        # Determine class distribution for balanced augmentation
        unique_ids, id_counts = np.unique(y, return_counts=True)
        max_count = np.max(id_counts)
        
        # Apply class-aware augmentation
        for id_value in unique_ids:
            # Get samples for this biometric ID
            id_mask = (y == id_value)
            X_id = X[id_mask]
            y_id = y[id_mask]
            
            # Determine number of augmentations needed for this class
            n_samples = np.sum(id_mask)
            n_augmentations = min(self.n_augmentations, int(np.ceil((max_count - n_samples) / n_samples)))
            
            if n_augmentations <= 0:
                continue
                
            print(f"  Generating {n_augmentations} augmentations per sample for biometric ID {int(id_value)}")
            
            # Get corresponding class labels and user IDs
            class_labels_id = class_labels[id_mask] if class_labels is not None else None
            user_ids_id = user_ids[id_mask] if user_ids is not None else None
            
            # Create augmented versions for this ID
            for i in range(n_augmentations):
                # Create a copy of the original features
                X_aug = X_id.copy()
                
                # Apply augmentation techniques
                noise_level = 0.01 + 0.01 * i
                noise = np.random.normal(0, noise_level, X_aug.shape) * np.mean(np.abs(X_aug))
                X_aug += noise
                
                scale_factor = 0.95 + 0.02 * i
                X_aug *= scale_factor
                
                jitter = np.random.uniform(-0.02, 0.02, X_aug.shape)
                X_aug += jitter * np.mean(np.abs(X_aug))
                
                # Combine augmented features with original target labels
                features_aug = np.column_stack((X_aug, y_id))
                
                # Add to augmented data arrays
                augmented_features.append(features_aug)
                if class_labels is not None:
                    augmented_labels.append(class_labels_id)
                if user_ids is not None:
                    augmented_ids.append(user_ids_id)
        
        # Combine all augmented data
        combined_features = np.vstack(augmented_features)
        combined_labels = np.concatenate(augmented_labels) if class_labels is not None else None
        combined_ids = np.concatenate(augmented_ids) if user_ids is not None else None
        
        # Calculate augmentation factor
        augmentation_factor = combined_features.shape[0] / features.shape[0]
        print(f"After augmentation: {combined_features.shape} ({augmentation_factor:.2f}x increase)")
        
        # Report class balance
        unique_ids_after, id_counts_after = np.unique(combined_features[:, -1], return_counts=True)
        print(f"Class distribution after augmentation:")
        for i, id_value in enumerate(unique_ids_after):
            print(f"  Biometric ID {int(id_value)}: {id_counts_after[i]} samples")
        
        return combined_features, combined_labels, combined_ids
    
    def _extract_basic_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract basic RSS features from each segment"""
        features = []
        class_labels = []
        user_ids = []
        has_class_data = self.class_col in data.columns
        has_user_data = self.user_id_col in data.columns
        
        # Process each segment
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing segments"):
            # Extract channel data
            channel_data = {col: row[col] for col in self.channel_cols}
            biometric_id = row[self.biometric_id_col]
            
            # Get original user ID if available
            if has_user_data:
                user_id = row[self.user_id_col]
                user_ids.append(user_id)
            else:
                user_ids.append(0)
            
            # Get class if available
            if has_class_data:
                class_label = row[self.class_col]
                class_labels.append(class_label)
            
            # Process segment
            feature_vector = self._process_segment(channel_data, biometric_id)
            features.append(feature_vector)
        
        features_array = np.array(features)
        class_labels_array = np.array(class_labels) if has_class_data else None
        user_ids_array = np.array(user_ids)
        
        return features_array, class_labels_array, user_ids_array
    
    def _process_segment(self, channel_data: Dict[str, np.ndarray], biometric_id: int) -> List[float]:
        """Process a single segment to extract RSS features"""
        feature_vector = []
        
        # Process each channel
        for channel in self.channel_cols:
            signal = channel_data[channel]
            
            try:
                self._validate_signal(signal)
                
                # Apply mean normalization if configured
                if self.apply_normalization:
                    normalized = self._mean_normalize(signal)
                else:
                    normalized = signal
                
                # Compute RSS
                rss = self._compute_rss(normalized)
                feature_vector.append(rss)
            except ValueError as e:
                feature_vector.append(0.0)
        
        # Add biometric ID
        feature_vector.append(biometric_id)
        
        return feature_vector

    def _mean_normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply mean normalization to signal"""
        N = len(x)
        denominator = (1 / N) * np.sum(np.abs(x))
        if denominator == 0:
            return np.zeros_like(x)
        return x / denominator
    
    def _compute_rss(self, s: np.ndarray) -> float:
        """Compute Root Sum Square (RSS)"""
        return np.sqrt(np.sum(np.abs(s)**2))

    def _validate_signal(self, signal: np.ndarray) -> None:
        """Validate signal format"""
        if signal.ndim != 1:
            raise ValueError("Signal must be a 1D array")
        if len(signal) < 10:
            raise ValueError("Signal must have at least 10 samples")

    def extract_features_from_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Extract features from a single EMG segment for real-time processing.
        
        Args:
            segment: 2D array of shape (samples, channels) containing EMG data
            
        Returns:
            Feature vector as 2D numpy array with shape (1, n_features)
        """
        # Create a dictionary of channel data
        channel_data = {}
        for i, col in enumerate(self.channel_cols):
            if i < segment.shape[1]:  # Make sure we don't exceed segment's dimensions
                channel_data[col] = segment[:, i]
        
        # Use a placeholder biometric ID (this will be replaced by the model prediction)
        placeholder_bio_id = 1
        
        # Process the segment using the existing method
        features_list = self._process_segment(channel_data, placeholder_bio_id)
        
        # Remove the biometric ID from the end of the feature list
        features_list = features_list[:-1]
        
        # Apply KFD transformation if enabled
        if self.use_kfd and hasattr(self, 'kfd_transformer'):
            # Get the feature vector as a 2D array
            features_array = np.array([features_list])
            
            # Apply the KFD transformation
            features_transformed = self.kfd_transformer.transform(features_array)
            return features_transformed
        
        # Return as a 2D array with a single row
        return np.array([features_list])