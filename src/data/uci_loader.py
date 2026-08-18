from pathlib import Path
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from .base_loader import BaseDataLoader
from tqdm import tqdm

class UCIDataLoader(BaseDataLoader):
    """Data loader for UCI EMG dataset with user identification focus
    
    Data structure:
    data/raw/UCI/
        01/  # User directory
            *.csv  # CSV files for this user
        02/
            *.csv
        ...etc
    
    Key changes:
    - Each arm (session) is treated as a separate biometric identity
    - User ID and arm ID are combined to create a unique biometric ID
    - Can specify the number of users to load for testing with smaller datasets
    """
    
    def __init__(self, config_path: str = "config/uci_config.yaml", users_to_load: int = None, random_selection: bool = False):
        super().__init__(config_path)
        # Basic settings
        self.channel_no = self.config['data']['channel_no']
        
        # Allow specifying number of users to load (for testing with smaller datasets)
        self.users_to_load = users_to_load if users_to_load is not None else self.config['data']['users']
        self.total_users = self.config['data']['users']  # Total available users in the dataset
        self.random_selection = random_selection
        
        self.sampling_rate = self.config['data']['sampling_rate']
        
        # Column names for the dataset
        self.channel_cols = [f'c{i}' for i in range(1, self.channel_no + 1)]
        self.user_id_col = 'id'          # Original user ID (1-36)
        self.session_id_col = 'idx'      # Session/arm ID (1-2)
        self.biometric_id_col = 'bio_id' # New combined biometric ID (1-72)
        self.class_col = 'class'         # Gesture class column
        self.time_col = 'time'           # Time column (will be dropped if not needed)
        
        # Essential columns for user identification
        self.essential_cols = (
            self.channel_cols +          # EMG channels
            [self.user_id_col] +         # Original user ID
            [self.session_id_col] +      # Session/arm ID
            [self.class_col]             # Class/gesture information
        )
        
        # Segmentation parameters for feature extraction
        self.segment_samples = self.config['feature_extraction']['segment_samples']
        self.overlap_samples = self.config['feature_extraction']['overlap_samples']
        
        print(f"Initialized UCI Data Loader to load {self.users_to_load} users (out of {self.total_users})")
        if self.random_selection:
            print(f"Using random user selection")
        #print(f"Each user has 2 arms, so we'll have up to {self.users_to_load * 2} biometric identities")
    
    def _create_biometric_id(self, user_id: int, session_id: int) -> int:
        """Create a unique biometric ID by combining user ID and session/arm ID
        
        Args:
            user_id: Original user ID (1-36)
            session_id: Session/arm ID (1-2)
            
        Returns:
            int: Unique biometric ID (1-72)
        """
        # Convert to 1-based indexing for the biometric ID
        # User 1, Arm 1 -> ID 1
        # User 1, Arm 2 -> ID 2
        # User 2, Arm 1 -> ID 3
        # User 2, Arm 2 -> ID 4
        # etc.
        return (user_id - 1) * 2 + session_id
    
    def _load_raw_data_impl(self) -> pd.DataFrame:
        """Implementation of raw data loading
        
        Returns:
            DataFrame with columns [c1...c8, id, idx, bio_id, class]
        """
        print(f"\nLoading data for {self.users_to_load} users...")
        
        all_user_data = []
        loaded_users = 0
        loaded_biometric_ids = set()
        
        # Determine which users to load
        if self.random_selection:
            import random
            # Select random user IDs, without replacement
            selected_user_ids = random.sample(range(1, self.total_users + 1), 
                                             min(self.users_to_load, self.total_users))
            print(f"Randomly selected users: {selected_user_ids}")
            user_ids_to_load = selected_user_ids
            # Store the selected user IDs for logging
            self.loaded_user_ids = selected_user_ids
        else:
            # Sequential loading
            user_ids_to_load = range(1, self.total_users + 1)
            # Will store the actual loaded user IDs
            self.loaded_user_ids = []
        
        # Load data for selected users
        for user_id in tqdm(user_ids_to_load, desc="Loading users"):
            # Stop if we've loaded enough users (only needed for sequential mode)
            if not self.random_selection and loaded_users >= self.users_to_load:
                break
                
            # Format user_id with zero padding (01, 02, etc.)
            user_id_str = f"{user_id:02d}"
            user_data = self._load_single_user(user_id_str)
            
            if user_data is not None:
                # Add biometric ID column (combining user ID and arm/session ID)
                user_data[self.biometric_id_col] = user_data.apply(
                    lambda row: self._create_biometric_id(row[self.user_id_col], row[self.session_id_col]), 
                    axis=1
                )
                
                # Track which biometric IDs we've loaded
                loaded_biometric_ids.update(user_data[self.biometric_id_col].unique())
                
                all_user_data.append(user_data)
                loaded_users += 1
                
                # Store user ID for sequential mode
                if not self.random_selection:
                    self.loaded_user_ids.append(user_id)
        
        if not all_user_data:
            raise ValueError("No valid data loaded for any user. Check your data directory structure.")
        
        # Combine all user data
        combined_data = pd.concat(all_user_data, ignore_index=True)
        
        # Print dataset summary
        print(f"\nDataset Summary:")
        print(f"Total users loaded: {loaded_users} out of {self.users_to_load} requested")
        print(f"Total biometric identities (user-arm combinations): {len(loaded_biometric_ids)}")
        print(f"Total samples: {len(combined_data)}")
        print(f"Columns: {combined_data.columns.tolist()}")
        
        # Print distribution of samples per biometric ID
        bio_id_counts = combined_data[self.biometric_id_col].value_counts()
        print(f"\nSamples per biometric ID:")
        print(f"  Min: {bio_id_counts.min()}")
        print(f"  Max: {bio_id_counts.max()}")
        print(f"  Mean: {bio_id_counts.mean():.1f}")
        
        return combined_data
    
    def _preprocess_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess EMG data
        
        Args:
            data: Raw EMG data
            
        Returns:
            Preprocessed data with quality checks and proper data types
        """
        print("Preprocessing data...")
        
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Drop time column if it exists and is not needed
        if self.time_col in data.columns:
            print(f"Dropping time column: {self.time_col}")
            data = data.drop(columns=[self.time_col])
        
        # Convert channel data to float32 for efficiency
        data[self.channel_cols] = data[self.channel_cols].astype(np.float32)
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values found")
            # Fill missing values with forward fill then backward fill
            data = data.ffill()
            data = data.bfill()
            
            # If still have missing values, fill with zeros
            remaining_missing = data.isnull().sum().sum()
            if remaining_missing > 0:
                print(f"Warning: {remaining_missing} missing values remain after forward/backward fill. Filling with zeros.")
                data = data.fillna(0)
        
        # Ensure user ID is integer
        if self.user_id_col in data.columns:
            data[self.user_id_col] = data[self.user_id_col].astype(int)
        
        # Ensure session ID is integer if present
        if self.session_id_col in data.columns:
            data[self.session_id_col] = data[self.session_id_col].astype(int)
        
        # Ensure biometric ID is integer
        if self.biometric_id_col in data.columns:
            data[self.biometric_id_col] = data[self.biometric_id_col].astype(int)
        
        # Ensure class is integer if present
        if self.class_col in data.columns:
            data[self.class_col] = data[self.class_col].astype(int)
        
        # Signal quality checks
        for col in self.channel_cols:
            # Check for extreme values
            extreme_values = (data[col].abs() > 2000).sum()
            if extreme_values > 0:
                print(f"Warning: {extreme_values} extreme values detected in {col}")
                
                # Clip extreme values to a reasonable range
                data[col] = data[col].clip(-2000, 2000)
            
            # Check for flat signals (potential sensor issues)
            flat_ratio = (data[col].diff() == 0).sum() / len(data)
            if flat_ratio > 0.1:
                print(f"Warning: Possible sensor issue in {col} - {flat_ratio:.2%} of signal is flat")
        
        return data
    
    def _segment_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Segment the continuous EMG data for feature extraction
        
        Args:
            data: DataFrame with continuous EMG data
            
        Returns:
            DataFrame with segmented data, maintaining channel columns and biometric IDs
        """
        print("\nSegmenting data...")
        segmented_data = []
        
        # Group by biometric ID (user-arm combination)
        grouped = data.groupby(self.biometric_id_col)
        
        for bio_id, bio_data in tqdm(grouped, desc="Segmenting by biometric ID"):
            # Create segments with overlap
            for start in range(0, len(bio_data) - self.segment_samples + 1, 
                            self.segment_samples - self.overlap_samples):
                end = start + self.segment_samples
                segment = bio_data.iloc[start:end]
                
                if len(segment) == self.segment_samples:
                    # Keep the data in DataFrame format with channel columns
                    segment_dict = {
                        col: segment[col].values for col in self.channel_cols
                    }
                    
                    # Add biometric ID (primary identifier for the model)
                    segment_dict[self.biometric_id_col] = bio_id
                    
                    # Also keep original user ID and session ID for analysis
                    if self.user_id_col in segment.columns:
                        segment_dict[self.user_id_col] = segment[self.user_id_col].iloc[0]
                    
                    if self.session_id_col in segment.columns:
                        segment_dict[self.session_id_col] = segment[self.session_id_col].iloc[0]
                    
                    # Add class information if available (most common class in segment)
                    if self.class_col in segment.columns:
                        segment_dict[self.class_col] = segment[self.class_col].mode().iloc[0]
                    
                    segmented_data.append(segment_dict)
        
        segments_df = pd.DataFrame(segmented_data)
        print(f"Total segments: {len(segments_df)}")
        
        # Print distribution of segments per biometric ID
        if self.biometric_id_col in segments_df.columns:
            bio_id_counts = segments_df[self.biometric_id_col].value_counts()
            print(f"\nSegments per biometric ID:")
            print(f"  Min: {bio_id_counts.min()}")
            print(f"  Max: {bio_id_counts.max()}")
            print(f"  Mean: {bio_id_counts.mean():.1f}")
        
        return segments_df
    
    def _load_single_user(self, user_id: str) -> Optional[pd.DataFrame]:
        """Load and validate data for a single user
        
        Args:
            user_id: User ID string (e.g., "01")
            
        Returns:
            DataFrame with user's EMG data or None if loading fails
        """
        # Look for user directory
        user_dir = self.raw_data_path / user_id
        
        if not user_dir.exists():
            print(f"No directory found for user {user_id}")
            return None
        
        try:
            # Look for a CSV file with the same name as the directory
            file_path = user_dir / f"{user_id}.csv"
            
            if not file_path.exists():
                print(f"No CSV file found at {file_path}")
                # Try to find any CSV file in the directory
                csv_files = list(user_dir.glob("*.csv"))
                if csv_files:
                    file_path = csv_files[0]
                    print(f"Using alternative CSV file: {file_path.name}")
                else:
                    print(f"No CSV files found in {user_dir}")
                    return None
                
            # Load the CSV file
            try:
                data = pd.read_csv(file_path)
                print(f"Loaded {len(data)} rows from {file_path}")
            except Exception as e:
                print(f"Error reading CSV file {file_path}: {e}")
                return None
            
            # Check if we have the expected EMG channel columns
            missing_channels = [col for col in self.channel_cols if col not in data.columns]
            if missing_channels:
                print(f"Missing expected channel columns: {missing_channels}")
                # Look for columns that might contain EMG data (numeric columns)
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                # If we have at least 8 numeric columns, use the first 8 for EMG channels
                if len(numeric_cols) >= self.channel_no:
                    print(f"Using first {self.channel_no} numeric columns as EMG channels")
                    # Create a mapping from original column names to our expected channel names
                    channel_mapping = {numeric_cols[i]: self.channel_cols[i] for i in range(self.channel_no)}
                    # Rename the columns
                    data = data.rename(columns=channel_mapping)
                else:
                    print(f"Not enough numeric columns found ({len(numeric_cols)})")
                    return None
            
            # Add user ID column if not present
            if self.user_id_col not in data.columns:
                data[self.user_id_col] = int(user_id)
            
            # Add session ID column if not present (with a default value)
            if self.session_id_col not in data.columns:
                print(f"Warning: Session ID column missing for user {user_id}, adding default value")
                data[self.session_id_col] = 1  # Default session ID
            
            # Add class column if not present (with a default value)
            if self.class_col not in data.columns and 'class' in data.columns:
                # Rename 'class' to self.class_col if it exists but with different case
                data = data.rename(columns={'class': self.class_col})
            elif self.class_col not in data.columns:
                print(f"Warning: Class column missing for user {user_id}, adding default value")
                data[self.class_col] = 0  # Default class
            
            # Keep only necessary columns plus time if it exists
            all_cols = self.essential_cols.copy()
            if self.time_col in data.columns:
                all_cols.append(self.time_col)
            
            # Check if all essential columns exist
            missing_essential = [col for col in self.essential_cols if col not in data.columns]
            if missing_essential:
                print(f"Missing essential columns for user {user_id}: {missing_essential}")
                return None
            
            # Select only the columns we need
            try:
                data = data[all_cols]
            except KeyError as e:
                print(f"Error selecting columns for user {user_id}: {e}")
                print(f"Available columns: {data.columns.tolist()}")
                return None
            
            # Check if we have both session IDs (arms)
            unique_sessions = data[self.session_id_col].unique()
            if len(unique_sessions) < 2:
                print(f"Warning: User {user_id} has only {len(unique_sessions)} session(s) instead of 2")
            
            return data
            
        except Exception as e:
            print(f"Error loading data for user {user_id}: {e}")
            return None

    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        output_path = self.processed_data_path / filename
        data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    # Implement the abstract methods from BaseDataLoader
    def load_raw_data(self):
        """Load raw data from source - implementation of abstract method"""
        return self._load_raw_data_impl()
    
    def preprocess_data(self, data=None):
        """Preprocess raw data - implementation of abstract method"""
        if data is None:
            data = self.load_raw_data()
        return self._preprocess_data_impl(data)
    
    def segment_data(self, data=None):
        """Segment data for feature extraction - implementation of abstract method"""
        if data is None:
            data = self.preprocess_data()
        return self._segment_data_impl(data) 