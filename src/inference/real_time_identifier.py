import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque, Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import threading
import queue

class RealTimeEMGIdentifier:
    """
    Real-time EMG-based user identification system.
    
    This class provides functionality for:
    1. Processing streaming EMG data with sliding windows
    2. Aggregating multiple predictions for robust identification
    3. Managing re-identification periods
    4. Benchmarking identification performance
    """
    
    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        window_size: int = 1000,
        step_size: int = 200,
        reidentification_period: float = 30.0,
        confidence_threshold: float = 0.7,
        prediction_buffer_size: int = 5,
        feature_extractor: Any = None
    ):
        """
        Initialize the real-time EMG identifier.
        
        Args:
            model: Trained model with predict method
            config: Configuration dictionary
            window_size: Number of samples in each analysis window
            step_size: Number of samples to advance for each new window
            reidentification_period: Time in seconds between forced re-identifications
            confidence_threshold: Minimum confidence level to accept an identification
            prediction_buffer_size: Number of predictions to aggregate for final decision
            feature_extractor: Optional feature extractor from the pipeline
        """
        # Store the original model
        self.original_model = model
        
        # For sklearn pipelines, extract the scaler and classifier
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            self.is_sklearn_pipeline = True
            self.scaler = model.named_steps['scaler']
            self.model = model.named_steps['classifier']
            print("Extracted classifier and scaler from sklearn pipeline")
            
            # Create a test sample to make sure the scaler is working
            if hasattr(self.scaler, 'n_features_in_'):
                num_features = self.scaler.n_features_in_
                test_sample = np.zeros((1, num_features))
                
                try:
                    # Try to transform the test sample
                    self.scaler.transform(test_sample)
                    print(f"Scaler test successful: {num_features} features")
                except Exception as e:
                    print(f"Scaler test failed: {e}, falling back to original model")
                    self.model = self.original_model
                    self.is_sklearn_pipeline = False
                    self.scaler = None
        else:
            self.is_sklearn_pipeline = False
            self.scaler = None
            self.model = model
        
        self.config = config
        self.window_size = window_size
        self.step_size = step_size
        self.reidentification_period = reidentification_period
        self.confidence_threshold = confidence_threshold
        
        # Store feature extractor if provided
        if feature_extractor is not None:
            self.config['feature_extractor'] = feature_extractor
        
        # State variables
        self.current_user = None
        self.current_confidence = 0.0
        self.last_identification_time = 0
        self.emg_buffer = deque(maxlen=window_size)
        self.prediction_buffer = deque(maxlen=prediction_buffer_size)
        
        # Channel information
        self.channel_no = config['data']['channel_no']
        self.channel_cols = [f'c{i}' for i in range(1, self.channel_no + 1)]
        
        # Performance tracking
        self.processing_times = []
        self.identification_times = []
        self.prediction_history = []
        
        # For simulation
        self.simulation_queue = queue.Queue()
        self.simulation_running = False
        
        # Test the model with dummy data
        self._test_model_inference()
        
        print(f"Real-time EMG identifier initialized:")
        print(f"  Window size: {window_size} samples")
        print(f"  Step size: {step_size} samples")
        print(f"  Re-identification period: {reidentification_period} seconds")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Feature extractor: {'Custom' if 'feature_extractor' in self.config else 'Default RSS'}")
        print(f"  Model type: {'sklearn Pipeline' if self.is_sklearn_pipeline else type(self.model).__name__}")
    
    def _test_model_inference(self):
        """Test the model with dummy data to ensure it's ready for inference"""
        try:
            # Create a dummy window with correct shape
            dummy_window = []
            for _ in range(self.window_size):
                dummy_sample = {col: 0.0 for col in self.channel_cols}
                dummy_window.append(dummy_sample)
            
            # Extract features for the dummy window
            features = self._extract_features(dummy_window)
            
            # Attempt to make a prediction
            if self.is_sklearn_pipeline:
                # For sklearn pipelines, test prediction with the classifier
                self.model.predict(features)
                print("Model test successful with sklearn classifier")
            else:
                # For other models, test with the original model
                self.original_model.predict(features)
                print("Model test successful with original model")
                
        except Exception as e:
            print(f"Warning: Model test failed: {e}")
            print("Will attempt to use original model during inference")
    
    def process_sample(self, sample: Dict[str, float]) -> Optional[Tuple[int, float]]:
        """
        Process a single EMG sample and update identification if needed.
        
        Args:
            sample: Dictionary with EMG channel values
            
        Returns:
            Tuple of (user_id, confidence) if identification changed, None otherwise
        """
        # Add sample to buffer
        self.emg_buffer.append(sample)
        
        # Check if buffer is full for processing
        if len(self.emg_buffer) < self.window_size:
            return None
        
        # Check if we need to re-identify based on time
        current_time = time.time()
        time_since_last = current_time - self.last_identification_time
        should_reidentify = (
            time_since_last >= self.reidentification_period or 
            self.current_user is None
        )
        
        # Process the window if needed
        if should_reidentify or len(self.emg_buffer) >= self.window_size:
            self.last_identification_time = current_time
            
            # Process the window
            start_time = time.time()
            user_id, confidence = self._process_window(list(self.emg_buffer))
            processing_time = time.time() - start_time
            
            # Track performance
            self.processing_times.append(processing_time)
            
            # Add to prediction buffer
            self.prediction_buffer.append((user_id, confidence))
            
            # Get aggregated prediction
            if len(self.prediction_buffer) >= 1:  # Can adjust minimum required predictions
                agg_user_id, agg_confidence = self._aggregate_predictions()
                
                # Update current user if confidence is high enough or forced re-identification
                if agg_confidence >= self.confidence_threshold or should_reidentify:
                    # Only report if user changed or first identification
                    if self.current_user != agg_user_id:
                        self.current_user = agg_user_id
                        self.current_confidence = agg_confidence
                        return (agg_user_id, agg_confidence)
            
            # Slide the window for next processing
            for _ in range(min(self.step_size, len(self.emg_buffer))):
                if self.emg_buffer:
                    self.emg_buffer.popleft()
        
        return None
    
    def _process_window(self, window: List[Dict[str, float]]) -> Tuple[int, float]:
        """
        Process a window of EMG samples to identify the user.
        
        Args:
            window: List of sample dictionaries
            
        Returns:
            Tuple of (predicted_user_id, confidence)
        """
        # Extract features from window - this will handle scaling if needed
        features = self._extract_features(window)
        
        # Make prediction and get confidence
        try:
            # Try using predict_proba for models that support it
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)
                prediction = self.model.predict(features)
                confidence = np.max(proba)
            else:
                # Fall back to simple prediction
                prediction = self.model.predict(features)
                confidence = 1.0  # Default confidence
                
            return int(prediction[0]), float(confidence)
            
        except Exception as e:
            # If we encounter an error with the extracted classifier, try using the original model
            print(f"Warning: Error with classifier: {e}")
            print("Falling back to original model")
            
            # Reset to the original model for future calls
            self.model = self.original_model
            self.is_sklearn_pipeline = False
            self.scaler = None
            
            # Make prediction with original model
            prediction = self.model.predict(features)
            confidence = 1.0  # Default confidence
            
            return int(prediction[0]), float(confidence)
    
    def _extract_features(self, window: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract features from a window of EMG samples.
        
        This method can work with both per-arm and per-subject approaches
        by using the feature extraction logic from the config.
        
        Args:
            window: List of sample dictionaries
            
        Returns:
            Feature vector as numpy array
        """
        # Convert window to DataFrame for easier processing
        df = pd.DataFrame(window)
        
        # Check if we should use the feature extractor from a pipeline
        feature_extractor = self.config.get('feature_extractor', None)
        if feature_extractor and hasattr(feature_extractor, 'extract_features'):
            # Use the feature extractor from the pipeline
            
            # Extract the segment data
            segment = df[self.channel_cols].values
            
            # Create a wrapper method if extract_features_from_segment doesn't exist
            if not hasattr(feature_extractor, 'extract_features_from_segment'):
                # Manually extract features using the _process_segment method or fallback to RSS
                if hasattr(feature_extractor, '_process_segment'):
                    # Create a channel data dictionary for compatibility with _process_segment
                    channel_data = {}
                    for i, col in enumerate(self.channel_cols):
                        if col in df.columns:
                            channel_data[col] = df[col].values
                    
                    # Use a placeholder biometric ID (this will be replaced by the model prediction)
                    placeholder_bio_id = 1
                    features_list = feature_extractor._process_segment(channel_data, placeholder_bio_id)
                    
                    # Remove the biometric ID from the end of the feature list
                    features_list = features_list[:-1]
                    
                    # Return as a 2D array with a single row
                    features = np.array([features_list])
                else:
                    # Fallback to simple RSS extraction
                    print("No compatible feature extraction method found. Falling back to basic RSS extraction.")
                    features = self._extract_rss_features(df)
            else:
                # The method exists, use it directly
                features = feature_extractor.extract_features_from_segment(segment)
        else:
            # Fallback to simple RSS feature extraction
            features = self._extract_rss_features(df)
        
        # For sklearn pipelines, apply scaling separately
        if self.is_sklearn_pipeline and self.scaler is not None:
            try:
                # Verify feature dimensions match what the scaler expects
                if hasattr(self.scaler, 'n_features_in_') and features.shape[1] != self.scaler.n_features_in_:
                    print(f"Warning: Feature dimensions mismatch. Expected {self.scaler.n_features_in_}, got {features.shape[1]}")
                    # For now, use the original model which handles this internally
                    return features
                
                # Apply scaling
                features = self.scaler.transform(features)
            except Exception as e:
                print(f"Warning: Could not apply scaler: {e}")
                # In case of scaling failure, we'll let the model handle it
        
        return features
    
    def _extract_rss_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract basic RSS features from the EMG data.
        
        Args:
            df: DataFrame with EMG channel data
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        for channel in self.channel_cols:
            if channel in df.columns:
                signal = df[channel].values
                # Compute RSS (Root Sum Square)
                rss = np.sqrt(np.sum(np.square(signal)))
                features.append(rss)
            else:
                features.append(0.0)
        
        return np.array([features])
    
    def _aggregate_predictions(self) -> Tuple[int, float]:
        """
        Aggregate multiple predictions for robust identification.
        
        Returns:
            Tuple of (most_likely_user_id, confidence)
        """
        if not self.prediction_buffer:
            return None, 0.0
        
        # Count user IDs
        user_counts = Counter([p[0] for p in self.prediction_buffer])
        
        # Find most common user ID
        most_common_user, count = user_counts.most_common(1)[0]
        
        # Calculate confidence as proportion of votes
        confidence = count / len(self.prediction_buffer)
        
        # Also consider the average confidence of the predictions for this user
        user_confidences = [p[1] for p in self.prediction_buffer if p[0] == most_common_user]
        avg_confidence = sum(user_confidences) / len(user_confidences) if user_confidences else 0
        
        # Combine vote proportion and average confidence
        combined_confidence = 0.7 * confidence + 0.3 * avg_confidence
        
        return most_common_user, combined_confidence
    
    def reset(self) -> None:
        """Reset the identifier state for a new identification session."""
        self.current_user = None
        self.current_confidence = 0.0
        self.last_identification_time = 0
        self.emg_buffer.clear()
        self.prediction_buffer.clear()
    
    def simulate_stream(
        self, 
        data: pd.DataFrame, 
        true_labels: Optional[np.ndarray] = None,
        simulated_rate_hz: float = 200.0,
        real_time: bool = False
    ) -> Dict[str, Any]:
        """
        Simulate a streaming environment with recorded data.
        
        Args:
            data: DataFrame with EMG data (channels as columns)
            true_labels: Ground truth user IDs if available (for accuracy calculation)
            simulated_rate_hz: Simulated sampling rate in Hz
            real_time: Whether to simulate in actual real-time (True) or process at max speed (False)
            
        Returns:
            Dictionary with simulation results
        """
        print(f"\nSimulating EMG stream at {simulated_rate_hz} Hz...")
        self.reset()
        
        # Prepare results storage
        sample_interval = 1.0 / simulated_rate_hz
        identifications = []
        processing_delays = []
        
        # Start simulation time tracking
        sim_start_time = time.time()
        self.last_identification_time = sim_start_time
        
        # Initialize variables for performance tracking
        identification_count = 0
        first_identification_time = None
        last_identification_time = None
        
        # Process each sample
        for i, row in enumerate(data.iterrows()):
            index, sample = row
            
            # Convert row to dictionary for channel values
            sample_dict = {col: float(sample[col]) for col in self.channel_cols if col in sample}
            
            # Process the sample
            start_process = time.time()
            result = self.process_sample(sample_dict)
            process_time = time.time() - start_process
            
            # Track performance
            processing_delays.append(process_time)
            
            # If we got an identification, record it
            if result is not None:
                user_id, confidence = result
                curr_time = time.time() - sim_start_time
                
                # Update identification tracking
                if first_identification_time is None:
                    first_identification_time = curr_time
                last_identification_time = curr_time
                identification_count += 1
                
                identifications.append({
                    'sample_index': i,
                    'time': curr_time,
                    'user_id': user_id,
                    'confidence': confidence,
                    'true_user_id': true_labels[i] if true_labels is not None else None
                })
                
                print(f"Identification at {curr_time:.2f}s: User {user_id} (confidence: {confidence:.2f})")
            
            # Simulate real-time delay if requested
            if real_time:
                elapsed = time.time() - sim_start_time
                expected_time = i * sample_interval
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
        
        # Calculate performance metrics
        sim_duration = time.time() - sim_start_time
        
        # Create results summary
        results = {
            'duration': sim_duration,
            'sample_count': len(data),
            'identifications': identifications,
            'identification_count': identification_count,
            'first_identification_time': first_identification_time,
            'average_processing_time': np.mean(processing_delays),
            'max_processing_time': np.max(processing_delays),
            'identification_interval': (last_identification_time - first_identification_time) / max(1, identification_count - 1) 
                                      if identification_count > 1 else None
        }
        
        # Calculate accuracy if true labels provided
        if true_labels is not None and identifications:
            # Match identifications to true labels
            true_ids = []
            pred_ids = []
            for ident in identifications:
                if ident['true_user_id'] is not None:
                    true_ids.append(ident['true_user_id'])
                    pred_ids.append(ident['user_id'])
            
            if true_ids:
                results['accuracy'] = accuracy_score(true_ids, pred_ids)
        
        # Print summary
        print("\nSimulation completed:")
        print(f"  Duration: {results['duration']:.2f} seconds")
        print(f"  Processed {results['sample_count']} samples")
        print(f"  Made {results['identification_count']} identifications")
        if first_identification_time is not None:
            print(f"  First identification at: {results['first_identification_time']:.2f} seconds")
        if results.get('identification_interval'):
            print(f"  Average time between identifications: {results['identification_interval']:.2f} seconds")
        print(f"  Average processing delay: {results['average_processing_time']*1000:.2f} ms")
        if 'accuracy' in results:
            print(f"  Identification accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def start_streaming_simulation(
        self, 
        data: pd.DataFrame, 
        simulated_rate_hz: float = 200.0,
        callback=None
    ) -> None:
        """
        Start a background thread that simulates streaming data.
        
        Args:
            data: DataFrame with EMG data
            simulated_rate_hz: Simulated sampling rate in Hz
            callback: Function to call with each identification result
        """
        if self.simulation_running:
            print("Simulation already running")
            return
        
        self.simulation_running = True
        
        # Start streaming thread
        def stream_runner():
            sample_interval = 1.0 / simulated_rate_hz
            self.reset()
            start_time = time.time()
            
            for i, row in enumerate(data.iterrows()):
                if not self.simulation_running:
                    break
                
                index, sample = row
                
                # Convert row to dictionary for channel values
                sample_dict = {col: float(sample[col]) for col in self.channel_cols if col in sample}
                
                # Process the sample
                result = self.process_sample(sample_dict)
                
                # Put result in queue for main thread to consume
                if result is not None:
                    self.simulation_queue.put((i, time.time() - start_time, result))
                    if callback:
                        callback(i, time.time() - start_time, result)
                
                # Simulate real-time delay
                elapsed = time.time() - start_time
                expected_time = i * sample_interval
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
            
            self.simulation_running = False
            self.simulation_queue.put(None)  # Signal end of stream
        
        threading.Thread(target=stream_runner).start()
    
    def stop_streaming_simulation(self) -> None:
        """Stop the streaming simulation."""
        self.simulation_running = False
    
    def get_next_identification(self) -> Optional[Tuple[int, float, Tuple[int, float]]]:
        """
        Get the next identification result from the simulation queue.
        
        Returns:
            Tuple of (sample_index, time, (user_id, confidence)) or None if end of stream
        """
        try:
            return self.simulation_queue.get(block=False)
        except queue.Empty:
            return None
    
    def plot_performance(self, results: Dict[str, Any]) -> None:
        """
        Plot performance metrics from a simulation.
        
        Args:
            results: Simulation results dictionary
        """
        if not results or 'identifications' not in results or not results['identifications']:
            print("No identification results to plot")
            return
        
        # Convert identifications to DataFrame for easier plotting
        idents_df = pd.DataFrame(results['identifications'])
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Identification timeline
        plt.subplot(2, 2, 1)
        plt.scatter(idents_df['time'], idents_df['user_id'], c=idents_df['confidence'], 
                   cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(label='Confidence')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Identified User ID')
        plt.title('User Identification Timeline')
        
        # Plot 2: Confidence histogram
        plt.subplot(2, 2, 2)
        plt.hist(idents_df['confidence'], bins=10, alpha=0.7)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Identification Confidence Distribution')
        
        # Plot 3: Time between identifications
        if len(idents_df) > 1:
            plt.subplot(2, 2, 3)
            time_diffs = idents_df['time'].diff().dropna()
            plt.hist(time_diffs, bins=10, alpha=0.7)
            plt.axvline(x=results.get('identification_interval', 0), 
                      color='r', linestyle='--', label='Average')
            plt.xlabel('Time Between Identifications (seconds)')
            plt.ylabel('Count')
            plt.title('Re-identification Interval Distribution')
            plt.legend()
        
        # Plot 4: Processing time
        plt.subplot(2, 2, 4)
        plt.plot(self.processing_times, alpha=0.7)
        plt.axhline(y=results['average_processing_time'], color='r', 
                   linestyle='--', label='Average')
        plt.xlabel('Sample Index')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time per Window')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # If we have true labels, plot confusion matrix
        if 'accuracy' in results:
            if all(i.get('true_user_id') is not None for i in results['identifications']):
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                true_ids = [i['true_user_id'] for i in results['identifications']]
                pred_ids = [i['user_id'] for i in results['identifications']]
                
                plt.figure(figsize=(10, 8))
                cm = confusion_matrix(true_ids, pred_ids)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted User ID')
                plt.ylabel('True User ID')
                plt.title('User Identification Confusion Matrix')
                plt.show()
    
    def benchmark_response_time(
        self, 
        data: pd.DataFrame, 
        window_sizes: List[int] = None,
        step_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """
        Benchmark response time with different window and step sizes.
        
        Args:
            data: DataFrame with EMG data
            window_sizes: List of window sizes to test
            step_sizes: List of step sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if window_sizes is None:
            window_sizes = [500, 1000, 1500, 2000]
        
        if step_sizes is None:
            step_sizes = [100, 200, 300, 400]
        
        results = []
        
        print("\nBenchmarking response time with different parameters:")
        
        # Save original settings
        orig_window = self.window_size
        orig_step = self.step_size
        
        # Test all combinations
        for window in window_sizes:
            for step in step_sizes:
                # Skip invalid combinations
                if step > window:
                    continue
                
                print(f"Testing window={window}, step={step}...")
                
                # Update settings
                self.window_size = window
                self.step_size = step
                self.reset()
                
                # Run simulation (limited samples for speed)
                max_samples = min(5000, len(data))
                sim_result = self.simulate_stream(
                    data.iloc[:max_samples], 
                    simulated_rate_hz=200.0,
                    real_time=False
                )
                
                # Store results
                results.append({
                    'window_size': window,
                    'step_size': step,
                    'first_identification_time': sim_result.get('first_identification_time'),
                    'average_processing_time': sim_result.get('average_processing_time'),
                    'identification_interval': sim_result.get('identification_interval')
                })
        
        # Restore original settings
        self.window_size = orig_window
        self.step_size = orig_step
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot 1: First identification time
        plt.subplot(2, 2, 1)
        for window in window_sizes:
            window_results = results_df[results_df['window_size'] == window]
            if not window_results.empty:
                plt.plot(window_results['step_size'], window_results['first_identification_time'], 
                       marker='o', label=f'Window {window}')
        plt.xlabel('Step Size')
        plt.ylabel('Time to First Identification (s)')
        plt.title('Response Time by Window/Step Size')
        plt.legend()
        
        # Plot 2: Processing time
        plt.subplot(2, 2, 2)
        for window in window_sizes:
            window_results = results_df[results_df['window_size'] == window]
            if not window_results.empty:
                plt.plot(window_results['step_size'], window_results['average_processing_time']*1000, 
                       marker='o', label=f'Window {window}')
        plt.xlabel('Step Size')
        plt.ylabel('Average Processing Time (ms)')
        plt.title('Processing Time by Window/Step Size')
        plt.legend()
        
        # Plot 3: Identification interval
        plt.subplot(2, 2, 3)
        for window in window_sizes:
            window_results = results_df[results_df['window_size'] == window]
            if not window_results.empty:
                valid_results = window_results[window_results['identification_interval'].notna()]
                if not valid_results.empty:
                    plt.plot(valid_results['step_size'], valid_results['identification_interval'], 
                           marker='o', label=f'Window {window}')
        plt.xlabel('Step Size')
        plt.ylabel('Average Time Between Identifications (s)')
        plt.title('Re-identification Interval by Window/Step Size')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\nBenchmark Results:")
        print(results_df.sort_values(by=['window_size', 'step_size']))
        
        # Find optimal settings
        if not results_df.empty:
            # Filter out rows with missing values
            valid_results = results_df.dropna(subset=['first_identification_time'])
            
            if not valid_results.empty:
                # Find fastest response time
                fastest_response = valid_results.loc[valid_results['first_identification_time'].idxmin()]
                
                print("\nRecommended settings for fastest initial response:")
                print(f"  Window size: {int(fastest_response['window_size'])}")
                print(f"  Step size: {int(fastest_response['step_size'])}")
                print(f"  Expected first identification time: {fastest_response['first_identification_time']:.2f} seconds")
                print(f"  Expected processing time: {fastest_response['average_processing_time']*1000:.2f} ms")
        
        return {
            'results_df': results_df,
            'window_sizes': window_sizes,
            'step_sizes': step_sizes
        } 