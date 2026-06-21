import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class TFEMGUserIdentifier:
    """
    EMG-based user identification model using TensorFlow Multi-Layer Perceptron.
    
    Implements a neural network approach for EMG-based user identification
    with configurable network architecture and training parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EMG user identifier model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.random_state = config.get('model', {}).get('random_state', 42)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # TensorFlow parameters
        self.hidden_layer_sizes = config.get('model', {}).get('hidden_layer_sizes', [100, 50])
        self.activation = config.get('model', {}).get('activation', 'relu')
        self.batch_size = config.get('model', {}).get('tf_batch_size', 32)
        self.epochs = config.get('model', {}).get('tf_epochs', 50)
        self.dropout_rate = config.get('model', {}).get('tf_dropout_rate', 0.2)
        self.l2_lambda = config.get('model', {}).get('tf_l2_lambda', 0.001)  # L2 regularization
        self.use_batch_norm = config.get('model', {}).get('tf_use_batch_norm', True)  # Batch normalization
        self.optimizer_name = config.get('model', {}).get('tf_optimizer', 'adam')
        self.learning_rate = config.get('model', {}).get('tf_learning_rate', 0.001)
        self.use_lr_schedule = config.get('model', {}).get('tf_use_lr_schedule', True)
        self.early_stopping = config.get('model', {}).get('tf_early_stopping', True)
        self.patience = config.get('model', {}).get('tf_patience', 10)
        self.class_weight_balanced = config.get('model', {}).get('tf_class_weight_balanced', True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        print(f"Using TensorFlow MLP with {self.hidden_layer_sizes} hidden layers")
        print(f"Regularization: L2={self.l2_lambda}, Dropout={self.dropout_rate}")
        print(f"Using batch normalization: {self.use_batch_norm}")
        print(f"Using learning rate scheduling: {self.use_lr_schedule}")
        print(f"Using balanced class weights: {self.class_weight_balanced}")
        
        # Model will be created during training when we know the input shape
        self.model = None
        
    def _create_model(self, input_shape: int, num_classes: int) -> tf.keras.Model:
        """
        Create the TensorFlow model architecture
        
        Args:
            input_shape: Number of input features
            num_classes: Number of output classes (user IDs)
            
        Returns:
            Compiled TensorFlow model
        """
        # Define regularizer
        regularizer = tf.keras.regularizers.l2(self.l2_lambda)
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_shape,)))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layer_sizes):
            # Add dense layer with regularization
            model.add(Dense(
                units, 
                activation=None,  # No activation yet, will apply after batch norm
                kernel_regularizer=regularizer,
                name=f'dense_{i}'
            ))
            
            # Add batch normalization if enabled
            if self.use_batch_norm:
                model.add(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i}'))
            
            # Add activation
            model.add(tf.keras.layers.Activation(self.activation, name=f'activation_{i}'))
            
            # Add dropout
            model.add(Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Output layer
        model.add(Dense(num_classes, activation='softmax', name='output'))
        
        # Setup optimizer with learning rate
        if self.optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=self.learning_rate, weight_decay=self.l2_lambda)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_callbacks(self, monitor: str = 'val_loss') -> List:
        """
        Get callbacks for model training
        
        Args:
            monitor: Metric to monitor for early stopping and LR reduction
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Add early stopping if configured
        if self.early_stopping:
            early_stop = EarlyStopping(
                monitor=monitor,
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Add learning rate scheduler if configured
        if self.use_lr_schedule:
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,  # Reduce by half
                patience=self.patience // 2,  # Reduce before early stopping kicks in
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        return callbacks
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for imbalanced datasets
        
        Args:
            y: Array of class labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        n_classes = len(class_counts)
        
        # Compute weights: inverse of frequency
        weights = {}
        for i in range(n_classes):
            if class_counts[i] > 0:  # Avoid division by zero
                weights[i] = total_samples / (n_classes * class_counts[i])
            else:
                weights[i] = 1.0
                
        return weights

    def train(self, features: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the model on extracted features.
        
        Args:
            features: Feature matrix with shape (n_samples, n_features + 1)
                     Last column should contain the target labels (user IDs)
            test_size: Proportion of data to use for testing
            
        Returns:
            dict: Training results including accuracy, training time, etc.
        """
        print("\nTraining TensorFlow user identification model...")
        
        # Extract features and target
        X = features[:, :-1]  # All columns except the last
        y = features[:, -1]   # Last column is the target (user ID)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Convert user IDs to integers if necessary
        y = y.astype(int)
        
        # Map original user IDs to consecutive integers starting from 0
        # This is critical for TensorFlow's to_categorical function
        unique_ids = np.unique(y)
        self.id_mapping = {original_id: idx for idx, original_id in enumerate(unique_ids)}
        self.reverse_mapping = {idx: original_id for original_id, idx in self.id_mapping.items()}
        
        # Transform the IDs to 0-indexed for keras
        y_transformed = np.array([self.id_mapping[id_val] for id_val in y])
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=test_size, random_state=self.random_state, stratify=y_transformed
        )
        print(f"Training on {len(X_train)} samples")
        print(f"Testing on {len(X_test)} samples")
        
        # Determine number of classes (now they are 0-indexed)
        self.num_classes = len(unique_ids)
        print(f"Number of classes (unique user IDs): {self.num_classes}")
        
        # Convert targets to one-hot encoding
        y_train_categorical = to_categorical(y_train, num_classes=self.num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=self.num_classes)
        
        # Create model
        input_shape = X_train.shape[1]
        self.model = self._create_model(input_shape, self.num_classes)
        
        # Print model summary
        self.model.summary()
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Compute class weights if enabled
        class_weights = None
        if self.class_weight_balanced:
            class_weights = self._compute_class_weights(y_train)
            print("Using class weights:", class_weights)
        
        # Train the model
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train_categorical,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Evaluate the model
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy using the transformed labels
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store test data for later visualization
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        # Save training history for plotting
        self.history = history.history
        
        # Return results
        results = {
            'accuracy': accuracy,
            'training_time': training_time,
            'classification_report': report,
            'history': history.history
        }
        
        return results
    
    def evaluate_with_gestures(self, features: np.ndarray, 
                              class_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model performance for each gesture type
        
        Args:
            features: Feature matrix with shape (n_samples, n_features + 1)
            class_labels: Array of gesture class labels
            
        Returns:
            dict: Performance metrics by gesture
        """
        print("\nEvaluating model performance by gesture type...")
        
        # Extract features and target
        X = features[:, :-1]
        y = features[:, -1]
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Convert user IDs to integers
        y = y.astype(int)
        
        # If ID mapping isn't already created, create it
        if not hasattr(self, 'id_mapping'):
            unique_ids = np.unique(y)
            self.id_mapping = {original_id: idx for idx, original_id in enumerate(unique_ids)}
            self.reverse_mapping = {idx: original_id for original_id, idx in self.id_mapping.items()}
        
        # Transform the IDs to 0-indexed for keras
        y_transformed = np.array([self.id_mapping[id_val] for id_val in y])
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, _, class_labels_test = train_test_split(
            X, y_transformed, class_labels, test_size=0.2, random_state=self.random_state, stratify=y_transformed
        )
        
        # Train the model if not already trained
        if self.model is None:
            # Determine number of classes
            self.num_classes = len(np.unique(y_transformed))
            
            # Convert targets to one-hot encoding
            y_train_categorical = to_categorical(y_train, num_classes=self.num_classes)
            
            # Create model
            input_shape = X_train.shape[1]
            self.model = self._create_model(input_shape, self.num_classes)
            
            # Define callbacks
            callbacks = []
            if self.early_stopping:
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1
                )
                callbacks.append(early_stop)
            
            # Train the model
            self.model.fit(
                X_train, y_train_categorical,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )
            
            self.X_test = X_test
            self.y_test = y_test
            y_pred_proba = self.model.predict(X_test)
            self.y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get unique gesture classes
        unique_classes = np.unique(class_labels_test)
        
        # Evaluate for each gesture
        gesture_performance = {}
        for gesture in unique_classes:
            # Get samples for this gesture
            gesture_mask = (class_labels_test == gesture)
            X_gesture = X_test[gesture_mask]
            y_gesture = y_test[gesture_mask]
            
            if len(X_gesture) == 0:
                continue
            
            # Make predictions
            y_pred_proba = self.model.predict(X_gesture)
            y_pred_gesture = np.argmax(y_pred_proba, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_gesture, y_pred_gesture)
            
            # Store results
            gesture_performance[int(gesture)] = {
                'accuracy': accuracy,
                'samples': len(X_gesture)
            }
            
            print(f"Gesture {int(gesture)}: Accuracy = {accuracy:.4f} ({len(X_gesture)} samples)")
        
        return gesture_performance
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file"""
        if self.model is None:
            print("Model needs to be trained before saving")
            return
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the TensorFlow model
        tf_path = str(Path(filepath).with_suffix(''))  # Remove extension for TF format
        self.model.save(tf_path)
        
        # Save the scaler separately
        import joblib
        scaler_path = str(Path(filepath).with_suffix('.scaler'))
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {tf_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file"""
        # Load the TensorFlow model
        tf_path = str(Path(filepath).with_suffix(''))  # Remove extension for TF format
        self.model = tf.keras.models.load_model(tf_path)
        
        # Load the scaler
        import joblib
        scaler_path = str(Path(filepath).with_suffix('.scaler'))
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from {tf_path}")
        print(f"Scaler loaded from {scaler_path}")
    
    def visualize_confusion_matrix(self, filepath: Optional[str] = None) -> None:
        """
        Visualize the confusion matrix of model predictions
        
        Args:
            filepath: Optional path to save the visualization
        """
        if not hasattr(self, 'conf_matrix'):
            print("Model needs to be trained before visualizing confusion matrix")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Get unique classes from test data if available
        if hasattr(self, 'y_test'):
            classes = np.unique(self.y_test).astype(int)
        else:
            classes = np.arange(len(self.conf_matrix)).astype(int)
        
        # Plot confusion matrix
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'User Identification Confusion Matrix - Accuracy: {accuracy_score(self.y_test, self.y_pred):.4f}')
        plt.ylabel('True User ID')
        plt.xlabel('Predicted User ID')
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {filepath}")
        else:
            plt.show()
    
    def visualize_training_history(self, filepath: Optional[str] = None) -> None:
        """
        Visualize the training history
        
        Args:
            filepath: Optional path to save the visualization
        """
        if not hasattr(self, 'history'):
            print("Model needs to be trained before visualizing history")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {filepath}")
        else:
            plt.show()
    
    def visualize_gesture_performance(self, gesture_results: Dict[int, Dict[str, float]], 
                                     filepath: Optional[str] = None) -> None:
        """
        Visualize performance across different gestures
        
        Args:
            gesture_results: Dictionary of gesture performance metrics
            filepath: Optional path to save the visualization
        """
        if not gesture_results:
            print("No gesture performance data available")
            return
        
        # Extract gestures, accuracies and sample counts
        gestures = []
        accuracies = []
        sample_counts = []
        
        for gesture, metrics in sorted(gesture_results.items()):
            gestures.append(f"Gesture {gesture}")
            accuracies.append(metrics['accuracy'])
            sample_counts = sample_counts + [metrics['samples']]
        
        # Create a figure with two subplots - bar chart and sample distribution
        plt.figure(figsize=(14, 8))
        
        # Plot 1: Accuracy by gesture
        plt.subplot(1, 2, 1)
        bars = plt.bar(gestures, accuracies, color='skyblue')
        
        # Add accuracy values on top of bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)  # Set y-axis limit with some padding
        plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(accuracies):.4f}')
        plt.title('User Identification Accuracy by Gesture Type')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend()
        
        # Plot 2: Sample count by gesture
        plt.subplot(1, 2, 2)
        bars = plt.bar(gestures, sample_counts, color='lightgreen')
        
        # Add sample count values on top of bars
        for bar, count in zip(bars, sample_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.title('Sample Count by Gesture Type')
        plt.ylabel('Number of Samples')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Gesture performance visualization saved to {filepath}")
        else:
            plt.show()
    
    def save_results_summary(self, results: Dict[str, Any], gesture_results: Optional[Dict[int, Dict[str, float]]] = None, 
                            filepath: Optional[str] = None) -> None:
        """
        Save a comprehensive summary of model results
        
        Args:
            results: Results dictionary from training
            gesture_results: Optional dictionary of gesture performance metrics
            filepath: Optional path to save the summary
        """
        if filepath is None:
            filepath = "results_summary.txt"
        
        with open(filepath, 'w') as f:
            f.write("====================================================\n")
            f.write("      EMG-Based User Identification Results         \n")
            f.write("====================================================\n\n")
            
            # Basic results
            f.write(f"Model Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Training Time: {results.get('training_time', 'N/A'):.2f} seconds\n\n")
            
            # Model architecture
            f.write("Model Architecture:\n")
            f.write(f"  Hidden Layers: {self.hidden_layer_sizes}\n")
            f.write(f"  Activation: {self.activation}\n")
            f.write(f"  Dropout Rate: {self.dropout_rate}\n")
            f.write(f"  L2 Regularization: {self.l2_lambda}\n")
            f.write(f"  Batch Normalization: {self.use_batch_norm}\n")
            f.write(f"  Optimizer: {self.optimizer_name}\n\n")
            
            # Number of classes
            if hasattr(self, 'num_classes'):
                f.write(f"Number of Users: {self.num_classes}\n\n")
            
            # Classification report summary
            if 'classification_report' in results:
                report = results['classification_report']
                f.write("Classification Report Summary:\n")
                f.write(f"  Macro Precision: {report.get('macro avg', {}).get('precision', 'N/A'):.4f}\n")
                f.write(f"  Macro Recall: {report.get('macro avg', {}).get('recall', 'N/A'):.4f}\n")
                f.write(f"  Macro F1-Score: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}\n\n")
            
            # Gesture-specific performance
            if gesture_results:
                f.write("Performance by Gesture Type:\n")
                for gesture, metrics in sorted(gesture_results.items()):
                    f.write(f"  Gesture {gesture}: Accuracy = {metrics['accuracy']:.4f} ({metrics['samples']} samples)\n")
            
            # Add timestamp
            import datetime
            f.write(f"\nResults generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Results summary saved to {filepath}")
    
    def save_results_csv(self, results: Dict[str, Any], gesture_results: Optional[Dict[int, Dict[str, float]]] = None, 
                        filepath: Optional[str] = None) -> None:
        """
        Save results in CSV format for easy analysis in spreadsheets
        
        Args:
            results: Results dictionary from training
            gesture_results: Optional dictionary of gesture performance metrics
            filepath: Optional path to save the CSV file
        """
        if filepath is None:
            filepath = "results.csv"
        
        import csv
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(["Metric", "Value"])
            
            # Basic results
            writer.writerow(["Accuracy", f"{results.get('accuracy', 'N/A'):.4f}"])
            writer.writerow(["Training Time (s)", f"{results.get('training_time', 'N/A'):.2f}"])
            
            # Model architecture
            writer.writerow(["Hidden Layers", str(self.hidden_layer_sizes)])
            writer.writerow(["Number of Users", str(getattr(self, 'num_classes', 'N/A'))])
            
            # Additional metrics if available
            if 'classification_report' in results:
                report = results['classification_report']
                writer.writerow(["Macro Precision", f"{report.get('macro avg', {}).get('precision', 'N/A'):.4f}"])
                writer.writerow(["Macro Recall", f"{report.get('macro avg', {}).get('recall', 'N/A'):.4f}"])
                writer.writerow(["Macro F1-Score", f"{report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}"])
            
            # Add empty row before gesture results
            writer.writerow([])
            
            # Gesture-specific performance
            if gesture_results:
                writer.writerow(["Gesture", "Accuracy", "Samples"])
                
                # Calculate mean values for summary
                mean_accuracy = np.mean([metrics['accuracy'] for metrics in gesture_results.values()])
                total_samples = sum(metrics['samples'] for metrics in gesture_results.values())
                
                for gesture, metrics in sorted(gesture_results.items()):
                    writer.writerow([f"Gesture {gesture}", f"{metrics['accuracy']:.4f}", str(metrics['samples'])])
                
                # Add summary row
                writer.writerow(["Mean", f"{mean_accuracy:.4f}", str(total_samples)])
        
        print(f"Results CSV saved to {filepath}")
    
    def save_comprehensive_results(self, 
                                 results: Dict[str, Any], 
                                 gesture_results: Optional[Dict[int, Dict[str, float]]] = None,
                                 exp_dir: Optional[str] = None) -> None:
        """
        Save comprehensive results including visualizations and data files
        
        Args:
            results: Results dictionary from training
            gesture_results: Optional dictionary of gesture performance metrics
            exp_dir: Optional directory to save results (will be created if it doesn't exist)
        """
        if exp_dir is None:
            exp_dir = "experiment_results"
        
        # Create experiment directory
        exp_path = Path(exp_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Save text summary
        self.save_results_summary(results, gesture_results, exp_path / "results_summary.txt")
        
        # Save CSV results
        self.save_results_csv(results, gesture_results, exp_path / "results.csv")
        
        # Save visualizations
        self.visualize_confusion_matrix(exp_path / "confusion_matrix.png")
        self.visualize_training_history(exp_path / "training_history.png")
        
        if gesture_results:
            self.visualize_gesture_performance(gesture_results, exp_path / "gesture_performance.png")
        
        # Save model if needed
        self.save_model(exp_path / "model.joblib")
        
        print(f"\nComprehensive results saved to {exp_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted user IDs
        """
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        y_pred_proba = self.model.predict(X_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred 