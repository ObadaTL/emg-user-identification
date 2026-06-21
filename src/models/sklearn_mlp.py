import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

class EMGUserIdentifier:
    """
    EMG-based user identification model using Multi-Layer Perceptron.
    
    Implements a neural network approach for EMG-based user identification
    with configurable network architecture and training parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EMG user identifier model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.random_state = config.get('model', {}).get('random_state', 42)
        self.config = config
        
        # MLP classifier parameters
        hidden_layer_sizes = config.get('model', {}).get('hidden_layer_sizes', [100, 50])
        activation = config.get('model', {}).get('activation', 'relu')
        solver = config.get('model', {}).get('solver', 'adam')
        alpha = config.get('model', {}).get('alpha', 0.0001)
        learning_rate = config.get('model', {}).get('learning_rate', 'adaptive')
        max_iter = config.get('model', {}).get('max_iter', 4000)
        
        # KFD configuration
        self.use_kfd = config.get('feature_extraction', {}).get('use_kfd', True)
        self.kfd_kernel = config.get('feature_extraction', {}).get('kfd_kernel', 'poly')
        self.kfd_components = config.get('feature_extraction', {}).get('kfd_components', 10)
        self.kfd_gamma = config.get('feature_extraction', {}).get('kfd_gamma', None)
        
        # Create classifier
        classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=self.random_state
        )
        print(f"Using MLP classifier with {hidden_layer_sizes} hidden layers")
        
        # Create the pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
    
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
        print("\nTraining user identification model...")
        
        # Extract features and target
        X = features[:, :-1]  # All columns except the last
        y = features[:, -1]   # Last column is the target (user ID)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"Training on {len(X_train)} samples")
        print(f"Testing on {len(X_test)} samples")
        
        # Apply KFD if enabled, properly fitting only on training data
        if self.use_kfd:
            from src.features.kfd import KernelFisherDiscriminant
            print(f"Applying KFD transformation (kernel={self.kfd_kernel}, components={self.kfd_components})")
            
            # Initialize KFD
            kfd = KernelFisherDiscriminant(
                n_components=self.kfd_components,
                kernel=self.kfd_kernel,
                gamma=self.kfd_gamma
            )
            
            # Fit on training data only
            X_train_kfd = kfd.fit_transform(X_train, y_train)
            
            # Transform test data using the fitted transformer
            X_test_kfd = kfd.transform(X_test)
            
            # Store the KFD transformer
            self.kfd_transformer = kfd
            
            # Replace original data with KFD-transformed data
            X_train, X_test = X_train_kfd, X_test_kfd
            
            print(f"KFD transformation applied: {X_train.shape[1]} -> {X_train_kfd.shape[1]} features")
        
        # Perform cross-validation if enabled
        perform_cv = self.config.get('training', {}).get('perform_cv', True)
        cv_results = None
        
        if perform_cv:
            print("\nPerforming 5-fold cross-validation...")
            from sklearn.model_selection import KFold, cross_val_score
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv)
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"Individual fold scores: {cv_scores}")
            cv_results = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        else:
            print("Cross-validation disabled")
        
        # Train the model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
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
        
        # Return results
        results = {
            'accuracy': accuracy,
            'training_time': training_time,
            'classification_report': report
        }
        
        # Add cross-validation results if available
        if cv_results:
            for key, value in cv_results.items():
                results[key] = value
        
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
        
        # Check if model has already been trained and has test data
        if hasattr(self, 'X_test') and hasattr(self, 'y_test') and hasattr(self, 'y_pred'):
            print("Using existing test data for gesture evaluation")
            
            # Get unique gesture classes from the provided class labels
            # First, we need to find which class labels correspond to the test set
            _, _, _, _, _, class_labels_test = train_test_split(
                features[:, :-1], features[:, -1], class_labels, 
                test_size=0.2, random_state=self.random_state, stratify=features[:, -1]
            )
            
            # Evaluate per gesture using existing test data
            unique_classes = np.unique(class_labels_test)
            
            # Evaluate for each gesture
            gesture_performance = {}
            for gesture in unique_classes:
                # Get samples for this gesture
                gesture_mask = (class_labels_test == gesture)
                X_gesture = self.X_test[gesture_mask]
                y_gesture = self.y_test[gesture_mask]
                
                if len(X_gesture) == 0:
                    continue
                
                # Use the existing predictions
                y_pred_gesture = self.model.predict(X_gesture)
                accuracy = accuracy_score(y_gesture, y_pred_gesture)
                
                # Store results
                gesture_performance[int(gesture)] = {
                    'accuracy': accuracy,
                    'samples': len(X_gesture)
                }
                
                print(f"Gesture {int(gesture)}: Accuracy = {accuracy:.4f} ({len(X_gesture)} samples)")
            
            return gesture_performance
        
        # If model hasn't been trained yet, perform a full evaluation
        print("Model hasn't been trained yet, performing full evaluation")
        
        # Extract features and target
        X = features[:, :-1]
        y = features[:, -1]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test, class_labels_train, class_labels_test = train_test_split(
            X, y, class_labels, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Apply KFD if enabled
        if self.use_kfd:
            from src.features.kfd import KernelFisherDiscriminant
            print(f"Applying KFD transformation (kernel={self.kfd_kernel}, components={self.kfd_components})")
            
            # Initialize KFD
            kfd = KernelFisherDiscriminant(
                n_components=self.kfd_components,
                kernel=self.kfd_kernel,
                gamma=self.kfd_gamma
            )
            
            # Fit on training data only
            X_train = kfd.fit_transform(X_train, y_train)
            
            # Transform test data using the fitted transformer
            X_test = kfd.transform(X_test)
            
            # Store the KFD transformer
            self.kfd_transformer = kfd
            
            print(f"KFD transformation applied: {X.shape[1]} -> {X_train.shape[1]} features")
        
        # Train the model
        print("Training model for gesture-specific evaluation")
        self.model.fit(X_train, y_train)
        
        # Store test data for future use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        
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
            
            # Predict and calculate accuracy
            y_pred_gesture = self.model.predict(X_gesture)
            accuracy = accuracy_score(y_gesture, y_pred_gesture)
            
            # Store results
            gesture_performance[int(gesture)] = {
                'accuracy': accuracy,
                'samples': len(X_gesture)
            }
            
            print(f"Gesture {int(gesture)}: Accuracy = {accuracy:.4f} ({len(X_gesture)} samples)")
        
        return gesture_performance
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and KFD transformer to a file
        
        Args:
            filepath: Path to save the model file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and KFD transformer
        save_data = {
            'model': self.model,
            'use_kfd': self.use_kfd,
            'kfd_transformer': getattr(self, 'kfd_transformer', None)
        }
        
        joblib.dump(save_data, filepath)
        print(f"Model and KFD transformer saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and KFD transformer from a file
        
        Args:
            filepath: Path to the saved model file
        """
        saved_data = joblib.load(filepath)
        
        # Support both old format (just the model) and new format (dict with model and transformer)
        if isinstance(saved_data, dict):
            self.model = saved_data['model']
            self.use_kfd = saved_data['use_kfd']
            if saved_data['kfd_transformer'] is not None:
                self.kfd_transformer = saved_data['kfd_transformer']
        else:
            # Old format - just the model
            self.model = saved_data
        
        print(f"Model loaded from {filepath}")
        if hasattr(self, 'kfd_transformer'):
            print(f"KFD transformer also loaded")
    
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
    
    def visualize_gesture_performance(self, gesture_performance: Dict[int, Dict[str, float]], 
                                     filepath: Optional[str] = None) -> None:
        """
        Visualize performance across different gestures
        
        Args:
            gesture_performance: Dictionary with performance metrics per gesture
            filepath: Optional path to save the visualization
        """
        if not gesture_performance:
            print("No gesture performance data available")
            return
        
        # Extract data for plotting
        gestures = sorted(gesture_performance.keys())
        accuracies = [gesture_performance[g]['accuracy'] for g in gestures]
        samples = [gesture_performance[g]['samples'] for g in gestures]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot 1: Accuracy by gesture
        bars = ax1.bar(range(len(gestures)), accuracies, color='skyblue')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracies[i]:.4f}', ha='center', va='bottom')
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('User Identification Accuracy by Gesture Type')
        ax1.set_xticks(range(len(gestures)))
        ax1.set_xticklabels([f'Gesture {g}' for g in gestures])
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Sample count by gesture
        bars = ax2.bar(range(len(gestures)), samples, color='lightgreen')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{samples[i]}', ha='center', va='bottom')
        
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Count by Gesture Type')
        ax2.set_xticks(range(len(gestures)))
        ax2.set_xticklabels([f'Gesture {g}' for g in gestures])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Gesture performance visualization saved to {filepath}")
        else:
            plt.show()

    def save_results_summary(self, results: Dict[str, Any], gesture_performance: Dict[int, Dict[str, float]],
                           filepath: str) -> None:
        """
        Save a comprehensive summary of model results
        
        Args:
            results: Model results dictionary
            gesture_performance: Dictionary with performance metrics per gesture
            filepath: Path to save the summary
        """
        with open(filepath, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("EMG-Based User Identification Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall results
            f.write("Overall Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Training Time: {results.get('training_time', 'N/A'):.2f} seconds\n")
            
            # Model architecture
            f.write("\nModel Architecture:\n")
            f.write("-" * 20 + "\n")
            
            hidden_layers = self.config.get('model', {}).get('hidden_layer_sizes', [100, 50])
            activation = self.config.get('model', {}).get('activation', 'relu')
            solver = self.config.get('model', {}).get('solver', 'adam')
            alpha = self.config.get('model', {}).get('alpha', 0.0001)
            
            f.write(f"Model Type: scikit-learn MLPClassifier\n")
            f.write(f"Hidden Layers: {hidden_layers}\n")
            f.write(f"Activation: {activation}\n")
            f.write(f"Solver: {solver}\n")
            f.write(f"Alpha: {alpha}\n")
            
            # Gesture-specific performance
            if gesture_performance:
                f.write("\nGesture-Specific Performance:\n")
                f.write("-" * 30 + "\n")
                
                for gesture, metrics in sorted(gesture_performance.items()):
                    f.write(f"Gesture {gesture}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"  Samples: {metrics['samples']}\n")
                    f.write("\n")
                
                # Find best performing gesture
                best_gesture = max(gesture_performance.items(), 
                                 key=lambda x: x[1]['accuracy'])
                f.write(f"Best Performing Gesture: Gesture {best_gesture[0]} ")
                f.write(f"(Accuracy: {best_gesture[1]['accuracy']:.4f})\n")
            
            # Add timestamp
            import datetime
            f.write(f"\nGenerated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"Results summary saved to {filepath}")

    def save_results_csv(self, results: Dict[str, Any], gesture_performance: Dict[int, Dict[str, float]],
                        filepath: str) -> None:
        """
        Save results in CSV format for easy analysis
        
        Args:
            results: Model results dictionary
            gesture_performance: Dictionary with performance metrics per gesture
            filepath: Path to save the CSV file
        """
        # Create main results dataframe
        main_results = {
            'accuracy': [results.get('accuracy', 'N/A')],
            'training_time': [results.get('training_time', 'N/A')],
            'model_type': ['sklearn_mlp'],
            'hidden_layers': [str(self.config.get('model', {}).get('hidden_layer_sizes', [100, 50]))]
        }
        
        # Add gesture accuracies
        if gesture_performance:
            for gesture, metrics in sorted(gesture_performance.items()):
                main_results[f'gesture_{gesture}_accuracy'] = [metrics['accuracy']]
                main_results[f'gesture_{gesture}_samples'] = [metrics['samples']]
        
        # Create and save dataframe
        df = pd.DataFrame(main_results)
        df.to_csv(filepath, index=False)
        print(f"Results CSV saved to {filepath}")

    def save_comprehensive_results(self, results: Dict[str, Any], gesture_performance: Dict[int, Dict[str, float]],
                                 output_dir: str) -> None:
        """
        Save all results, including visualizations and data files
        
        Args:
            results: Model results dictionary
            gesture_performance: Dictionary with performance metrics per gesture
            output_dir: Directory to save results
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save visualizations
        self.visualize_confusion_matrix(output_path / 'confusion_matrix.png')
        
        if gesture_performance:
            self.visualize_gesture_performance(gesture_performance, output_path / 'gesture_performance.png')
        
        # Save text summary
        self.save_results_summary(results, gesture_performance, output_path / 'results_summary.txt')
        
        # Save CSV results
        self.save_results_csv(results, gesture_performance, output_path / 'results.csv')
        
        print(f"\nComprehensive results saved to {output_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict user identity from feature vectors
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Array of predicted user IDs
        """
        # Check if model is trained
        if not hasattr(self, 'model') or not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained yet")
        
        # Apply KFD transformation if it was used during training
        if self.use_kfd and hasattr(self, 'kfd_transformer'):
            X = self.kfd_transformer.transform(X)
        
        # Make prediction
        return self.model.predict(X) 