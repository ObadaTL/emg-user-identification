"""Training-set-only data augmentation for EMG biometric features.

IMPORTANT: augment_training_set() must only ever be called on a training split,
after it has been separated from the test/validation data. Augmented rows are
small perturbations of an existing row (noise + scale + jitter) and are highly
correlated with their source row. Augmenting before a train/test split lets
near-duplicate rows end up on both sides of the split, which leaks information
from the test set into training and inflates reported accuracy.
"""

from typing import Optional, Tuple

import numpy as np


def augment_training_set(
    X: np.ndarray,
    y: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    max_augmentations: int = 3,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Class-balance and expand a training set via noise/scale/jitter perturbation.

    For each biometric identity, generates up to `max_augmentations` perturbed
    copies per sample, capped so that no class is pushed far past the size of
    the largest class in the training set.

    Args:
        X: Training feature matrix, shape (n_samples, n_features).
        y: Training target (biometric ID) vector, shape (n_samples,).
        class_labels: Optional parallel gesture-class labels to carry through
            the augmentation, shape (n_samples,).
        max_augmentations: Maximum augmented copies to generate per sample.
        random_state: Seed for reproducible augmentation noise.

    Returns:
        Tuple of (X_augmented, y_augmented, class_labels_augmented). The first
        block of rows is always the original, unmodified training data.
    """
    rng = np.random.default_rng(random_state)

    augmented_X = [X]
    augmented_y = [y]
    augmented_labels = [class_labels] if class_labels is not None else None

    unique_ids, id_counts = np.unique(y, return_counts=True)
    max_count = np.max(id_counts)

    print("Applying data augmentation to training split only...")
    print(f"Training set shape before augmentation: {X.shape}")

    for id_value in unique_ids:
        id_mask = (y == id_value)
        X_id = X[id_mask]
        y_id = y[id_mask]
        n_samples = np.sum(id_mask)

        n_augmentations = min(max_augmentations, int(np.ceil((max_count - n_samples) / n_samples)))
        if n_augmentations <= 0:
            continue

        class_labels_id = class_labels[id_mask] if class_labels is not None else None

        for i in range(n_augmentations):
            X_aug = X_id.copy()

            noise_level = 0.01 + 0.01 * i
            noise = rng.normal(0, noise_level, X_aug.shape) * np.mean(np.abs(X_aug))
            X_aug += noise

            scale_factor = 0.95 + 0.02 * i
            X_aug *= scale_factor

            jitter = rng.uniform(-0.02, 0.02, X_aug.shape)
            X_aug += jitter * np.mean(np.abs(X_aug))

            augmented_X.append(X_aug)
            augmented_y.append(y_id)
            if class_labels is not None:
                augmented_labels.append(class_labels_id)

    X_combined = np.vstack(augmented_X)
    y_combined = np.concatenate(augmented_y)
    labels_combined = np.concatenate(augmented_labels) if class_labels is not None else None

    print(f"Training set shape after augmentation: {X_combined.shape} "
          f"({X_combined.shape[0] / X.shape[0]:.2f}x increase)")

    return X_combined, y_combined, labels_combined
