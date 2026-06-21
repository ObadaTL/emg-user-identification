import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg

class KernelFisherDiscriminant(BaseEstimator, TransformerMixin):
    """
    Kernel Fisher Discriminant Analysis for nonlinear dimensionality reduction
    
    Parameters:
    -----------
    n_components : int
        Number of components to keep
    kernel : str
        Kernel type to be used ('linear', 'rbf', 'poly', etc.)
    gamma : float
        Kernel coefficient for 'rbf', 'poly', etc.
    degree : int
        Degree for 'poly' kernel
    coef0 : float
        Independent term in 'poly' kernel
    """
    
    def __init__(self, n_components=None, kernel='rbf', gamma=None, degree=3, coef0=1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
    def fit(self, X, y):
        """
        Fit the KFD model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        n_samples, n_features = X.shape
        
        # Store training data
        self.X_fit_ = X
        
        # Compute kernel matrix - only pass relevant parameters based on kernel type
        kernel_params = {'gamma': self.gamma}
        if self.kernel == 'poly':
            kernel_params['degree'] = self.degree
            kernel_params['coef0'] = self.coef0
        
        K = pairwise_kernels(X, metric=self.kernel, **kernel_params)
        
        # Center the kernel matrix
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        # Compute between-class and within-class scatter matrices
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Default: keep n_classes-1 components
        if self.n_components is None:
            self.n_components_ = n_classes - 1
        else:
            self.n_components_ = min(self.n_components, n_classes - 1)
        
        # Calculate scatter matrices in kernel space
        S_w = np.zeros((N, N))
        S_b = np.zeros((N, N))
        
        # Class means in kernel space
        means = np.zeros((n_classes, N))
        for i, c in enumerate(classes):
            class_mask = (y == c)
            n_c = np.sum(class_mask)
            means[i] = np.sum(K[:, class_mask], axis=1) / n_c
        
        # Overall mean
        overall_mean = np.mean(K, axis=1)
        
        # Between-class scatter
        for i, c in enumerate(classes):
            n_c = np.sum(y == c)
            mean_diff = means[i] - overall_mean
            S_b += n_c * np.outer(mean_diff, mean_diff)
        
        # Within-class scatter
        for i, c in enumerate(classes):
            class_mask = (y == c)
            K_c = K[:, class_mask]
            mean_c = means[i].reshape(-1, 1)
            centered = K_c - mean_c
            S_w += centered.dot(centered.T)
        
        # Add regularization to S_w to make it invertible
        S_w += np.eye(S_w.shape[0]) * 1e-5
        
        # Solve generalized eigenvalue problem: S_b * alpha = lambda * S_w * alpha
        evals, evecs = linalg.eigh(S_b, S_w)
        
        # Sort eigenvectors by eigenvalues in descending order
        indices = np.argsort(evals)[::-1]
        self.evals_ = evals[indices]
        self.evecs_ = evecs[:, indices]
        
        # Keep only the top n_components eigenvectors
        self.evecs_ = self.evecs_[:, :self.n_components_]
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data
        
        Returns:
        --------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data
        """
        # Compute kernel between X and training data - only pass relevant parameters based on kernel type
        kernel_params = {'gamma': self.gamma}
        if self.kernel == 'poly':
            kernel_params['degree'] = self.degree
            kernel_params['coef0'] = self.coef0
            
        K = pairwise_kernels(X, self.X_fit_, metric=self.kernel, **kernel_params)
        
        # Project data
        return K.dot(self.evecs_)
