from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseFeatureExtractor(ABC):
    """Base class for EMG-based user identification feature extraction
    
    Implements core methods for:
    1. Mean normalization of EMG signals
    2. Root Sum Square (RSS) computation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def extract_features(self, data):
        """Extract features from raw EMG data
        
        Args:
            data: Raw EMG signal data
            
        Returns:
            Extracted features
        """
        pass
    
    def _mean_normalize(self, x: np.ndarray) -> np.ndarray:
        """Perform mean normalization of EMG signal
        
        Implements the formula:
        s_N = x / ((1/N) * sum(|x_i|))
        
        Args:
            x: Input EMG signal
            
        Returns:
            np.ndarray: Normalized signal
        """
        N = len(x)
        denominator = (1 / N) * np.sum(np.abs(x))
        if denominator == 0:
            return np.zeros_like(x)
        return x / denominator
    
    def _compute_rss(self, s: np.ndarray) -> float:
        """Compute Root Sum Square (RSS) of normalized signal
        
        Implements the formula:
        s_RSS = sqrt(sum(|s_n|^2))
        
        Args:
            s: Normalized EMG signal
            
        Returns:
            float: RSS value
        """
        return np.sqrt(np.sum(np.abs(s)**2))
    
    def _validate_signal(self, data: np.ndarray) -> None:
        """Validate EMG signal data
        
        Args:
            data: EMG signal data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if len(data) == 0:
            raise ValueError("Empty signal data")
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Signal contains NaN or Inf values")
    
    def _extract_channel_features(self, channel_data: np.ndarray) -> float:
        """Extract features from a single channel
        
        Process:
        1. Validate signal
        2. Apply mean normalization
        3. Compute RSS
        
        Args:
            channel_data: Single channel EMG data
            
        Returns:
            float: RSS feature value
        """
        self._validate_signal(channel_data)
        normalized_signal = self._mean_normalize(channel_data)
        return self._compute_rss(normalized_signal) 