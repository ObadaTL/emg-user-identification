from abc import ABC, abstractmethod
from pathlib import Path
import yaml

class BaseDataLoader(ABC):
    """Base class for all data loaders"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.interim_data_path = Path(self.config['paths']['interim_data'])
        
        # Create directories if they don't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.interim_data_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @abstractmethod
    def load_raw_data(self):
        """Load raw data from source"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data):
        """Preprocess raw data"""
        pass
    
    @abstractmethod
    def segment_data(self, data):
        """Segment data for feature extraction"""
        pass