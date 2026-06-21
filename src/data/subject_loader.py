from pathlib import Path
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from .uci_loader import UCIDataLoader

class SubjectLoader(UCIDataLoader):
    """Modified UCI Data Loader that treats each subject as a single user
    
    Unlike the original loader that treats each arm/session as a separate biometric identity,
    this loader assigns the same biometric ID to both arms of the same subject.
    
    This allows studying how treating each subject as a whole (rather than each arm separately)
    affects the performance of the user identification model.
    """
    
    def __init__(self, config_path: str = "config/uci_config.yaml", users_to_load: int = None, random_selection: bool = False):
        super().__init__(config_path, users_to_load, random_selection)
        print(f"Using SubjectLoader: Each subject will be treated as one user (both arms combined)")
        print(f"Total biometric identities will be {self.users_to_load} instead of {self.users_to_load * 2}")
    
    def _create_biometric_id(self, user_id: int, session_id: int) -> int:
        """Create a biometric ID using only the user ID (ignoring the session/arm)
        
        Args:
            user_id: Original user ID (1-36)
            session_id: Session/arm ID (1-2) - ignored in this implementation
            
        Returns:
            int: Biometric ID equal to the user_id
        """
        # Simply use the user_id as the biometric_id
        # Both arms of the same user will have the same biometric ID
        return user_id 