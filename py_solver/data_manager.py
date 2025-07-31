"""
Data Manager Module
Provides functionality for reading CSV files, validating data dimensions, and managing training data
"""

import pandas as pd
import numpy as np
import os
import glob
import time
from typing import Union, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from .config import model_config, StandardizationInfo, MinMaxNormInfo


class DataManager:
    """
    Data Manager class for handling CSV data loading, validation, and preprocessing
    """
    
    def __init__(self, config: model_config):
        """
        Initialize DataManager with model configuration
        
        Args:
            config: Model configuration object containing input/output dimensions
        """
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.total_dim = self.input_dim + self.output_dim
        
        # Data storage
        self.features = None
        self.targets = None
        self.is_loaded = False
        
        # Standardization parameters
        self.input_mean = None
        self.input_std = None
        self.target_mean = None
        self.target_std = None
        
        # Min-max normalization parameters
        self.input_min = None
        self.input_max = None
        self.target_min = None
        self.target_max = None
        
        # Training/validation split
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.is_split = False
        
        print(f"DataManager initialized with input_dim={self.input_dim}, output_dim={self.output_dim}, norm_type={self.config.norm_type}")
    
    def read_training_csv(self, csv_files: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read training data from CSV file(s)
        
        Args:
            csv_files: Single CSV file path, list of CSV file paths, or wildcard pattern
            
        Returns:
            Tuple of (features, targets) as numpy arrays
            
        Raises:
            ValueError: If data dimensions don't match expected dimensions
            FileNotFoundError: If CSV files don't exist
        """
        # Convert to list if single file
        if isinstance(csv_files, str):
            if '*' in csv_files or '?' in csv_files:
                csv_files = glob.glob(csv_files)
                if not csv_files:
                    raise FileNotFoundError(f"No matching CSV files found: {csv_files}")
            else:
                csv_files = [csv_files]
        
        if not isinstance(csv_files, list):
            raise ValueError("csv_files must be a string, list of strings, or wildcard pattern")
        
        print(f"Loading {len(csv_files)} CSV file(s)...")
        start_time = time.time()
        
        all_features = []
        all_targets = []
        total_samples = 0
        
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"File not found: {csv_file}")
            
            features, targets = self._read_single_csv(csv_file)
            all_features.append(features)
            all_targets.append(targets)
            total_samples += len(features)
        
        # Combine all data
        self.features = np.vstack(all_features)
        self.targets = np.vstack(all_targets)
        self.is_loaded = True
        
        total_time = time.time() - start_time
        print(f"Data loading completed: {total_samples:,} samples, total time: {total_time:.2f}s")
        print(f"Data shape: features {self.features.shape}, targets {self.targets.shape}")
        
        return self.features, self.targets
    
    def _read_single_csv(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a single CSV file and validate dimensions
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Tuple of (features, targets) as numpy arrays
            
        Raises:
            ValueError: If data dimensions don't match expected dimensions
        """
        print(f"Loading: {csv_file}")
        start_time = time.time()
        
        # Read CSV file
        try:
            data = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {csv_file}: {e}")
        
        # Validate dimensions
        actual_dim = data.shape[1]
        if actual_dim != self.total_dim:
            raise ValueError(
                f"Data dimension mismatch in {csv_file}: "
                f"expected {self.total_dim} columns (input_dim={self.input_dim} + output_dim={self.output_dim}), "
                f"but got {actual_dim} columns"
            )
        
        # Split into features and targets
        features = data.iloc[:, :self.input_dim].values.astype(np.float32)
        targets = data.iloc[:, self.input_dim:].values.astype(np.float32)
        
        # If output_dim is 1, flatten targets
        if self.output_dim == 1:
            targets = targets.flatten().reshape(-1, 1)
        
        load_time = time.time() - start_time
        print(f"Loading time: {load_time:.2f}s, data size: {len(data):,} samples")
        
        return features, targets
    
    def validate_data(self) -> bool:
        """
        Validate loaded data
        
        Returns:
            True if data is valid, False otherwise
        """
        if not self.is_loaded:
            print("No data loaded")
            return False
        
        if self.features is None or self.targets is None:
            print("Features or targets are None")
            return False
        
        if len(self.features) != len(self.targets):
            print("Number of features and targets don't match")
            return False
        
        if self.features.shape[1] != self.input_dim:
            print(f"Feature dimension mismatch: expected {self.input_dim}, got {self.features.shape[1]}")
            return False
        
        if self.targets.shape[1] != self.output_dim:
            print(f"Target dimension mismatch: expected {self.output_dim}, got {self.targets.shape[1]}")
            return False
        
        print("Data validation passed")
        return True
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets
        
        Args:
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if not self.is_loaded:
            raise ValueError("No data loaded. Call read_training_csv() first.")
        
        if not self.validate_data():
            raise ValueError("Data validation failed")
        
        print(f"Splitting data with test_size={test_size}...")
        start_time = time.time()
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.features, self.targets, 
            test_size=test_size, 
            random_state=random_state
        )
        
        self.is_split = True
        split_time = time.time() - start_time
        
        print(f"Data splitting completed: {split_time:.2f}s")
        print(f"Training set: {len(self.X_train):,} samples")
        print(f"Validation set: {len(self.X_val):,} samples")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def compute_normalization_params(self) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute normalization parameters from training data based on config.norm_type
        
        Returns:
            For standardization: Tuple of (input_mean, input_std, target_mean, target_std)
            For min_max: Tuple of (input_min, input_max, target_min, target_max)
        """
        if not self.is_split:
            raise ValueError("Data not split. Call split_data() first.")
        
        print("Computing normalization parameters...")
        start_time = time.time()
        
        if self.config.norm_type == "standardization":
            # Compute parameters from training data only
            self.input_mean = np.mean(self.X_train, axis=0, dtype=np.float32)
            self.input_std = np.std(self.X_train, axis=0, dtype=np.float32)
            self.target_mean = np.mean(self.y_train, axis=0, dtype=np.float32)
            self.target_std = np.std(self.y_train, axis=0, dtype=np.float32)
            
            # Handle zero standard deviation
            self.input_std = np.where(self.input_std == 0, 1.0, self.input_std)
            self.target_std = np.where(self.target_std == 0, 1.0, self.target_std)
        elif self.config.norm_type == "min_max":
            # For min-max normalization, we need to compute min/max values
            self.input_min = np.min(self.X_train, axis=0)
            self.input_max = np.max(self.X_train, axis=0)
            self.target_min = np.min(self.y_train, axis=0)
            self.target_max = np.max(self.y_train, axis=0)
        else:
            raise ValueError(f"Unsupported normalization type: {self.config.norm_type}")
        
        norm_time = time.time() - start_time
        print(f"Normalization parameters computed: {norm_time:.2f}s")
        
        if self.config.norm_type == "standardization":
            return self.input_mean, self.input_std, self.target_mean, self.target_std
        elif self.config.norm_type == "min_max":
            return self.input_min, self.input_max, self.target_min, self.target_max
        else:
            raise ValueError(f"Unsupported normalization type: {self.config.norm_type}")
    
    def get_normalization_info(self) -> Union[StandardizationInfo, MinMaxNormInfo]:
        """
        Get normalization info object based on config
        
        Returns:
            StandardizationInfo or MinMaxNormInfo object
        """
        if self.config.norm_type == "standardization":
            if self.input_mean is None:
                raise ValueError("Normalization parameters not computed. Call compute_normalization_params() first.")
            return StandardizationInfo(
                input_mean=self.input_mean.tolist(),
                input_std=self.input_std.tolist(),
                target_mean=self.target_mean.tolist() if self.output_dim > 1 else float(self.target_mean[0]),
                target_std=self.target_std.tolist() if self.output_dim > 1 else float(self.target_std[0])
            )
        elif self.config.norm_type == "min_max":
            if self.input_min is None:
                raise ValueError("Normalization parameters not computed. Call compute_normalization_params() first.")
            return MinMaxNormInfo(
                input_min=self.input_min.tolist(),
                input_max=self.input_max.tolist(),
                target_min=self.target_min.tolist() if self.output_dim > 1 else float(self.target_min[0]),
                target_max=self.target_max.tolist() if self.output_dim > 1 else float(self.target_max[0])
            )
        else:
            raise ValueError(f"Unsupported normalization type: {self.config.norm_type}")
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary containing data summary
        """
        if not self.is_loaded:
            return {"error": "No data loaded"}
        
        summary = {
            "total_samples": len(self.features),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "features_shape": self.features.shape,
            "targets_shape": self.targets.shape,
            "is_split": self.is_split,
            "has_norm_params": (self.input_mean is not None) if self.config.norm_type == "standardization" else (self.input_min is not None)
        }
        
        if self.is_split:
            summary.update({
                "train_samples": len(self.X_train),
                "val_samples": len(self.X_val)
            })
        
        if self.config.norm_type == "standardization":
            if self.input_mean is not None:
                summary.update({
                    "input_mean_range": [float(self.input_mean.min()), float(self.input_mean.max())],
                    "input_std_range": [float(self.input_std.min()), float(self.input_std.max())],
                    "target_mean_range": [float(self.target_mean.min()), float(self.target_mean.max())],
                    "target_std_range": [float(self.target_std.min()), float(self.target_std.max())]
                })
        elif self.config.norm_type == "min_max":
            if self.input_min is not None:
                summary.update({
                    "input_min_range": [float(self.input_min.min()), float(self.input_min.max())],
                    "input_max_range": [float(self.input_max.min()), float(self.input_max.max())],
                    "target_min_range": [float(self.target_min.min()), float(self.target_min.max())],
                    "target_max_range": [float(self.target_max.min()), float(self.target_max.max())]
                })
        
        return summary
    
    def has_normalization_params(self) -> bool:
        """
        Check if normalization parameters have been computed
        
        Returns:
            True if normalization parameters are available, False otherwise
        """
        if self.config.norm_type == "standardization":
            return self.input_mean is not None
        elif self.config.norm_type == "min_max":
            return self.input_min is not None
        else:
            return False
    
    def print_summary(self):
        """Print data summary"""
        summary = self.get_data_summary()
        
        if "error" in summary:
            print(f"{summary['error']}")
            return
        
        print("\nData Summary:")
        print(f"   Total samples: {summary['total_samples']:,}")
        print(f"   Input dimension: {summary['input_dim']}")
        print(f"   Output dimension: {summary['output_dim']}")
        print(f"   Features shape: {summary['features_shape']}")
        print(f"   Targets shape: {summary['targets_shape']}")
        print(f"   Data split: {summary['is_split']}")
        print(f"   Normalization computed: {summary['has_norm_params']}")
        
        if summary['is_split']:
            print(f"   Training samples: {summary['train_samples']:,}")
            print(f"   Validation samples: {summary['val_samples']:,}")
        
        if self.config.norm_type == "standardization":
            if summary['has_norm_params']:
                print(f"   Input mean range: {summary['input_mean_range']}")
                print(f"   Input std range: {summary['input_std_range']}")
                print(f"   Target mean range: {summary['target_mean_range']}")
                print(f"   Target std range: {summary['target_std_range']}")
        elif self.config.norm_type == "min_max":
            if summary['has_norm_params']:
                print(f"   Input min range: {summary['input_min_range']}")
                print(f"   Input max range: {summary['input_max_range']}")
                print(f"   Target min range: {summary['target_min_range']}")
                print(f"   Target max range: {summary['target_max_range']}")


# Example usage function
def example_usage():
    """Example of how to use DataManager with different normalization types"""
    
    # Example 1: Standardization
    print("=== Example 1: Standardization ===")
    config_std = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="exp_decay",
        mix_norm=True
    )
    
    data_manager_std = DataManager(config_std)
    
    # Example CSV files (replace with actual paths)
    csv_files = [
        "../train_data/training_poisson_data_RPM500.csv",
        "../train_data/training_poisson_data_RPM1000.csv"
    ]
    
    try:
        # Read data
        features, targets = data_manager_std.read_training_csv(csv_files)
        
        # Print summary
        data_manager_std.print_summary()
        
        # Split data
        X_train, X_val, y_train, y_val = data_manager_std.split_data(test_size=0.2)
        
        # Compute normalization parameters
        input_mean, input_std, target_mean, target_std = data_manager_std.compute_normalization_params()
        
        # Get normalization info
        norm_info = data_manager_std.get_normalization_info()
        print(f"\nNormalization info type: {type(norm_info).__name__}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Min-Max Normalization
    print("\n=== Example 2: Min-Max Normalization ===")
    config_minmax = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="min_max",
        activation="exp_decay",
        mix_norm=True
    )
    
    data_manager_minmax = DataManager(config_minmax)
    
    try:
        # Read data
        features, targets = data_manager_minmax.read_training_csv(csv_files)
        
        # Print summary
        data_manager_minmax.print_summary()
        
        # Split data
        X_train, X_val, y_train, y_val = data_manager_minmax.split_data(test_size=0.2)
        
        # Compute normalization parameters
        input_min, input_max, target_min, target_max = data_manager_minmax.compute_normalization_params()
        
        # Get normalization info
        norm_info = data_manager_minmax.get_normalization_info()
        print(f"\nNormalization info type: {type(norm_info).__name__}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage() 