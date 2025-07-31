"""
Inference Module
Provides functionality for model inference, prediction services, and ONNX inference
"""

import os
import numpy as np
import torch
import pandas as pd
import onnxruntime as ort
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

from .config import model_config, ModelType
from .dnn import FlexibleDNN
from .model_manager import ModelManager
from .model_factory import create_model


class InferenceEngine:
    """
    Enhanced inference engine for making predictions with trained models
    Supports both PyTorch and ONNX models with automatic format detection
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize InferenceEngine
        
        Args:
            model_path: Optional path to model file (.pt or .onnx) for immediate loading
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.model = None
        self.onnx_session = None
        self.model_metadata = None
        self.model_config = None
        self.model_type = None  # 'pytorch' or 'onnx'
        self.norm_info = None
        
        # Load model if path provided
        if model_path:
            self.load_model_from_path(model_path)
        
        print(f"InferenceEngine initialized with device: {self.device}")
        if model_path:
            print(f"Model type: {self.model_type}")
            print(f"Model path: {model_path}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def load_model_from_path(self, model_path: str):
        """
        Load model from file path (supports both .pt and .onnx)
        
        Args:
            model_path: Path to the model file (.pt or .onnx)
        """
        model_path_obj = Path(model_path)
        
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_path_obj.suffix.lower() == '.pt':
            self._load_pytorch_model_from_path(model_path_obj)
        elif model_path_obj.suffix.lower() == '.onnx':
            self._load_onnx_model_from_path(model_path_obj)
        else:
            raise ValueError(f"Unsupported model format: {model_path_obj.suffix}. Supported: .pt, .onnx")
    
    def _load_pytorch_model_from_path(self, model_path: Path):
        """Load PyTorch model from file path"""
        self.model_type = 'pytorch'
        
        # Load model metadata and config
        model_dir = model_path.parent
        config_path = model_dir / "config.json"
        
        if config_path.exists():
            # Load config from saved metadata
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract config from nested structure
            if 'config' in config_data:
                config_dict = config_data['config']
            else:
                config_dict = config_data
            
            # Reconstruct config
            self.model_config = model_config(
                input_dim=config_dict['input_dim'],
                output_dim=config_dict['output_dim'],
                base_neurons=config_dict.get('base_neurons', 16),
                dropout_prob=config_dict.get('dropout_prob', 0.025),
                model_type=ModelType(config_dict.get('model_type', 'flexible_dnn')),
                norm_type=config_dict.get('norm_type', 'standardization'),
                activation=config_dict.get('activation', 'relu'),
                mix_norm=config_dict.get('mix_norm', True)
            )
            
            # Load normalization info if available
            norm_info_path = model_dir / "norm_info.json"
            if norm_info_path.exists():
                with open(norm_info_path, 'r') as f:
                    norm_data = json.load(f)
                self.norm_info = norm_data
        else:
            # Fallback: create basic config
            print("Warning: No config.json found. Using default config.")
            self.model_config = model_config(
                input_dim=5,  # Default, should be updated
                output_dim=1,  # Default, should be updated
                model_type=ModelType.FLEXIBLE_DNN
            )
        
        # Create model and load weights
        self.model = create_model(self.model_config)
        
        # Load the saved model file
        saved_data = torch.load(model_path, map_location=self.device)
        
        # Check if the saved data contains model_state_dict or is directly the state_dict
        if isinstance(saved_data, dict) and 'model_state_dict' in saved_data:
            # Load from the nested structure
            self.model.load_state_dict(saved_data['model_state_dict'])
        else:
            # Load directly as state_dict
            self.model.load_state_dict(saved_data)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"PyTorch model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_onnx_model_from_path(self, model_path: Path):
        """Load ONNX model from file path"""
        self.model_type = 'onnx'
        
        # Load ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Get model metadata
        input_meta = self.onnx_session.get_inputs()[0]
        output_meta = self.onnx_session.get_outputs()[0]
        
        # Create basic config from ONNX model info
        self.model_config = model_config(
            input_dim=input_meta.shape[1] if len(input_meta.shape) > 1 else 1,
            output_dim=output_meta.shape[1] if len(output_meta.shape) > 1 else 1,
            model_type=ModelType.FLEXIBLE_DNN
        )
        
        # Try to load config from metadata directory
        model_dir = model_path.parent
        config_path = model_dir / "config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract config from nested structure
            if 'config' in config_data:
                config_dict = config_data['config']
            else:
                config_dict = config_data
            
            # Update config with saved data
            self.model_config = model_config(
                input_dim=config_dict['input_dim'],
                output_dim=config_dict['output_dim'],
                base_neurons=config_dict.get('base_neurons', 16),
                dropout_prob=config_dict.get('dropout_prob', 0.025),
                model_type=ModelType(config_dict.get('model_type', 'flexible_dnn')),
                norm_type=config_dict.get('norm_type', 'standardization'),
                activation=config_dict.get('activation', 'relu'),
                mix_norm=config_dict.get('mix_norm', True)
            )
            
            # Load normalization info if available
            norm_info_path = model_dir / "norm_info.json"
            if norm_info_path.exists():
                with open(norm_info_path, 'r') as f:
                    norm_data = json.load(f)
                self.norm_info = norm_data
        
        print(f"ONNX model loaded successfully")
        print(f"Input shape: {input_meta.shape}")
        print(f"Output shape: {output_meta.shape}")
    
    def load_pytorch_model(self, model: FlexibleDNN, config: model_config):
        """
        Load a PyTorch model for inference
        
        Args:
            model: The PyTorch model
            config: Model configuration
        """
        self.model = model.to(self.device)
        self.model.eval()
        self.model_config = config
        
        print(f"PyTorch model loaded successfully")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Input dimension: {config.input_dim}")
        print(f"Output dimension: {config.output_dim}")
    
    def load_model_from_manager(self, version: str, base_save_dir: str = "models"):
        """
        Load a model from ModelManager
        
        Args:
            version: Model version to load
            base_save_dir: Base directory for model storage
        """
        manager = ModelManager(base_save_dir)
        self.model, self.model_metadata = manager.load_model(version, self.device)
        
        # Extract config from metadata
        if self.model_metadata and 'config' in self.model_metadata:
            self.model_config = self.model_metadata['config']
        
        print(f"Model loaded from manager: {version}")
    
    def load_onnx_model(self, onnx_path: str):
        """
        Load an ONNX model for inference
        
        Args:
            onnx_path: Path to the ONNX model file
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        try:
            # Create inference session
            self.onnx_session = ort.InferenceSession(onnx_path)
            
            # Get model info
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            input_shape = self.onnx_session.get_inputs()[0].shape
            output_shape = self.onnx_session.get_outputs()[0].shape
            
            print(f"ONNX model loaded successfully!")
            print(f"Input name: {input_name}, shape: {input_shape}")
            print(f"Output name: {output_name}, shape: {output_shape}")
            
            # Store input name for later use
            self.onnx_input_name = input_name
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def predict_pytorch(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions using PyTorch model
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No PyTorch model loaded. Call load_pytorch_model() first.")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Validate input dimensions
        if self.model_config and data.shape[1] != self.model_config.input_dim:
            raise ValueError(f"Expected input dimension {self.model_config.input_dim}, "
                           f"but got {data.shape[1]}")
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(data_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_onnx(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions using ONNX model
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predictions
        """
        if self.onnx_session is None:
            raise ValueError("No ONNX model loaded. Call load_onnx_model() first.")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Get input name
        input_name = self.onnx_session.get_inputs()[0].name
        
        # Make prediction
        predictions = self.onnx_session.run(None, {input_name: data.astype(np.float32)})[0]
        
        return predictions
    
    def predict(self, data: Union[np.ndarray, List, pd.DataFrame], use_onnx: bool = False) -> np.ndarray:
        """
        Make predictions using the loaded model
        
        Args:
            data: Input data (numpy array, list, or pandas DataFrame)
            use_onnx: Whether to use ONNX model (if available)
            
        Returns:
            Predictions
        """
        # Convert input to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Validate input dimensions
        if self.model_config and data.shape[1] != self.model_config.input_dim:
            raise ValueError(f"Expected input dimension {self.model_config.input_dim}, got {data.shape[1]}")
        
        # Determine which model to use
        if use_onnx and self.onnx_session is not None:
            return self.predict_onnx(data)
        elif self.model is not None:
            return self.predict_pytorch(data)
        elif self.onnx_session is not None:
            # Use ONNX if available and no specific preference
            return self.predict_onnx(data)
        else:
            raise ValueError("No model loaded. Call load_pytorch_model() or load_onnx_model() first.")
    
    def batch_predict(self, data: np.ndarray, batch_size: int = 32, 
                     use_onnx: bool = False) -> np.ndarray:
        """
        Make batch predictions
        
        Args:
            data: Input data for prediction
            batch_size: Batch size for processing
            use_onnx: Whether to use ONNX model (if available)
            
        Returns:
            Predictions
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_samples = len(data)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch_data = data[i:i+batch_size]
            batch_predictions = self.predict(batch_data, use_onnx)
            predictions.append(batch_predictions)
        
        return np.vstack(predictions)
    
    def predict_with_metadata(self, data: np.ndarray, use_onnx: bool = False) -> Dict[str, Any]:
        """
        Make predictions with additional metadata
        
        Args:
            data: Input data for prediction
            use_onnx: Whether to use ONNX model (if available)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        import time
        
        start_time = time.time()
        predictions = self.predict(data, use_onnx)
        inference_time = time.time() - start_time
        
        result = {
            'predictions': predictions,
            'inference_time': inference_time,
            'input_shape': data.shape,
            'output_shape': predictions.shape,
            'model_type': 'onnx' if use_onnx and self.onnx_session else 'pytorch',
            'device': self.device
        }
        
        # Add model metadata if available
        if self.model_metadata:
            result['model_metadata'] = self.model_metadata
        
        return result
    
    def predict_csv(self, csv_path: str, input_columns: Optional[List[str]] = None, use_onnx: bool = False) -> np.ndarray:
        """
        Make predictions on data from CSV file
        
        Args:
            csv_path: Path to CSV file
            input_columns: List of column names to use as input (if None, use all columns)
            use_onnx: Whether to use ONNX model (if available)
            
        Returns:
            Predictions as numpy array
        """
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Select input columns
        if input_columns is None:
            # Use all columns if no specific columns specified
            input_data = df.values
        else:
            # Use specified columns
            if not all(col in df.columns for col in input_columns):
                raise ValueError(f"Some specified columns not found in CSV: {input_columns}")
            input_data = df[input_columns].values
        
        # Make predictions
        return self.predict(input_data, use_onnx)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': self.device,
            'input_dim': self.model_config.input_dim if self.model_config else None,
            'output_dim': self.model_config.output_dim if self.model_config else None,
            'norm_type': self.model_config.norm_type if self.model_config else None,
            'mix_norm': self.model_config.mix_norm if self.model_config else None
        }
        
        if self.model_type == 'pytorch' and self.model is not None:
            info.update({
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            })
        elif self.model_type == 'onnx' and self.onnx_session is not None:
            input_meta = self.onnx_session.get_inputs()[0]
            output_meta = self.onnx_session.get_outputs()[0]
            info.update({
                'input_shape': input_meta.shape,
                'output_shape': output_meta.shape,
                'input_name': input_meta.name,
                'output_name': output_meta.name
            })
        
        return info


class PredictionService:
    """
    High-level prediction service for easy model deployment
    """
    
    def __init__(self, model_path: str, model_type: str = 'auto', device: str = 'auto'):
        """
        Initialize PredictionService
        
        Args:
            model_path: Path to model file or model version
            model_type: Type of model ('pytorch', 'onnx', 'auto')
            device: Device to use
        """
        self.engine = InferenceEngine(device)
        self.model_path = model_path
        self.model_type = model_type
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on path and type"""
        if self.model_type == 'auto':
            # Auto-detect model type
            if self.model_path.endswith('.onnx'):
                self.model_type = 'onnx'
            elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.model_type = 'pytorch'
            else:
                # Assume it's a model version from ModelManager
                self.model_type = 'manager'
        
        if self.model_type == 'onnx':
            self.engine.load_onnx_model(self.model_path)
        elif self.model_type == 'pytorch':
            # For PyTorch models, we need additional config info
            raise NotImplementedError("Direct PyTorch model loading not implemented. "
                                   "Use ModelManager to load PyTorch models.")
        elif self.model_type == 'manager':
            self.engine.load_model_from_manager(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        use_onnx = self.model_type == 'onnx'
        return self.engine.predict(data, use_onnx)
    
    def predict_with_metadata(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions with metadata
        
        Args:
            data: Input data
            
        Returns:
            Dictionary containing predictions and metadata
        """
        use_onnx = self.model_type == 'onnx'
        return self.engine.predict_with_metadata(data, use_onnx)


# Convenience functions
def create_inference_engine(model_path: str, model_type: str = 'auto', 
                          device: str = 'auto') -> InferenceEngine:
    """
    Convenience function to create an inference engine
    
    Args:
        model_path: Path to model file or model version
        model_type: Type of model ('pytorch', 'onnx', 'auto')
        device: Device to use
        
    Returns:
        Configured InferenceEngine
    """
    service = PredictionService(model_path, model_type, device)
    return service.engine


def predict_single(model_path: str, data: np.ndarray, model_type: str = 'auto',
                  device: str = 'auto') -> np.ndarray:
    """
    Convenience function for single prediction
    
    Args:
        model_path: Path to model file or model version
        data: Input data
        model_type: Type of model ('pytorch', 'onnx', 'auto')
        device: Device to use
        
    Returns:
        Predictions
    """
    service = PredictionService(model_path, model_type, device)
    return service.predict(data) 