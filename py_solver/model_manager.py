"""
Model Manager Module
Provides functionality for model saving, loading, version management, and configuration management
"""

import os
import torch
import torch.onnx
import json
import time
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path

from .config import model_config, StandardizationInfo, MinMaxNormInfo, ModelType
from .dnn import FlexibleDNN
from .model_factory import create_model


class ModelManager:
    """
    Model manager for handling model saving, loading, and version management
    """
    
    def __init__(self, base_save_dir: str = "models"):
        """
        Initialize ModelManager
        
        Args:
            base_save_dir: Base directory for saving models
        """
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(exist_ok=True)
        
        # Model registry for tracking saved models
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from file"""
        registry_path = self.base_save_dir / "model_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load model registry: {e}")
                self.model_registry = {}
    
    def _save_registry(self, save_dir: Optional[Path] = None):
        """Save model registry to file"""
        if save_dir is not None:
            # Save registry to the specific model directory
            registry_path = save_dir / "model_registry.json"
        else:
            # Save registry to the base directory (for backward compatibility)
            registry_path = self.base_save_dir / "model_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save model registry: {e}")
    
    def save_model(self, model: FlexibleDNN, model_name: str, 
                  config: model_config, metadata: Optional[Dict[str, Any]] = None,
                  save_onnx: bool = True, is_offline: bool = True) -> Dict[str, str]:
        """
        Save model with comprehensive metadata
        
        Args:
            model: The model to save
            model_name: Name for the model
            config: Model configuration
            metadata: Additional metadata (training metrics, etc.)
            save_onnx: Whether to also save ONNX format
            is_offline: Whether this is offline training (affects path naming)
            
        Returns:
            Dictionary containing paths to saved files
        """
        # Create version name based on training type
        if is_offline:
            # Offline training: add timestamp suffix
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            version_name = f"{model_name}_{timestamp}"
        else:
            # Online training: use model_name directly
            version_name = model_name
        
        # Create save directory with dynamic path based on is_offline
        if is_offline:
            # For offline training: add timestamp suffix to the last path component
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_path = Path(self.base_save_dir)
            if base_path.name:  # If base_path has a name component
                new_name = f"{base_path.name}_{timestamp}"
                save_dir = base_path.parent / new_name
            else:
                save_dir = base_path / timestamp
        else:
            # For online training: use original path
            save_dir = self.base_save_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert norm_info to serializable format
        norm_info_data = None
        if config.norm_info is not None:
            if config.norm_type == "standardization":
                norm_info_data = {
                    'type': 'standardization',
                    'input_mean': config.norm_info.input_mean,
                    'input_std': config.norm_info.input_std,
                    'target_mean': float(config.norm_info.target_mean) if config.norm_info.target_mean is not None else None,
                    'target_std': float(config.norm_info.target_std) if config.norm_info.target_std is not None else None
                }
            elif config.norm_type == "min_max":
                norm_info_data = {
                    'type': 'min_max',
                    'input_min': config.norm_info.input_min,
                    'input_max': config.norm_info.input_max,
                    'target_min': float(config.norm_info.target_min) if config.norm_info.target_min is not None else None,
                    'target_max': float(config.norm_info.target_max) if config.norm_info.target_max is not None else None
                }
        
        model_metadata = {
            'model_name': model_name,
            'version': version_name,
            'timestamp': timestamp,
            'training_type': 'offline' if is_offline else 'online',
            'config': {
                'input_dim': config.input_dim,
                'output_dim': config.output_dim,
                'base_neurons': config.base_neurons,
                'dropout_prob': config.dropout_prob,
                'model_type': config.model_type.value,
                'norm_type': config.norm_type,
                'norm_info': norm_info_data,
                'activation': config.activation,
                'mix_norm': config.mix_norm
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            }
        }
        
        # Add custom metadata
        if metadata:
            model_metadata['custom_metadata'] = metadata
        
        # Save PyTorch model
        pt_path = save_dir / f"{model_name}.pt"
        self._save_pytorch_model(model, pt_path, model_metadata)
        
        # Save configuration
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save ONNX model if requested
        onnx_path = None
        if save_onnx:
            onnx_path = save_dir / f"{model_name}.onnx"
            self._save_onnx_model(model, onnx_path, config.input_dim)
        
        # Update registry
        self.model_registry[version_name] = {
            'model_name': model_name,
            'timestamp': timestamp,
            'pt_path': str(pt_path),
            'config_path': str(config_path),
            'onnx_path': str(onnx_path) if onnx_path else None,
            'metadata': model_metadata
        }
        # Save registry to the specific model directory
        self._save_registry(save_dir)
        
        print(f"Model saved successfully:")
        print(f"  Version: {version_name}")
        print(f"  PyTorch model: {pt_path}")
        print(f"  Configuration: {config_path}")
        if onnx_path:
            print(f"  ONNX model: {onnx_path}")
        
        return {
            'version': version_name,
            'pt_path': str(pt_path),
            'config_path': str(config_path),
            'onnx_path': str(onnx_path) if onnx_path else None
        }
    
    def _save_pytorch_model(self, model: FlexibleDNN, save_path: Path, metadata: Dict[str, Any]):
        """Save PyTorch model with metadata"""
        # Prepare the norm_info data
        norm_info_data = None
        if hasattr(model, 'norm_info') and model.norm_info is not None:
            if model.norm_type == "standardization":
                norm_info_data = {
                    'type': 'standardization',
                    'input_mean': model.norm_info.input_mean,
                    'input_std': model.norm_info.input_std,
                    'target_mean': float(model.norm_info.target_mean) if model.norm_info.target_mean is not None else None,
                    'target_std': float(model.norm_info.target_std) if model.norm_info.target_std is not None else None
                }
            elif model.norm_type == "min_max":
                norm_info_data = {
                    'type': 'min_max',
                    'input_min': model.norm_info.input_min,
                    'input_max': model.norm_info.input_max,
                    'target_min': float(model.norm_info.target_min) if model.norm_info.target_min is not None else None,
                    'target_max': float(model.norm_info.target_max) if model.norm_info.target_max is not None else None
                }
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': metadata['config'],
            'norm_info_data': norm_info_data,
            'metadata': metadata
        }, save_path)
    
    def _save_onnx_model(self, model: FlexibleDNN, save_path: Path, input_dim: int):
        """Save ONNX model"""
        model.eval()
        
        # Get the device of the model
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, input_dim, device=device)
        
        torch.onnx.export(
            model,
            (dummy_input,),
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def load_model(self, version: str, device: str = 'cpu') -> Tuple[FlexibleDNN, Dict[str, Any]]:
        """
        Load model by version
        
        Args:
            version: Model version to load
            device: Device to load model on
            
        Returns:
            Tuple of (model, metadata)
        """
        if version not in self.model_registry:
            raise ValueError(f"Model version '{version}' not found in registry")
        
        model_info = self.model_registry[version]
        pt_path = Path(model_info['pt_path'])
        
        if not pt_path.exists():
            raise FileNotFoundError(f"Model file not found: {pt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(pt_path, map_location=device)
        saved_model_config = checkpoint['model_config']
        
        # Restore norm_info
        norm_info: Optional[Union[StandardizationInfo, MinMaxNormInfo]] = None
        if 'norm_info_data' in checkpoint and checkpoint['norm_info_data'] is not None:
            norm_info_data = checkpoint['norm_info_data']
            if norm_info_data['type'] == 'standardization':
                norm_info = StandardizationInfo(
                    input_mean=norm_info_data['input_mean'],
                    input_std=norm_info_data['input_std'],
                    target_mean=norm_info_data['target_mean'],
                    target_std=norm_info_data['target_std']
                )
            elif norm_info_data['type'] == 'min_max':
                norm_info = MinMaxNormInfo(
                    input_min=norm_info_data['input_min'],
                    input_max=norm_info_data['input_max'],
                    target_min=norm_info_data['target_min'],
                    target_max=norm_info_data['target_max']
                )
        
        # Create config object for model creation
        config = model_config(
            input_dim=saved_model_config['input_dim'],
            output_dim=saved_model_config['output_dim'],
            base_neurons=saved_model_config['base_neurons'],
            dropout_prob=saved_model_config['dropout_prob'],
            model_type=saved_model_config['model_type'],
            norm_type=saved_model_config['norm_type'],
            activation=saved_model_config['activation'],
            mix_norm=saved_model_config['mix_norm'],
            norm_info=norm_info
        )
        
        # Create model using factory
        model = create_model(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        metadata = checkpoint.get('metadata', {})
        
        print(f"Model loaded successfully:")
        print(f"  Version: {version}")
        print(f"  Device: {device}")
        print(f"  Parameters: {metadata.get('model_info', {}).get('total_parameters', 'Unknown'):,}")
        
        return model, metadata
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all saved models
        
        Returns:
            Dictionary of model versions and their metadata
        """
        return self.model_registry.copy()
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string or None if not found
        """
        versions = [v for v, info in self.model_registry.items() 
                   if info['model_name'] == model_name]
        
        if not versions:
            return None
        
        # Sort by timestamp and return latest
        versions.sort(key=lambda v: self.model_registry[v]['timestamp'], reverse=True)
        return versions[0]
    
    def delete_model(self, version: str) -> bool:
        """
        Delete a model version
        
        Args:
            version: Model version to delete
            
        Returns:
            True if successful, False otherwise
        """
        if version not in self.model_registry:
            print(f"Model version '{version}' not found")
            return False
        
        try:
            model_info = self.model_registry[version]
            
            # Delete files
            for path_key in ['pt_path', 'config_path', 'onnx_path']:
                if path_key in model_info and model_info[path_key]:
                    path = Path(model_info[path_key])
                    if path.exists():
                        path.unlink()
            
            
            # Remove from registry
            del self.model_registry[version]
            self._save_registry()
            
            print(f"Model version '{version}' deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting model version '{version}': {e}")
            return False
    
    def export_model_to_onnx(self, version: str, output_path: Optional[str] = None) -> str:
        """
        Export a saved model to ONNX format
        
        Args:
            version: Model version to export
            output_path: Output path for ONNX file (optional)
            
        Returns:
            Path to the exported ONNX file
        """
        model, metadata = self.load_model(version)
        
        if output_path is None:
            output_path = f"{version}.onnx"
        
        self._save_onnx_model(model, Path(output_path), metadata['config']['input_dim'])
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path


# Convenience functions
def save_model_with_manager(model: FlexibleDNN, model_name: str, config: model_config,
                           metadata: Optional[Dict[str, Any]] = None,
                           base_save_dir: str = "models") -> Dict[str, str]:
    """
    Convenience function to save a model using ModelManager
    
    Args:
        model: The model to save
        model_name: Name for the model
        config: Model configuration
        metadata: Additional metadata
        base_save_dir: Base directory for saving
        
    Returns:
        Dictionary containing paths to saved files
    """
    manager = ModelManager(base_save_dir)
    return manager.save_model(model, model_name, config, metadata)


def load_model_with_manager(version: str, device: str = 'cpu',
                          base_save_dir: str = "models") -> Tuple[FlexibleDNN, Dict[str, Any]]:
    """
    Convenience function to load a model using ModelManager
    
    Args:
        version: Model version to load
        device: Device to load model on
        base_save_dir: Base directory for loading
        
    Returns:
        Tuple of (model, metadata)
    """
    manager = ModelManager(base_save_dir)
    return manager.load_model(version, device) 