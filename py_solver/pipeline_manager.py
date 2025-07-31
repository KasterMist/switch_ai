"""
Pipeline Manager Module
Provides a complete pipeline for DNN model training and evaluation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

from .config import model_config
from .data_manager import DataManager
from .model_factory import create_model
from .model_manager import ModelManager
from .inference import InferenceEngine
from .dnn import FlexibleDNN


class PipelineManager:
    """
    Complete pipeline manager for DNN model training and evaluation
    """
    
    def __init__(self, config: model_config, device: str = 'auto', model_save_dir: str = "models"):
        """
        Initialize PipelineManager
        
        Args:
            config: Model configuration
            device: Device to use ('cpu', 'cuda', or 'auto')
            model_save_dir: Directory for saving models
        """
        self.config = config
        self.device = self._setup_device(device)
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(model_save_dir)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.training_history: List[Dict[str, Any]] = []
        
        print(f"PipelineManager initialized with device: {self.device}")
        print(f"Model config: {config}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for training"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def _normalize_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize only target values using the computed normalization parameters
        
        Args:
            y: Target values
            
        Returns:
            Normalized target values
        """
        if not self.config.mix_norm:
            return y
        
        if self.config.norm_type == "standardization":
            # Standardize target using config.norm_info
            if hasattr(self.config.norm_info, 'target_mean') and self.config.norm_info.target_mean is not None:
                y_norm = (y - self.config.norm_info.target_mean) / self.config.norm_info.target_std
            else:
                # Fallback to data_manager if config doesn't have the info
                if hasattr(self.data_manager, 'target_mean') and self.data_manager.target_mean is not None:
                    y_norm = (y - self.data_manager.target_mean) / self.data_manager.target_std
                else:
                    # If no normalization info available, return original data
                    print("Warning: No normalization parameters available, returning original data")
                    return y
        elif self.config.norm_type == "min_max":
            # Min-max normalize target using config.norm_info
            if hasattr(self.config.norm_info, 'target_min') and self.config.norm_info.target_min is not None:
                y_norm = (y - self.config.norm_info.target_min) / (self.config.norm_info.target_max - self.config.norm_info.target_min)
            else:
                # Fallback to data_manager if config doesn't have the info
                if hasattr(self.data_manager, 'target_min') and self.data_manager.target_min is not None:
                    y_norm = (y - self.data_manager.target_min) / (self.data_manager.target_max - self.data_manager.target_min)
                else:
                    # If no normalization info available, return original data
                    print("Warning: No normalization parameters available, returning original data")
                    return y
        else:
            return y
        
        return y_norm
    
    def load_data(self, csv_files: Union[str, List[str]], test_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Load and prepare data for training
        
        Args:
            csv_files: CSV file path(s) or pattern
            test_size: Proportion of data for validation
            
        Returns:
            Dictionary containing training and validation data
        """
        print(f"\n{'='*50}")
        print("Loading Data")
        print(f"{'='*50}")
        
        # Load data
        features, targets = self.data_manager.read_training_csv(csv_files)
        
        # Print data summary
        self.data_manager.print_summary()
        
        # Split data
        X_train, X_val, y_train, y_val = self.data_manager.split_data(test_size=test_size)
        
        # Compute normalization parameters
        norm_params = self.data_manager.compute_normalization_params()
        
        # Update config with normalization info
        self.config.norm_info = self.data_manager.get_normalization_info()
        
        # Only normalize target values (y), not input features (X)
        # Input features will be normalized inside the model
        y_train_norm = self._normalize_targets(y_train)
        y_val_norm = self._normalize_targets(y_val)

        print(f"Normalization info: {self.config.norm_info}")
        print(f"Data loading completed successfully!")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Target values normalized for training")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train_norm,
            'y_val': y_val_norm
        }
    
    def create_model(self):
        """
        Create and setup the model based on config.model_type
        
        Returns:
            Created model instance
        """
        print(f"\n{'='*50}")
        print("Creating Model")
        print(f"{'='*50}")
        
        # Create model using factory
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        print(f"Model created successfully!")
        print(f"Model type: {self.config.model_type.value}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        return self.model
    
    def train_model(self, data: Dict[str, np.ndarray], epochs: int = 100, 
                   batch_size: int = 32, early_stopping_patience: int = 20) -> Dict[str, Any]:
        """
        Train the model without saving
        
        Args:
            data: Dictionary containing training and validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*50}")
        print("Training Model")
        print(f"{'='*50}")
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_model_metadata = None
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                # Model output is in original range, but we need normalized output for loss calculation
                # since targets are also normalized
                if self.config.mix_norm:
                    outputs = self.model._normalize_output(outputs)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(X_train) // batch_size + 1)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                # Model output is in original range, but we need normalized output for loss calculation
                # since targets are also normalized
                if self.config.mix_norm:
                    val_outputs = self.model._normalize_output(val_outputs)

                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                best_model_metadata = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
            else:
                patience_counter += 1
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Best Val Loss: {best_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs_trained': len(train_losses),
            'best_model_metadata': best_model_metadata
        }
        
        return self.training_history
    
    def train_and_save(self, data: Dict[str, np.ndarray], epochs: int = 100, 
                      batch_size: int = 32, early_stopping_patience: int = 20,
                      model_name: str = "trained_model", save_onnx: bool = True,
                      is_offline: bool = True, save_dir: str = None) -> Dict[str, Any]:
        """
        Train the model and save it
        
        Args:
            data: Dictionary containing training and validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            model_name: Name for the saved model
            save_onnx: Whether to save ONNX format
            is_offline: Whether this is offline training (affects save path)
            save_dir: Directory to save the model (optional, overrides default)
            
        Returns:
            Training history dictionary
        """
        
        if self.config.norm_type == "standardization":
            assert self.config.norm_info.input_mean is not None
            assert self.config.norm_info.input_std is not None
            assert self.config.norm_info.target_mean is not None
            assert self.config.norm_info.target_std is not None
        elif self.config.norm_type == "min_max":
            assert self.config.norm_info.input_min is not None
            assert self.config.norm_info.input_max is not None
            assert self.config.norm_info.target_min is not None
            assert self.config.norm_info.target_max is not None
        else:
            raise NotImplementedError("Norm type not implemented")
        
        # Train the model
        training_history = self.train_model(data, epochs, batch_size, early_stopping_patience)
        
        # Prepare training metadata
        training_metadata = {
            'training_history': {
                'train_losses': training_history['train_losses'],
                'val_losses': training_history['val_losses'],
                'best_val_loss': training_history['best_val_loss'],
                'training_time': training_history['training_time'],
                'epochs_trained': training_history['epochs_trained'],
                'early_stopping_epoch': training_history['best_model_metadata']['epoch'] if training_history['best_model_metadata'] else None
            }
        }
        
        if training_history['best_model_metadata']:
            training_metadata.update(training_history['best_model_metadata'])
        
        # Create a temporary ModelManager with the specified save_dir if provided
        if save_dir is not None:
            from py_solver.model_manager import ModelManager
            temp_model_manager = ModelManager(save_dir)
            save_result = temp_model_manager.save_model(
                self.model, model_name, self.config, training_metadata, 
                save_onnx=save_onnx, is_offline=is_offline
            )
        else:
            # Use the default ModelManager
            save_result = self.model_manager.save_model(
                self.model, model_name, self.config, training_metadata, 
                save_onnx=save_onnx, is_offline=is_offline
            )
        
        print(f"Best model saved using ModelManager: {save_result['version']}")
        
        return training_history
    
    def evaluate_model(self, test_data: np.ndarray, test_targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test input data
            test_targets: Test target values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*50}")
        print("Evaluating Model")
        print(f"{'='*50}")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert to tensors
        test_data_tensor = torch.FloatTensor(test_data).to(self.device)
        test_targets_tensor = torch.FloatTensor(test_targets).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(test_data_tensor)
        
        # Convert to numpy for metric calculation
        predictions_np = predictions.cpu().numpy()
        test_targets_np = test_targets_tensor.cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((predictions_np - test_targets_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_np - test_targets_np))
        
        # Calculate R-squared
        ss_res = np.sum((test_targets_np - predictions_np) ** 2)
        ss_tot = np.sum((test_targets_np - np.mean(test_targets_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"Evaluation Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
        
        return metrics
    

    
    def train_and_save_online(self, inputs: np.ndarray, outputs: np.ndarray,
                    epochs: int = 100, batch_size: int = 32,
                    test_size: float = 0.2, save_dir: str = "pipeline_output",
                    model_name: str = "trained_model") -> Dict[str, Any]:
        """
        Train model online with pre-loaded data
        
        Args:
            inputs: Input features (numpy array)
            outputs: Target values (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size for training
            test_size: Validation split ratio
            save_dir: Directory to save all outputs
            model_name: Name for the saved model
            
        Returns:
            Dictionary containing training results
        """
        print(f"\n{'='*60}")
        print("Running Online Training Pipeline")
        print(f"{'='*60}")
        
        # Step 1: Prepare data
        data = self._prepare_online_data(inputs, outputs, test_size=test_size)
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Train and save model (online training - no timestamp suffix)
        training_history = self.train_and_save(
            data, epochs=epochs, batch_size=batch_size,
            model_name=model_name, is_offline=False, save_dir=save_dir
        )
        
        print(f"\n{'='*60}")
        print("Online Training Pipeline Completed Successfully!")
        print(f"{'='*60}")
        
        return {
            'training_history': training_history,
            'save_dir': save_dir,
            'model_name': model_name
        }
    
    def _prepare_online_data(self, inputs: np.ndarray, outputs: np.ndarray, 
                           test_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepare data for online training
        
        Args:
            inputs: Input features (numpy array)
            outputs: Target values (numpy array)
            test_size: Proportion of data for validation
            
        Returns:
            Dictionary containing training and validation data
        """
        print(f"\n{'='*50}")
        print("Preparing Online Data")
        print(f"{'='*50}")
        
        # Ensure inputs and outputs are numpy arrays
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        if not isinstance(outputs, np.ndarray):
            outputs = np.array(outputs)
        
        # Ensure 2D arrays
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)
        
        print(f"Input data shape: {inputs.shape}")
        print(f"Output data shape: {outputs.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            inputs, outputs, test_size=test_size, random_state=42
        )
        
        # Compute normalization parameters for targets
        if self.config.mix_norm:
            if self.config.norm_type == "standardization":
                input_mean = np.mean(inputs)
                input_std = np.std(inputs)
                target_mean = np.mean(outputs)
                target_std = np.std(outputs)
                # Create StandardizationInfo object
                from py_solver.config import StandardizationInfo
                self.config.norm_info = StandardizationInfo(
                    input_mean=float(input_mean),
                    input_std=float(input_std),
                    target_mean=float(target_mean),
                    target_std=float(target_std)
                )
            elif self.config.norm_type == "min_max":
                input_min = np.min(inputs)
                input_max = np.max(inputs)
                target_min = np.min(outputs)
                target_max = np.max(outputs)
                # Create MinMaxNormInfo object
                from py_solver.config import MinMaxNormInfo
                self.config.norm_info = MinMaxNormInfo(
                    input_min=float(input_min),
                    input_max=float(input_max),
                    target_min=float(target_min),
                    target_max=float(target_max)
                )
        else:
            raise NotImplementedError("Mix norm setting false logic is not implemented")
        
        # Normalize target values
        y_train_norm = self._normalize_targets(y_train)
        y_val_norm = self._normalize_targets(y_val)
        
        print(f"Normalization info: {self.config.norm_info}")
        print(f"Data preparation completed successfully!")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Target values normalized for training")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train_norm,
            'y_val': y_val_norm
        }
    
    def train_and_save_offline(self, csv_files: Union[str, List[str]], 
                             epochs: int = 100, batch_size: int = 32,
                             test_size: float = 0.2, save_dir: str = "pipeline_output",
                             model_name: str = "trained_model") -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to model training
        
        Args:
            csv_files: Training CSV file(s)
            epochs: Number of training epochs
            batch_size: Batch size for training
            test_size: Validation split ratio
            save_dir: Directory to save all outputs
            model_name: Name for the saved model
            
        Returns:
            Dictionary containing all pipeline results
        """
        print(f"\n{'='*60}")
        print("Running Complete Pipeline")
        print(f"{'='*60}")
        
        # Step 1: Load data
        data = self.load_data(csv_files, test_size=test_size)

        # Step 2: Create model
        self.create_model()
        
        # Step 3: Train and save model (offline training - with timestamp suffix)
        training_history = self.train_and_save(
            data, epochs=epochs, batch_size=batch_size,
            model_name=model_name, is_offline=True, save_dir=save_dir
        )
        
        print(f"\n{'='*60}")
        print("Pipeline Completed Successfully!")
        print(f"{'='*60}")
        
        return {
            'training_history': training_history,
            'save_dir': save_dir,
            'model_name': model_name
        }
    
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Make batch predictions using the trained model
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data_tensor)
        
        return predictions.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"error": "No model created"}
        
        info = {
            'model_type': type(self.model).__name__,
            'device': str(next(self.model.parameters()).device),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'norm_type': self.config.norm_type,
            'activation': self.config.activation
        }
        
        return info


# Example usage functions
def example_pipeline():
    """Example of how to use the complete pipeline"""
    
    # Create configuration
    config = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="relu",
        mix_norm=True
    )
    
    # Create pipeline manager
    pipeline = PipelineManager(config)
    
    # Example CSV files (replace with actual paths)
    training_csv_files = [
        "train_data/training_poisson_data_RPM500.csv",
        "train_data/training_poisson_data_RPM1000.csv"
    ]
    
    try:
        # Run complete pipeline
        results = pipeline.train_and_save_offline(
            csv_files=training_csv_files,
            epochs=50,
            batch_size=32,
            save_dir="pipeline_output",
            model_name="example_model"
        )
        
        print(f"\nPipeline Results Summary:")
        print(f"Best validation loss: {results['training_history']['best_val_loss']:.6f}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def example_online_pipeline():
    """Example of how to use the online training pipeline"""
    
    # Create configuration
    config = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="relu",
        mix_norm=True
    )
    
    # Create pipeline manager
    pipeline = PipelineManager(config)
    
    # Generate example data (replace with your actual data)
    np.random.seed(42)
    n_samples = 1000
    inputs = np.random.randn(n_samples, 5)  # 5 input features
    outputs = np.random.randn(n_samples, 1)  # 1 output target
    
    try:
        # Run online training pipeline
        results = pipeline.train_and_save_online(
            inputs=inputs,
            outputs=outputs,
            epochs=50,
            batch_size=32,
            save_dir="pipeline_output",
            model_name="online_example_model"
        )
        
        print(f"\nOnline Pipeline Results Summary:")
        print(f"Best validation loss: {results['training_history']['best_val_loss']:.6f}")
        print(f"Model saved as: {results['model_name']}")
        
    except Exception as e:
        print(f"Online pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running offline pipeline example...")
    example_pipeline()
    
    print("\n" + "="*60)
    print("Running online pipeline example...")
    example_online_pipeline()