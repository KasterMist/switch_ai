import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import os
from typing import Dict, Any, Optional
from .config import MinMaxNormInfo, StandardizationInfo

class ExpDecayActivation(nn.Module):
    """self designed activation function: x * exp(-x^2 / (2*e))"""
    def __init__(self):
        super(ExpDecayActivation, self).__init__()
        self.e = 2.718281828459045  # euler number
    
    def forward(self, x):
        return x * torch.exp(-x.pow(2) / (2 * self.e))


class FlexibleDNN(nn.Module):
    """
    Flexible DNN model, support multiple normalization and activation functions
    """
    
    def __init__(self, input_dim=5, output_dim=1, base_neurons=16, dropout_prob=0.025,
                 norm_type="standardization", norm_info=None, activation="exp_decay", mix_norm=True):
        super(FlexibleDNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_neurons = base_neurons
        self.dropout_prob = dropout_prob
        self.norm_type = norm_type
        self.activation_type = activation
        self.mix_norm = mix_norm
        
        # Construct the network layers
        self.layer1 = nn.Linear(input_dim, 8 * base_neurons)
        self.layer2 = nn.Linear(8 * base_neurons, 4 * base_neurons)
        self.layer3 = nn.Linear(4 * base_neurons, 2 * base_neurons)
        self.output_layer = nn.Linear(2 * base_neurons, output_dim)
        
        # Set the activation function
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize the weights
        self._init_weights()
        
        # Set the normalization parameters
        self._setup_normalization(norm_info)
    
    def _get_activation(self, activation_type: str) -> nn.Module:
        """
        Return the corresponding activation function based on the activation function type
        """
        activation_map = {
            "exp_decay": ExpDecayActivation(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        
        if activation_type not in activation_map:
            raise ValueError(f"Unsupported activation function type: {activation_type}")
        
        return activation_map[activation_type]
    
    def _setup_normalization(self, norm_info):
        """
        Set the normalization parameters
        """
        if not self.mix_norm:
            return
        
        # If norm_info is None, create a default norm_info object
        if norm_info is None:
            if self.norm_type == "standardization":
                from .config import StandardizationInfo
                norm_info = StandardizationInfo()
            elif self.norm_type == "min_max":
                from .config import MinMaxNormInfo
                norm_info = MinMaxNormInfo()
        
        # Save the norm_info object for later use
        self.norm_info = norm_info
        
        if self.norm_type == "standardization":
            # Standardization parameters
            input_mean = torch.zeros(self.input_dim) if norm_info.input_mean is None else torch.tensor(norm_info.input_mean, dtype=torch.float32)
            input_std = torch.ones(self.input_dim) if norm_info.input_std is None else torch.tensor(norm_info.input_std, dtype=torch.float32)
            target_mean = torch.tensor(0.0) if norm_info.target_mean is None else torch.tensor(norm_info.target_mean, dtype=torch.float32)
            target_std = torch.tensor(1.0) if norm_info.target_std is None else torch.tensor(norm_info.target_std, dtype=torch.float32)
            
            self.register_buffer('input_mean', input_mean)
            self.register_buffer('input_std', input_std)
            self.register_buffer('target_mean', target_mean)
            self.register_buffer('target_std', target_std)
            
        elif self.norm_type == "min_max":
            # Min-max normalization parameters
            input_min = torch.zeros(self.input_dim) if norm_info.input_min is None else torch.tensor(norm_info.input_min, dtype=torch.float32)
            input_max = torch.ones(self.input_dim) if norm_info.input_max is None else torch.tensor(norm_info.input_max, dtype=torch.float32)
            target_min = torch.tensor(0.0) if norm_info.target_min is None else torch.tensor(norm_info.target_min, dtype=torch.float32)
            target_max = torch.tensor(1.0) if norm_info.target_max is None else torch.tensor(norm_info.target_max, dtype=torch.float32)
            
            self.register_buffer('input_min', input_min)
            self.register_buffer('input_max', input_max)
            self.register_buffer('target_min', target_min)
            self.register_buffer('target_max', target_max)
    
    def _init_weights(self):
        """
        Initialize the network weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _normalize_input(self, x):
        """
        Normalize the input
        """
        if not self.mix_norm:
            return x
        
        if self.norm_type == "standardization":
            return (x - self.input_mean) / self.input_std
        elif self.norm_type == "min_max":
            return (x - self.input_min) / (self.input_max - self.input_min)
        else:
            return x

    def _normalize_output(self, x):
        """
        Normalize the output
        """
        if not self.mix_norm:
            return x

        if self.norm_type == "standardization":
            return (x - self.target_mean) / self.target_std
        elif self.norm_type == "min_max":
            return (x - self.target_min) / (self.target_max - self.target_min)
        else:
            return x
    
    def _denormalize_output(self, x):
        """
        Denormalize the output
        """
        if not self.mix_norm:
            return x
        
        if self.norm_type == "standardization":
            return x * self.target_std + self.target_mean
        elif self.norm_type == "min_max":
            return x * (self.target_max - self.target_min) + self.target_min
        else:
            return x
    
    def forward(self, x):
        """
        Forward propagation
        """
        # Normalize the input
        x = self._normalize_input(x)
        
        # Network layers
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.activation(self.layer3(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        
        # Denormalize the output
        x = self._denormalize_output(x)
        
        return x
    
    def save_model(self, save_dir: str, model_name: str):
        """
        Save the model as .pt and .onnx format
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare the norm_info data
        norm_info_data = None
        if hasattr(self, 'norm_info') and self.norm_info is not None:
            if self.norm_type == "standardization":
                norm_info_data = {
                    'type': 'standardization',
                    'input_mean': self.norm_info.input_mean,
                    'input_std': self.norm_info.input_std,
                    'target_mean': self.norm_info.target_mean,
                    'target_std': self.norm_info.target_std
                }
            elif self.norm_type == "min_max":
                norm_info_data = {
                    'type': 'min_max',
                    'input_min': self.norm_info.input_min,
                    'input_max': self.norm_info.input_max,
                    'target_min': self.norm_info.target_min,
                    'target_max': self.norm_info.target_max
                }
        
        # Save the PyTorch model
        pt_path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'base_neurons': self.base_neurons,
                'dropout_prob': self.dropout_prob,
                'norm_type': self.norm_type,
                'activation': self.activation_type, 
                'mix_norm': self.mix_norm,
                'norm_info_data': norm_info_data
            }
        }, pt_path)
        
        # Save the ONNX model
        onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(
            self, dummy_input, onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        print(f"Model saved to: {pt_path} and {onnx_path}")
        return pt_path, onnx_path
    
    @classmethod
    def load_model(cls, model_path: str, device: str = 'cpu'):
        """
        Load the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model state and configuration
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']
        
        # Restore the norm_info object
        norm_info = None
        if 'norm_info_data' in model_config and model_config['norm_info_data'] is not None:
            norm_info_data = model_config['norm_info_data']
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
        
        # Remove the norm_info_data, because the constructor does not need this parameter
        if 'norm_info_data' in model_config:
            del model_config['norm_info_data']
        
        # Create the model instance
        model = cls(**model_config, norm_info=norm_info)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'base_neurons': self.base_neurons,
            'dropout_prob': self.dropout_prob,
            'norm_type': self.norm_type,
            'activation': self.activation_type, 
            'mix_norm': self.mix_norm
        }

    def get_norm_info(self):
        if self.norm_type == "standardization":
            return StandardizationInfo(
                input_mean=self.input_mean.cpu().numpy().tolist() if self.input_mean is not None else None,
                input_std=self.input_std.cpu().numpy().tolist() if self.input_std is not None else None,
                target_mean=float(self.target_mean.cpu().numpy()) if self.target_mean is not None else None,
                target_std=float(self.target_std.cpu().numpy()) if self.target_std is not None else None
            )
        elif self.norm_type == "min_max":   
            return MinMaxNormInfo(
                input_min=self.input_min.cpu().numpy().tolist() if self.input_min is not None else None,
                input_max=self.input_max.cpu().numpy().tolist() if self.input_max is not None else None,
                target_min=float(self.target_min.cpu().numpy()) if self.target_min is not None else None,
                target_max=float(self.target_max.cpu().numpy()) if self.target_max is not None else None
            )
        else:
            return None


# Keep backward compatibility
class DNN(FlexibleDNN):
    """
    Backward compatible DNN class
    """
    def __init__(self, input_dim=5, base_neurons=16, dropout_prob=0.025,
                 input_mean=None, input_std=None, target_mean=None, target_std=None):
        # Convert the old parameter format
        norm_info = None
        if input_mean is not None or input_std is not None or target_mean is not None or target_std is not None:
            norm_info = StandardizationInfo(
                input_mean=input_mean,
                input_std=input_std,
                target_mean=target_mean,
                target_std=target_std
            )
        
        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            base_neurons=base_neurons,
            dropout_prob=dropout_prob,
            norm_type="standardization",
            norm_info=norm_info,
            activation="exp_decay",
            mix_norm=True
        )