from dataclasses import dataclass
from typing import Union, Optional, List
from enum import Enum


class ModelType(Enum):
    """Model type enumeration"""
    FLEXIBLE_DNN = "flexible_dnn"
    TRANSFORMER = "transformer"
    CNN = "cnn"


@dataclass
class StandardizationInfo:
    """
    Standardization information class
    """
    input_mean: Optional[List[float]] = None
    input_std: Optional[List[float]] = None
    target_mean: Optional[float] = None
    target_std: Optional[float] = None


@dataclass
class MinMaxNormInfo:
    """
    Min-max normalization information class
    """
    input_min: Optional[List[float]] = None
    input_max: Optional[List[float]] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None


class model_config:
    """
    DNN model configuration class
    """
    
    norm_info: Union[StandardizationInfo, MinMaxNormInfo]
    
    def __init__(self,
                 input_dim: int = 5,
                 output_dim: int = 1,
                 base_neurons: int = 16,
                 dropout_prob: float = 0.025,
                 model_type: Union[str, ModelType] = ModelType.FLEXIBLE_DNN,  # ModelType enum or string
                 norm_type: str = "standardization",  # "standardization" 或 "min_max"
                 norm_info: Optional[Union[StandardizationInfo, MinMaxNormInfo]] = None,
                 activation: str = "exp_decay",  # "exp_decay", "relu", "tanh", "sigmoid"
                 mix_norm: bool = True):
        """
        Initialize the model configuration
        
        Args:
            input_dim: input dimension
            output_dim: output dimension
            base_neurons: base neurons
            dropout_prob: dropout probability
            model_type: model type (ModelType enum or string)
            norm_type: normalization type ("standardization" or "min_max")
            norm_info: normalization information object
            activation: activation function type
            mix_norm: whether to include normalization in the model
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_neurons = base_neurons
        self.dropout_prob = dropout_prob
        # Convert string to ModelType enum if needed
        if isinstance(model_type, str):
            try:
                self.model_type = ModelType(model_type)
            except ValueError:
                raise ValueError(f"Invalid model_type string: {model_type}. "
                               f"Valid values: {[mt.value for mt in ModelType]}")
        else:
            self.model_type = model_type
            
        self.norm_type = norm_type
        self.activation = activation
        self.mix_norm = mix_norm
        
        # Validate model_type
        self._validate_model_type()
        
        # Set the default norm_info based on the norm_type
        if norm_info is None:
            if norm_type == "standardization":
                self.norm_info = StandardizationInfo()
            elif norm_type == "min_max":
                self.norm_info = MinMaxNormInfo()
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")
        else:
            self.norm_info = norm_info
            
        # Validate the norm_info type and norm_type match
        self._validate_norm_info()
    
    def _validate_model_type(self):
        """Validate the model_type"""
        if not isinstance(self.model_type, ModelType):
            raise TypeError(f"model_type must be a ModelType enum, got {type(self.model_type)}")
    
    def _validate_norm_info(self):
        """Validate the norm_info type and norm_type match"""
        if self.norm_type == "standardization" and not isinstance(self.norm_info, StandardizationInfo):
            raise TypeError("When norm_type is 'standardization', norm_info must be StandardizationInfo type")
        elif self.norm_type == "min_max" and not isinstance(self.norm_info, MinMaxNormInfo):
            raise TypeError("When norm_type is 'min_max', norm_info must be MinMaxNormInfo type")
    
    def get_model_kwargs(self):
        """Get the dictionary of parameters passed to the model constructor"""
        kwargs = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'base_neurons': self.base_neurons,
            'dropout_prob': self.dropout_prob,
            'norm_type': self.norm_type,
            'activation': self.activation,
            'mix_norm': self.mix_norm,
            'norm_info': self.norm_info
        }
        
        return kwargs
    
    def get_dnn_kwargs(self):
        """Backward compatibility method - use get_model_kwargs instead"""
        return self.get_model_kwargs()
    
    def __str__(self):
        """Return the string representation of the configuration"""
        return (f"model_config(\n"
                f"  input_dim={self.input_dim},\n"
                f"  output_dim={self.output_dim},\n"
                f"  base_neurons={self.base_neurons},\n"
                f"  dropout_prob={self.dropout_prob},\n"
                f"  model_type={self.model_type.value},\n"
                f"  norm_type='{self.norm_type}',\n"
                f"  activation='{self.activation}',\n"
                f"  mix_norm={self.mix_norm},\n"
                f"  norm_info={self.norm_info}\n"
                f")")
    
    def __repr__(self):
        return self.__str__()
