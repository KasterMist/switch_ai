"""
Model factory module
Provide the function to create different types of models based on the configuration
"""

from typing import Type, Optional
from .config import model_config, ModelType
from .dnn import FlexibleDNN


# Model registry for different model types
MODEL_REGISTRY = {
    ModelType.FLEXIBLE_DNN: FlexibleDNN,
    # ModelType.TRANSFORMER: TransformerModel,  # Not implemented yet
    # ModelType.CNN: CNNModel,  # Not implemented yet
}


def create_model(config: model_config, model_class: Optional[Type] = None):
    """
    Create model instance based on the configuration
    
    Args:
        config: model configuration object
        model_class: model class, if None, use config.model_type to determine
    
    Returns:
        created model instance
    """
    if model_class is None:
        # Use model_type from config
        model_type = config.model_type
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type.value}. "
                           f"Supported types: {[mt.value for mt in MODEL_REGISTRY.keys()]}")
        model_class = MODEL_REGISTRY[model_type]
    
    # Get model parameters from config
    kwargs = config.get_model_kwargs()
    
    return model_class(**kwargs)


def create_dnn_model(config: model_config):
    """
    Create DNN model (backward compatibility)
    
    Args:
        config: model configuration object
    
    Returns:
        FlexibleDNN model instance
    """
    # Ensure config is set to flexible_dnn
    if config.model_type != ModelType.FLEXIBLE_DNN:
        raise ValueError(f"create_dnn_model requires model_type={ModelType.FLEXIBLE_DNN.value}, but got '{config.model_type.value}'")
    
    return create_model(config)


# Future can add more model types of factory function
def create_transformer_model(config: model_config):
    """
    Create Transformer model
    
    Args:
        config: model configuration object
    
    Returns:
        Transformer model instance
    """
    # Here can add the logic to create Transformer model
    # For example: return TransformerModel(config)
    raise NotImplementedError("Transformer model is not implemented")


def create_transformer_model(config: model_config):
    """
    Create Transformer model
    
    Args:
        config: model configuration object
    
    Returns:
        Transformer model instance
    """
    # Ensure config is set to transformer
    if config.model_type != ModelType.TRANSFORMER:
        raise ValueError(f"create_transformer_model requires model_type={ModelType.TRANSFORMER.value}, but got '{config.model_type.value}'")
    
    raise NotImplementedError("Transformer model is not implemented")


def create_cnn_model(config: model_config):
    """
    Create CNN model
    
    Args:
        config: model configuration object
    
    Returns:
        CNN model instance
    """
    # Ensure config is set to cnn
    if config.model_type != ModelType.CNN:
        raise ValueError(f"create_cnn_model requires model_type={ModelType.CNN.value}, but got '{config.model_type.value}'")
    
    raise NotImplementedError("CNN model is not implemented") 


def register_model_type(model_type: ModelType, model_class: Type):
    """
    Register a new model type
    
    Args:
        model_type: ModelType enum value
        model_class: model class to register
    """
    MODEL_REGISTRY[model_type] = model_class


def get_supported_model_types() -> list:
    """
    Get list of supported model types
    
    Returns:
        List of supported model type values (strings)
    """
    return [mt.value for mt in MODEL_REGISTRY.keys()]


def get_supported_model_types_enum() -> list:
    """
    Get list of supported model types as enums
    
    Returns:
        List of supported ModelType enums
    """
    return list(MODEL_REGISTRY.keys()) 