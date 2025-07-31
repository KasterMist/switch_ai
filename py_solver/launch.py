import numpy as np
import sys
import os
from py_solver.pipeline_manager import PipelineManager
from py_solver.config import model_config
from py_solver.data_manager import DataManager

def cplusplus_launch_python_training(config: model_config, inputs, outputs, save_dir: str, model_name: str):
    pipeline = PipelineManager(config, device='cuda', model_save_dir=save_dir)
    pipeline.train_and_save_online(
        inputs=inputs,
        outputs=outputs,
        epochs=20,  # Small number for testing
        batch_size=10240,
        test_size=0.2,
        save_dir=save_dir,
        model_name=model_name
    )

def launch_python_training_in_terminal(csv_path):
    # Create configuration
    config = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="exp_decay",
        mix_norm=True
    )
    data_manager = DataManager(config)
    features, targets = data_manager._read_single_csv(csv_path)
    inputs = features
    outputs = targets 
    cplusplus_launch_python_training(config, inputs, outputs, "models/terminal_model", "test_model")