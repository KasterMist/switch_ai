import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import time
from py_solver.pipeline_manager import PipelineManager
from py_solver.model_manager import ModelManager
from py_solver.inference import InferenceEngine, PredictionService
from py_solver.config import model_config, ModelType
from py_solver.data_manager import DataManager
from py_solver.model_factory import create_dnn_model


def train_model():
    config = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="exp_decay",
        mix_norm=True,
        model_type=ModelType.FLEXIBLE_DNN
    )

    # data manager creation
    data_manager = DataManager(config)
    csv_files = [
        "train_data/training_poisson_data_RPM500.csv",
        "train_data/training_poisson_data_RPM1000.csv",
        "train_data/training_poisson_data_RPM2500.csv",
    ]
    # features, targets = data_manager.read_training_csv(csv_files)

    data_manager.print_summary()
        
    # create a pipeline manager
    pipeline = PipelineManager(config, device='cuda', model_save_dir="models")
    pipeline.train_and_save(csv_files=csv_files, epochs=20, batch_size=10240, model_name="trained_model")

def test_model():
    engine = InferenceEngine("models/trained_model_20250729_145025/trained_model.onnx")

    df_predict = pd.read_csv("test_data/training_poisson_data_RPM1500.csv")
    df_predict = df_predict.drop(columns=["pressure"])

    prediction_results = engine.predict(df_predict)
    print("success")

def main():
    train_model()
    
if __name__ == "__main__":
    main()