#!/usr/bin/env python3
"""
Test script for online training functionality
"""

import numpy as np
import sys
import os
from py_solver.data_manager import DataManager

# Add the py_solver directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_solver'))

try:
    from py_solver.pipeline_manager import PipelineManager
    from py_solver.config import model_config
    print("Successfully imported pipeline modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all dependencies are installed")
    sys.exit(1)

def test_online_training():
    """Test the online training functionality"""
    
    print("="*60)
    print("Testing Online Training Pipeline")
    print("="*60)
    
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
    
    
    csv_path = "train_data/training_poisson_data_RPM1000.csv"
    data_manager = DataManager(config)
    features, targets = data_manager._read_single_csv(csv_path)
    inputs = features
    outputs = targets
    
    try:
        # Run online training pipeline
        results = pipeline.train_and_save_online(
            inputs=inputs,
            outputs=outputs,
            epochs=20,  # Small number for testing
            batch_size=10240,
            test_size=0.2,
            save_dir="test_output",
            model_name="test_online_model"
        )
        
        print(f"\nOnline Training Results:")
        print(f"  Best validation loss: {results['training_history']['best_val_loss']:.6f}")
        print(f"  Training time: {results['training_history']['training_time']:.2f} seconds")
        print(f"  Epochs trained: {results['training_history']['epochs_trained']}")
        print(f"  Model saved as: {results['model_name']}")
        print(f"  Output directory: {results['save_dir']}")
        
        # Test prediction
        print(f"\nTesting prediction with sample data...")
        sample_input = inputs[:5]  # First 5 samples
        predictions = pipeline.predict_batch(sample_input)
        print(f"  Sample predictions shape: {predictions.shape}")
        print(f"  Sample predictions: {predictions.flatten()[:3]}...")
        
        print(f"\nOnline training test completed successfully!")
        
    except Exception as e:
        print(f"Online training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_online_training()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
        sys.exit(1) 