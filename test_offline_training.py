#!/usr/bin/env python3
"""
Test script for offline training functionality
"""

import numpy as np
import sys
import os
import pandas as pd

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

def create_test_csv():
    """Create a test CSV file for offline training"""
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 200
    inputs = np.random.randn(n_samples, 5)  # 5 input features
    outputs = np.random.randn(n_samples, 1)  # 1 output target
    
    # Create DataFrame
    df = pd.DataFrame({
        'numberDensity': inputs[:, 0],
        'divU': inputs[:, 1],
        'APi': inputs[:, 2],
        'APjSum': inputs[:, 3],
        'ghostWeightP': inputs[:, 4],
        'pressure': outputs[:, 0]
    })
    
    # Save to CSV
    os.makedirs('test_data', exist_ok=True)
    csv_path = 'test_data/test_offline_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Created test CSV file: {csv_path}")
    
    return csv_path

def test_offline_training():
    """Test the offline training functionality"""
    
    print("="*60)
    print("Testing Offline Training Pipeline")
    print("="*60)
    
    # Create test CSV file
    # csv_path = create_test_csv()
    csv_path = "train_data/training_poisson_data_RPM1000.csv"
    # csv_path = ["train_data/training_poisson_data_RPM500.csv", "train_data/training_poisson_data_RPM1000.csv"]
    
    # Create configuration
    config = model_config(
        input_dim=5,
        output_dim=1,
        base_neurons=16,
        norm_type="standardization",
        activation="exp_decay",
        mix_norm=True
    )
    
    # Create pipeline manager
    pipeline = PipelineManager(config)
    
    try:
        # Run offline training pipeline
        results = pipeline.train_and_save_offline(
            csv_files=csv_path,
            epochs=20,  # Small number for testing
            batch_size=10240,
            test_size=0.2,
            save_dir="test_output",
            model_name="test_offline_model"
        )
        
        print(f"\nOffline Training Results:")
        print(f"  Best validation loss: {results['training_history']['best_val_loss']:.6f}")
        print(f"  Training time: {results['training_history']['training_time']:.2f} seconds")
        print(f"  Epochs trained: {results['training_history']['epochs_trained']}")
        print(f"  Model saved as: {results['model_name']}")
        print(f"  Output directory: {results['save_dir']}")
        
        # Test prediction
        print(f"\nTesting prediction with sample data...")
        sample_input = np.random.randn(5, 5)  # 5 samples
        predictions = pipeline.predict_batch(sample_input)
        print(f"  Sample predictions shape: {predictions.shape}")
        print(f"  Sample predictions: {predictions.flatten()[:3]}...")
        
        print(f"\n✅ Offline training test completed successfully!")
        
    except Exception as e:
        print(f"❌ Offline training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_offline_training()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 