#!/usr/bin/env python3
"""
Test script for the updated MNIST training with validation and early stopping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from labe.train_mnist import train_mnist

if __name__ == "__main__":
    print("Testing MNIST training with validation and early stopping...")
    print("Running a short test with minimal parameters...")
    
    # Run a quick test with very few epochs to verify functionality
    try:
        train_mnist(
            lr=1e-3,
            batch_size=128,  # Smaller batch for faster testing
            epochs_max=3,    # Very few epochs for quick test
            print_every_batch=5,
            zsize=50,        # Smaller latent size for speed
            val_split=0.2,   # Use 20% for validation in this test
            early_stopping_patience=2,
            min_delta=1e-4,
            save_every=1,    # Save every epoch in test
            model_path='./test_checkpoints'
        )
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
