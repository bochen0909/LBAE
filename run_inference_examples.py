#!/usr/bin/env python3
"""
Simple usage example for the inference demo script.
This shows how to run inference with different configurations.
"""

from inference_demo import inference_demo
import os

def main():
    """Example usage of the inference demo function"""
    
    print("🔬 LABE Inference Demo Examples")
    print("=" * 50)
    
    # Example 1: Use default checkpoint (best model)
    print("\n📝 Example 1: Using default best model checkpoint")
    try:
        results = inference_demo()
        print("✅ Demo 1 completed successfully!")
    except Exception as e:
        print(f"❌ Demo 1 failed: {e}")
    
    # Example 2: Use a specific checkpoint
    print("\n📝 Example 2: Using specific checkpoint")
    checkpoint_path = "./test_checkpoints/mnist_epoch_3.pt"
    if os.path.exists(checkpoint_path):
        try:
            results = inference_demo(
                checkpoint_path=checkpoint_path,
                device='auto',
                output_dir='./inference_output_epoch3'
            )
            print("✅ Demo 2 completed successfully!")
            print(f"   - Image label: {results['label']}")
            print(f"   - Reconstruction loss: {results['loss']:.6f}")
            print(f"   - Z vector dimensions: {len(results['z_vector'])}")
        except Exception as e:
            print(f"❌ Demo 2 failed: {e}")
    else:
        print(f"⚠️ Skipping Demo 2: {checkpoint_path} not found")
    
    # Example 3: Multiple runs for comparison
    print("\n📝 Example 3: Multiple runs for comparison")
    for i in range(3):
        try:
            results = inference_demo(
                output_dir=f'./inference_output_run_{i+1}'
            )
            print(f"✅ Run {i+1}: Label={results['label']}, Loss={results['loss']:.6f}")
        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
    
    print("\n🎉 All demos completed!")
    print("\nOutput directories created:")
    output_dirs = [
        './inference_output',
        './inference_output_epoch3', 
        './inference_output_run_1',
        './inference_output_run_2', 
        './inference_output_run_3'
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  📁 {output_dir}: {len(files)} files")


if __name__ == "__main__":
    main()
