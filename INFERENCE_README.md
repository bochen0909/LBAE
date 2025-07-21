# LABE Model Inference Scripts

This directory contains scripts for running inference with trained LABE (Lightweight Autoencoder) models on MNIST data.

## Files

- `inference_demo.py` - Main inference script that loads checkpoints and runs inference
- `run_inference_examples.py` - Example usage script showing different ways to run inference

## Requirements

Make sure you have the following Python packages installed:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Basic Usage

Run inference with the default best model checkpoint:

```bash
python inference_demo.py
```

### Advanced Usage

Run inference with specific parameters:

```bash
# Use a specific checkpoint
python inference_demo.py --checkpoint ./test_checkpoints/mnist_epoch_3.pt

# Specify device (auto, cuda, mps, cpu)
python inference_demo.py --device cuda

# Set custom output directory
python inference_demo.py --output ./my_inference_results

# Set random seed for reproducible results
python inference_demo.py --seed 42

# Combine multiple options
python inference_demo.py --checkpoint ./test_checkpoints/best_model.pt --device mps --output ./results --seed 123
```

### Example Script

Run multiple inference examples:

```bash
python run_inference_examples.py
```

This will run several inference examples and save results to different output directories.

## What the Script Does

1. **Loads Model**: Loads a trained LABE model from a checkpoint file
2. **Loads Data**: Downloads and loads MNIST test dataset
3. **Random Selection**: Randomly selects an image from the test set
4. **Inference**: Runs the image through the model to get reconstruction and hidden vector
5. **Saves Results**: Saves original image, reconstructed image, and comparison plot
6. **Prints Info**: Displays detailed information about the hidden z vector

## Output Files

The script creates the following outputs in the specified directory:

- `inference_comparison_XXXXX.png` - Side-by-side comparison of original and reconstructed images
- `original_XXXXX.png` - Original input image
- `reconstructed_XXXXX.png` - Model's reconstruction
- `z_vector_XXXXX.npy` - Hidden z vector saved as NumPy array

Where `XXXXX` is a random 5-digit number for uniqueness.

## Command Line Arguments

- `--checkpoint, -c`: Path to checkpoint file (default: `./test_checkpoints/best_model.pt`)
- `--device, -d`: Device to use (auto, cuda, mps, cpu) (default: auto)
- `--output, -o`: Output directory for results (default: `./inference_output`)
- `--seed, -s`: Random seed for reproducible results (optional)

## Available Checkpoints

The script looks for checkpoints in these locations:
- `./checkpoints/` - Training checkpoints from main training runs
- `./test_checkpoints/` - Available test checkpoints:
  - `best_model.pt` - Best performing model
  - `mnist_epoch_1.pt` - Model after 1 epoch
  - `mnist_epoch_2.pt` - Model after 2 epochs  
  - `mnist_epoch_3.pt` - Model after 3 epochs

## Output Information

The script prints detailed information including:

- Model hyperparameters (image size, z-vector dimensions, etc.)
- Selected image details (index, label, shape)
- Hidden z vector statistics (mean, std, min, max)
- First 20 and last 10 elements of the z vector
- Reconstruction loss
- File paths where results are saved

## Examples

### Example Output

```
=== LABE Model Inference Demo ===
Checkpoint: ./test_checkpoints/best_model.pt
Device: mps

Loading checkpoint...
Loaded checkpoint from epoch 3
Train loss: 0.045123
Validation loss: 0.048756

Model hyperparameters:
  - Image size: 32x32, Channels: 1
  - Z-vector size: 200
  - VAE model: ConvResBlock32
  - Binary reconstruction loss: True

Selected random image #1234 with label: 7
Image shape: torch.Size([1, 32, 32])

=== Hidden Z Vector Information ===
Z vector shape: torch.Size([1, 200])
Z vector statistics:
  - Mean: 0.123456
  - Std:  1.234567
  - Min:  -3.456789
  - Max:  4.567890

First 20 elements of Z vector:
[ 0.1234 -0.5678  1.2345 ... ]

Reconstruction loss: 0.043210

=== Inference Complete ===
Image label: 7
Reconstruction loss: 0.043210
Z vector saved with 200 dimensions
All outputs saved to: ./inference_output
```

## Troubleshooting

1. **Checkpoint not found**: Make sure the checkpoint file exists. The script will list available checkpoints if the specified one is not found.

2. **MNIST download fails**: The script automatically downloads MNIST data to `~/projects/data/MNIST`. Make sure you have internet access and write permissions.

3. **Device issues**: If CUDA/MPS is not available, the script will automatically fall back to CPU when using `--device auto`.

4. **Import errors**: Make sure all required packages are installed and the LABE module is properly set up in the Python path.

## Python API Usage

You can also use the inference function directly in Python:

```python
from inference_demo import inference_demo

# Basic usage
results = inference_demo()

# With custom parameters
results = inference_demo(
    checkpoint_path='./test_checkpoints/best_model.pt',
    device='auto',
    output_dir='./my_results'
)

# Access results
print(f"Label: {results['label']}")
print(f"Loss: {results['loss']}")
print(f"Z vector: {results['z_vector']}")
```
