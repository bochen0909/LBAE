# Latent Bernoulli Autoencoder (LBAE)

A PyTorch implementation of the paper [**Latent Bernoulli Autoencoder**](https://proceedings.icml.cc/static/paper_files/icml/2020/3022-Paper.pdf) by Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso and Paolo Remagnino, presented at [ICML 2020](https://icml.cc/Conferences/2020).

## Overview

The Latent Bernoulli Autoencoder (LBAE) is a variational autoencoder that uses Bernoulli latent variables for discrete representation learning. This implementation provides:

- **Complete LBAE model** with encoder and generator components
- **MNIST training pipeline** with validation and early stopping
- **Inference capabilities** for image reconstruction and latent vector extraction
- **Quantization techniques** for discrete latent representations

## Features

- ✅ Full LBAE implementation with PyTorch
- ✅ Training with validation split and early stopping
- ✅ Model checkpointing and best model saving
- ✅ Inference pipeline for image reconstruction
- ✅ Latent vector extraction and visualization
- ✅ MNIST dataset integration
- ✅ GPU acceleration support (CUDA/MPS)

## Installation

### Using Poetry (Recommended)
```bash
git clone https://github.com/ok1zjf/LBAE.git
cd LBAE
poetry install
```

### Using pip
```bash
git clone https://github.com/ok1zjf/LBAE.git
cd LBAE
pip install torch torchvision matplotlib tqdm
```

## Quick Start

### Training

Train the LBAE model on MNIST:

```python
from lbae.train_mnist import train_mnist

# Basic training with validation and early stopping
train_mnist(
    lr=1e-4,
    batch_size=512,
    epochs_max=100,
    val_split=0.1,              # 10% validation split
    early_stopping_patience=10  # Early stopping after 10 epochs without improvement
)
```

### Inference

Run inference on trained models:

```bash
# Basic inference with best model
python inference_demo.py

# Advanced usage
python inference_demo.py --checkpoint ./checkpoints/best_model.pt --device cuda --output ./results

# Run multiple examples
python run_inference_examples.py
```

## Usage

### Training Configuration

The training script supports various configuration options:

```python
train_mnist(
    lr=5e-5,                    # Learning rate
    batch_size=256,             # Batch size
    epochs_max=200,             # Maximum epochs
    val_split=0.15,             # Validation split ratio
    early_stopping_patience=15, # Early stopping patience
    min_delta=1e-6,             # Minimum improvement threshold
    zsize=100,                  # Latent vector size
    save_every=5,               # Checkpoint saving frequency
    model_path='./checkpoints'  # Model save directory
)
```

### Model Architecture

The LBAE consists of:

- **Encoder (E)**: Converts input images to latent representations
- **Generator (G)**: Reconstructs images from latent vectors
- **Quantizer**: Provides discrete latent representations
- **LABE Module**: Combines encoder and generator with loss computation

### Inference Options

```bash
# Command line options for inference_demo.py
--checkpoint    # Path to model checkpoint (default: ./checkpoints/best_model.pt)
--device        # Device to use: auto, cuda, mps, cpu (default: auto)
--output        # Output directory for results (default: ./inference_output)
--seed          # Random seed for reproducible results
```
  
## Advanced Usage

### Custom Model Configuration

```python
from lbae.lbae import LABE
from lbae.mnist_model import Generator, Encoder

# Create custom model
G = Generator(zsize=100)
E = Encoder(zsize=100)
model = LABE(G, E, zsize=100, kl_weight=1.0, binary_reco_loss=False)
```

### Batch Inference

```python
# Run inference on multiple images
from lbae.train_mnist import create_labe_model, load_checkpoint

model = create_labe_model(zsize=100)
model = load_checkpoint(model, './checkpoints/best_model.pt')

# Process batch of images
with torch.no_grad():
    reconstructions, mu, logvar, z, _ = model(batch_images)
```

## Testing

Run the test training script to verify installation:

```bash
python test_training.py
```

This will run a short training session with minimal parameters to ensure everything works correctly.

## Performance

The model achieves good reconstruction quality on MNIST:
- Fast training convergence with early stopping
- Compact latent representations
- High-quality reconstructions
- Efficient inference pipeline

## Documentation

- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - Comprehensive training documentation
- [`INFERENCE_README.md`](INFERENCE_README.md) - Detailed inference guide
- [Original Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/3022-Paper.pdf) - ICML 2020 publication

## Requirements

- Python 3.11+
- PyTorch 2.7+
- torchvision 0.22+
- matplotlib 3.10+
- tqdm 4.66+

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the terms specified in the [`LICENSE`](LICENSE) file.

## Cite

If you use this code or reference our paper in your work, please use the following citation:

```bibtex
@inproceedings{fajtl2020latent,
  title={Latent Bernoulli Autoencoder},
  author={Fajtl, Jiri and Argyriou, Vasileios and Monekosso, Dorothy and Remagnino, Paolo},
  booktitle={International Conference on Machine Learning},
  pages={2964--2974},
  year={2020},
  organization={PMLR}
}
```

