# LABE Training with Validation and Early Stopping

This document describes the enhanced MNIST training script with validation dataset support and early stopping functionality.

## New Features

### 1. Validation Dataset
- Automatically splits the training data into training and validation sets
- Default validation split is 10% (`val_split=0.1`)
- Validation is performed after each training epoch

### 2. Early Stopping
- Monitors validation loss to prevent overfitting
- Stops training when validation loss doesn't improve for a specified number of epochs
- Default patience is 10 epochs (`early_stopping_patience=10`)
- Minimum improvement threshold can be set (`min_delta=1e-6`)

### 3. Best Model Saving
- Automatically saves the model with the best validation loss
- Saved as `best_model.pt` in the model directory
- Regular checkpoint saving continues as before

## Usage

### Basic Usage
```python
from labe.train_mnist import train_mnist

# Train with validation and early stopping (default settings)
train_mnist(
    lr=1e-4,
    batch_size=512,
    epochs_max=100,
    val_split=0.1,                    # 10% validation split
    early_stopping_patience=10,       # Stop after 10 epochs without improvement
    min_delta=1e-6                    # Minimum improvement threshold
)
```

### Advanced Configuration
```python
# Custom validation and early stopping settings
train_mnist(
    lr=5e-5,
    batch_size=256,
    epochs_max=200,
    val_split=0.15,                   # 15% validation split
    early_stopping_patience=15,       # More patience for longer training
    min_delta=1e-5,                   # Stricter improvement threshold
    save_every=5
)
```

## New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `val_split` | float | 0.1 | Fraction of training data to use for validation |
| `early_stopping_patience` | int | 10 | Number of epochs to wait for improvement before stopping |
| `min_delta` | float | 1e-6 | Minimum change in validation loss to be considered an improvement |

## Output Files

### Checkpoints
- Regular checkpoints: `mnist_epoch_{epoch}.pt` (saved every `save_every` epochs)
- Best model: `best_model.pt` (saved whenever validation loss improves)

### Checkpoint Contents
Each checkpoint now includes:
- Model state dict
- Optimizer state dict
- Training and validation losses
- All hyperparameters including new validation/early stopping parameters

## Training Output

The enhanced training provides:
- Both training and validation metrics for each epoch
- Clear indication when a new best model is saved
- Early stopping notification with best achieved validation loss

Example output:
```
=== Epoch   5/100 Complete ===
Train Loss: 0.045123 | Val Loss: 0.048567
Train Reco: 0.045123 | Val Reco: 0.048567
✓ New best model saved: ./checkpoints/best_model.pt (Val Loss: 0.048567)

...

Early stopping triggered after 23 epochs!
Best validation loss: 0.042156 at epoch 18

Training completed!
Best validation loss achieved: 0.042156
Best model saved at: ./checkpoints/best_model.pt
```

## Loading Checkpoints

The `load_checkpoint` function has been updated to handle both old and new checkpoint formats:

```python
from labe.train_mnist import load_checkpoint, create_labe_model

# Load best model
model = create_labe_model(channels=1, img_size=32, zsize=200, 
                         vae_model='ConvResBlock32', kl_weight=1.0, 
                         binary_reco_loss=True)
epoch, train_loss, val_loss = load_checkpoint('./checkpoints/best_model.pt', model)
```

## Benefits

1. **Prevents Overfitting**: Validation monitoring helps detect when the model starts overfitting
2. **Automatic Stopping**: No need to manually monitor training - stops automatically when optimal
3. **Best Model Preservation**: Always keeps the best performing model regardless of when training stops
4. **Flexible Configuration**: Easy to adjust validation split and early stopping behavior
5. **Backward Compatibility**: All existing training scripts continue to work with default validation settings
