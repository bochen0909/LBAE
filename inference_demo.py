#!/usr/bin/env python3
"""
Script to load a LABE checkpoint, randomly select an MNIST image, 
apply the model, and save input/output images while printing the hidden z vector.
"""

import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from labe.datasets import MNISTEx
from labe.train_mnist import get_device, create_labe_model, load_checkpoint


def denormalize_image(tensor):
    """Convert tensor to numpy array suitable for visualization"""
    # Convert from tensor to numpy and scale to [0,1]
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


def save_image_comparison(original, reconstructed, z_vector, output_dir="./inference_output"):
    """Save original and reconstructed images side by side with z vector info"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    axes[0].imshow(denormalize_image(original), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(denormalize_image(reconstructed), cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    # Add z vector info as text
    fig.suptitle(f"LABE Model Inference\nZ-vector size: {len(z_vector)}", fontsize=12)
    
    # Save the comparison
    timestamp = np.random.randint(10000, 99999)  # Simple timestamp alternative
    output_path = os.path.join(output_dir, f'inference_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison image saved: {output_path}")
    
    # Save individual images
    original_path = os.path.join(output_dir, f'original_{timestamp}.png')
    reconstructed_path = os.path.join(output_dir, f'reconstructed_{timestamp}.png')
    
    plt.figure(figsize=(4, 4))
    plt.imshow(denormalize_image(original), cmap='gray')
    plt.axis('off')
    plt.savefig(original_path, dpi=150, bbox_inches='tight')
    print(f"✓ Original image saved: {original_path}")
    
    plt.figure(figsize=(4, 4))
    plt.imshow(denormalize_image(reconstructed), cmap='gray')
    plt.axis('off')
    plt.savefig(reconstructed_path, dpi=150, bbox_inches='tight')
    print(f"✓ Reconstructed image saved: {reconstructed_path}")
    
    plt.close('all')
    return output_path, original_path, reconstructed_path


def inference_demo(checkpoint_path=None, device='auto', output_dir="./inference_output"):
    """
    Main inference demonstration function
    
    Args:
        checkpoint_path: Path to the checkpoint file to load
        device: Device to run inference on ('auto', 'cuda', 'mps', 'cpu')
        output_dir: Directory to save output images
    """
    
    # Set default checkpoint if none provided
    if checkpoint_path is None:
        checkpoint_path = "./test_checkpoints/best_model.pt"
    
    print("=== LABE Model Inference Demo ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Get device
    device_obj = get_device(device)
    print(f"Device: {device_obj}")
    
    # Check if checkpoint exists
    if not os.path.isfile(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for checkpoint_dir in ["./checkpoints", "./test_checkpoints"]:
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                for cp in checkpoints:
                    print(f"  - {os.path.join(checkpoint_dir, cp)}")
        return
    
    # Load checkpoint to get hyperparameters
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    
    # Extract hyperparameters or use defaults
    hps = checkpoint.get('hps', {})
    channels = hps.get('channels', 1)
    img_size = hps.get('img_size', 32)
    zsize = hps.get('zsize', 200)
    vae_model = hps.get('vae_model', 'ConvResBlock32')
    kl_weight = hps.get('kl_weight', 1.0)
    binary_reco_loss = hps.get('binary_reco_loss', True)
    corrupt_method = hps.get('corrupt_method', 'blank')
    corrupt_args = hps.get('corrupt_args', [])
    
    print("Model hyperparameters:")
    print(f"  - Image size: {img_size}x{img_size}, Channels: {channels}")
    print(f"  - Z-vector size: {zsize}")
    print(f"  - VAE model: {vae_model}")
    print(f"  - Binary reconstruction loss: {binary_reco_loss}")
    
    # Create model
    print("Creating model...")
    model = create_labe_model(channels, img_size, zsize, vae_model, kl_weight, binary_reco_loss)
    model.to(device_obj)
    
    # Load checkpoint
    epoch, train_loss, val_loss = load_checkpoint(checkpoint_path, model)
    model.eval()  # Set to evaluation mode
    
    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    dataroot = "~/projects/data/"
    transform = transforms.Compose([
        transforms.Pad(2, fill=0),  # Pad to 32x32
        transforms.ToTensor(),
    ])
    
    test_dataset = MNISTEx(dataroot + 'MNIST', train=False, download=True, 
                          transform=transform, corrupt_method=corrupt_method, 
                          corrupt_args=corrupt_args)
    
    # Randomly select an image
    random_idx = random.randint(0, len(test_dataset) - 1)
    original_image, label, corrupted_image = test_dataset[random_idx]
    
    print(f"Selected random image #{random_idx} with label: {label}")
    print(f"Image shape: {original_image.shape}")
    
    # Prepare for inference
    original_batch = original_image.unsqueeze(0).to(device_obj)  # Add batch dimension
    corrupted_batch = corrupted_image.unsqueeze(0).to(device_obj)
    
    # Run inference
    print("Running model inference...")
    with torch.no_grad():
        reconstructed, mu, logvar, z, err_quant = model(corrupted_batch)
    
    # Print z vector information
    print("\n=== Hidden Z Vector Information ===")
    print(f"Z vector shape: {z.shape}")
    print("Z vector statistics:")
    print(f"  - Mean: {z.mean().item():.6f}")
    print(f"  - Std:  {z.std().item():.6f}")
    print(f"  - Min:  {z.min().item():.6f}")
    print(f"  - Max:  {z.max().item():.6f}")
    print("\nFirst 20 elements of Z vector:")
    print(z.squeeze().cpu().numpy()[:20])
    print("\nLast 10 elements of Z vector:")
    print(z.squeeze().cpu().numpy()[-10:])
    
    if err_quant is not None:
        print(f"\nQuantization error: {err_quant.item():.6f}")
    
    # Calculate reconstruction loss
    with torch.no_grad():
        total_loss, reco_loss, kld_loss = model.reconstruction_loss(
            original_batch, reconstructed, mu, logvar)
    print(f"\nReconstruction loss: {reco_loss.item():.6f}")
    
    # Save images and comparison
    print("\n=== Saving Results ===")
    comparison_path, original_path, reconstructed_path = save_image_comparison(
        original_image, reconstructed.squeeze(), z.squeeze(), output_dir)
    
    # Save z vector as numpy array
    z_output_path = os.path.join(output_dir, f'z_vector_{random.randint(10000, 99999)}.npy')
    np.save(z_output_path, z.squeeze().cpu().numpy())
    print(f"✓ Z vector saved: {z_output_path}")
    
    print("\n=== Inference Complete ===")
    print(f"Image label: {label}")
    print(f"Reconstruction loss: {reco_loss.item():.6f}")
    print(f"Z vector saved with {len(z.squeeze())} dimensions")
    print(f"All outputs saved to: {output_dir}")
    
    return {
        'original': original_image,
        'reconstructed': reconstructed.squeeze(),
        'z_vector': z.squeeze(),
        'label': label,
        'loss': reco_loss.item(),
        'paths': {
            'comparison': comparison_path,
            'original': original_path,
            'reconstructed': reconstructed_path,
            'z_vector': z_output_path
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LABE Model Inference Demo')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to checkpoint file (default: ./test_checkpoints/best_model.pt)')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--output', '-o', type=str, default='./inference_output',
                       help='Output directory for saved images (default: ./inference_output)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Run inference demo
    try:
        results = inference_demo(
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_dir=args.output
        )
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
