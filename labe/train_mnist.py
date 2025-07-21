import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .models5 import GenConvResBlock32, EncConvResBlock32
from .datasets import MNISTEx
from .lbae import LABE


# === Set Device ===
def get_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA requested but not available.")
    elif device_preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            raise ValueError("MPS requested but not available.")
    else:
        return torch.device("cpu")


def weight_init(m):
    """Initialize model weights"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def net_info(model):
    """Print model parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def train_mnist(lr=1e-4, batch_size=512, epochs_max=100, 
                print_every_batch=10, device='auto', workers=4,
                channels=1, img_size=32, zsize=200, vae_model='ConvResBlock32',
                vae=False, kl_weight=1.0, l2=0, corrupt_method='blank', corrupt_args=None,
                binary_reco_loss=True, shared_weights=False,
                save_every=10, model_path='./checkpoints'):
    """Main training function for MNIST using LABE class"""
    print("Starting MNIST Training...")
    
    # Handle default arguments
    device_obj = get_device(device)
    if corrupt_args is None:
        corrupt_args = []
     
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Using device: {device_obj}")
    
    # Load data
    print("Loading MNIST dataset...")
    dataroot = "~/projects/data/"
    transform = transforms.Compose([
        transforms.Pad(2, fill=0),  # Pad to 32x32
        transforms.ToTensor(),
    ])
    
    train_dataset = MNISTEx(dataroot + 'MNIST', train=True, download=True, 
                          transform=transform, corrupt_method=corrupt_method, 
                          corrupt_args=corrupt_args)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=workers, drop_last=True, pin_memory=True)
    
    print(f'Training size: {len(train_dataset)}')
    
    # Initialize models
    print("Initializing models...")
    model = create_labe_model(channels, img_size, zsize, vae_model, kl_weight, binary_reco_loss)
    
    print("Encoder:")
    net_info(model.E)
    print("Generator:")
    net_info(model.G)
    
    model.to(device_obj)
    print(f"LABE model moved to {device_obj}")
    
    # Setup optimizer
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=l2)
    
    print(f"Starting training for {epochs_max} epochs...")
    
    # Training loop
    for epoch in range(epochs_max):
        model.train()
        
        epoch_loss = 0
        epoch_reco_loss = 0
        epoch_kld_loss = 0
        num_batches = 0
        
        for batch_idx, (x, target, xc) in enumerate(train_dataloader):
            batch_size_current = x.size(0)
            
            x = x.to(device_obj)
            xc = xc.to(device_obj)
            
            # Forward pass through LABE
            ws = model.E.layers if shared_weights else None
            xr, mu, logvar, z, err_quant = model(xc, ws)
            
            # Calculate losses using LABE's built-in method
            total_loss, loss_reco, loss_kld = model.reconstruction_loss(x, xr, mu, logvar if vae else None)
            
            # Add KL divergence for VAE
            if vae and loss_kld is not None:
                # LABE's reconstruction_loss doesn't compute KLD, so we compute it here
                varlog = torch.clamp(logvar, -10, 10)
                mu_clamped = torch.clamp(mu, -10, 10)
                loss_kld = -0.5 * torch.sum(1 + varlog - mu_clamped.pow(2) - varlog.exp())
                loss_kld = loss_kld / batch_size_current
                loss_kld = kl_weight * loss_kld / zsize
                total_loss = loss_reco + loss_kld
            else:
                total_loss = loss_reco
                loss_kld = torch.tensor(0.0)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += total_loss.item()
            epoch_reco_loss += loss_reco.item()
            if vae:
                epoch_kld_loss += loss_kld.item()
            num_batches += 1
            
            # Print batch statistics
            if batch_idx % print_every_batch == 0:
                print(f'Epoch {epoch:3d}/{epochs_max} '
                      f'Batch {batch_idx:4d}/{len(train_dataloader)} '
                      f'Loss: {total_loss.item():.6f} '
                      f'Reco: {loss_reco.item():.6f} '
                      + (f'KLD: {loss_kld.item():.6f}' if vae else ''))
        
        # End of epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_reco_loss = epoch_reco_loss / num_batches
        avg_kld_loss = epoch_kld_loss / num_batches if vae else 0
        
        print(f'=== Epoch {epoch:3d}/{epochs_max} Complete ===')
        print(f'Average Loss: {avg_loss:.6f}')
        print(f'Average Reco Loss: {avg_reco_loss:.6f}')
        if vae:
            print(f'Average KLD Loss: {avg_kld_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs_max - 1:
            checkpoint_path = os.path.join(model_path, f'mnist_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'hps': {'lr': lr, 'batch_size': batch_size, 
                       'epochs_max': epochs_max, 'print_every_batch': print_every_batch,
                       'device': device, 'workers': workers,
                       'channels': channels, 'img_size': img_size, 'zsize': zsize,
                       'vae_model': vae_model, 'vae': vae, 'kl_weight': kl_weight,
                       'l2': l2, 'corrupt_method': corrupt_method, 'corrupt_args': corrupt_args,
                       'binary_reco_loss': binary_reco_loss,
                       'shared_weights': shared_weights, 'save_every': save_every,
                       'model_path': model_path}
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            # Also save as latest
            latest_path = os.path.join(model_path, 'mnist_latest.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'hps': {'lr': lr, 'batch_size': batch_size, 
                       'epochs_max': epochs_max, 'print_every_batch': print_every_batch,
                       'device': device, 'workers': workers,
                       'channels': channels, 'img_size': img_size, 'zsize': zsize,
                       'vae_model': vae_model, 'vae': vae, 'kl_weight': kl_weight,
                       'l2': l2, 'corrupt_method': corrupt_method, 'corrupt_args': corrupt_args,
                       'binary_reco_loss': binary_reco_loss,
                       'shared_weights': shared_weights, 'save_every': save_every,
                       'model_path': model_path}
            }, latest_path)
    
    print("Training completed!")

def create_labe_model(channels, img_size, zsize, vae_model, kl_weight, binary_reco_loss):
    """Create and initialize a LABE model with given hyperparameters"""
    # Initialize individual models with correct parameters
    # EncConvResBlock32 expects (channels, zsize, zround)
    # For zround, we'll use a default value of 4 (decimal points for quantization)
    E = EncConvResBlock32(channels, zsize, zround=4)
    
    # GenConvResBlock32 expects (channels, dataset, zsize)  
    # We're training on MNIST, so dataset='mnist'
    G = GenConvResBlock32(channels, dataset='mnist', zsize=zsize)
    
    # Apply weight initialization
    weight_init(E)
    weight_init(G)
    
    # Create LABE model
    model = LABE(G=G, E=E, zsize=zsize, kl_weight=kl_weight, 
                 binary_reco_loss=binary_reco_loss)
    
    return model

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load a saved checkpoint for LABE model"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Loaded checkpoint from epoch {epoch}, loss: {loss:.6f}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, None

if __name__ == "__main__":
    # Example usage with explicit parameters:
    train_mnist(
        lr=1e-4,
        batch_size=512,
        epochs_max=100,
        zsize=200,
        kl_weight=1.0
    )
    
    # For different configurations, you can call with different parameters:
    # train_mnist(lr=5e-5, batch_size=256, epochs_max=200)
    #
    # To resume from checkpoint, you would need to modify the function
    # or create a separate resume function