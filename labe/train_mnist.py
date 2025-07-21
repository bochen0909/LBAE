import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .mnist_model import GenConvResBlock32, EncConvResBlock32
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


def validate_model(model, val_dataloader, device_obj, vae=False, shared_weights=False):
    """Validate the model on validation dataset"""
    model.eval()
    val_loss = 0
    val_reco_loss = 0
    val_kld_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, target, xc in val_dataloader:
            x = x.to(device_obj)
            xc = xc.to(device_obj)
            
            # Forward pass through LABE
            ws = model.E.layers if shared_weights else None
            xr, mu, logvar, z, err_quant = model(xc, ws)
            
            # Calculate losses using LABE's built-in method
            total_loss, loss_reco, loss_kld = model.reconstruction_loss(x, xr, mu, logvar if vae else None)
            
            total_loss = loss_reco
            loss_kld = torch.tensor(0.0)
            
            # Statistics
            val_loss += total_loss.item()
            val_reco_loss += loss_reco.item()
            if vae:
                val_kld_loss += loss_kld.item()
            num_batches += 1
    
    avg_val_loss = val_loss / num_batches
    avg_val_reco_loss = val_reco_loss / num_batches
    avg_val_kld_loss = val_kld_loss / num_batches if vae else 0
    
    return avg_val_loss, avg_val_reco_loss, avg_val_kld_loss


class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            self.best_epoch = epoch
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience


def train_mnist(lr=1e-4, batch_size=512, epochs_max=100, 
                print_every_batch=10, device='auto', workers=4,
                channels=1, img_size=32, zsize=200, vae_model='ConvResBlock32',
                vae=False, kl_weight=1.0, l2=0, corrupt_method='blank', corrupt_args=None,
                binary_reco_loss=True, shared_weights=False,
                save_every=10, model_path='./checkpoints',
                early_stopping_patience=10, min_delta=1e-6, val_split=0.1):
    """Main training function for MNIST using LABE class with validation and early stopping"""
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
    
    # Create full training dataset
    full_train_dataset = MNISTEx(dataroot + 'MNIST', train=True, download=True, 
                                transform=transform, corrupt_method=corrupt_method, 
                                corrupt_args=corrupt_args)
    
    # Split dataset into train and validation
    full_size = len(full_train_dataset)
    val_size = int(full_size * val_split)
    train_size = full_size - val_size
    
    print(f"Dataset split: Training size: {train_size}, Validation size: {val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size])
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=workers, drop_last=True, pin_memory=True)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=workers, drop_last=False, pin_memory=True)
    
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
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=min_delta)
    
    print(f"Starting training for {epochs_max} epochs with early stopping (patience={early_stopping_patience})...")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs_max):
        model.train()
        
        epoch_loss = 0
        epoch_reco_loss = 0
        epoch_kld_loss = 0
        num_batches = 0
        
        for batch_idx, (x, target, xc) in enumerate(train_dataloader):
            
            x = x.to(device_obj)
            xc = xc.to(device_obj)
            
            # Forward pass through LABE
            ws = model.E.layers if shared_weights else None
            xr, mu, logvar, z, err_quant = model(xc, ws)
            
            # Calculate losses using LABE's built-in method
            total_loss, loss_reco, loss_kld = model.reconstruction_loss(x, xr, mu, logvar if vae else None)
            
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
        
        # Validation
        avg_val_loss, avg_val_reco_loss, avg_val_kld_loss = validate_model(
            model, val_dataloader, device_obj, vae, shared_weights)
        
        print(f'=== Epoch {epoch:3d}/{epochs_max} Complete ===')
        print(f'Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
        print(f'Train Reco: {avg_reco_loss:.6f} | Val Reco: {avg_val_reco_loss:.6f}')
        if vae:
            print(f'Train KLD: {avg_kld_loss:.6f} | Val KLD: {avg_val_kld_loss:.6f}')
        
        # Save best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_path, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'hps': {'lr': lr, 'batch_size': batch_size, 
                       'epochs_max': epochs_max, 'print_every_batch': print_every_batch,
                       'device': device, 'workers': workers,
                       'channels': channels, 'img_size': img_size, 'zsize': zsize,
                       'vae_model': vae_model, 'vae': vae, 'kl_weight': kl_weight,
                       'l2': l2, 'corrupt_method': corrupt_method, 'corrupt_args': corrupt_args,
                       'binary_reco_loss': binary_reco_loss,
                       'shared_weights': shared_weights, 'save_every': save_every,
                       'model_path': model_path, 'early_stopping_patience': early_stopping_patience,
                       'min_delta': min_delta, 'val_split': val_split}
            }, best_model_path)
            print(f'✓ New best model saved: {best_model_path} (Val Loss: {avg_val_loss:.6f})')
        
        # Early stopping check
        if early_stopping(avg_val_loss, epoch):
            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            print(f'Best validation loss: {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch + 1}')
            break
        
        # Save regular checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs_max - 1:
            checkpoint_path = os.path.join(model_path, f'mnist_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'hps': {'lr': lr, 'batch_size': batch_size, 
                       'epochs_max': epochs_max, 'print_every_batch': print_every_batch,
                       'device': device, 'workers': workers,
                       'channels': channels, 'img_size': img_size, 'zsize': zsize,
                       'vae_model': vae_model, 'vae': vae, 'kl_weight': kl_weight,
                       'l2': l2, 'corrupt_method': corrupt_method, 'corrupt_args': corrupt_args,
                       'binary_reco_loss': binary_reco_loss,
                       'shared_weights': shared_weights, 'save_every': save_every,
                       'model_path': model_path, 'early_stopping_patience': early_stopping_patience,
                       'min_delta': min_delta, 'val_split': val_split}
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            
    print("\nTraining completed!")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(f"Best model saved at: {os.path.join(model_path, 'best_model.pt')}")

def create_labe_model(channels, img_size, zsize, vae_model, kl_weight, binary_reco_loss):
    """Create and initialize a LABE model with given hyperparameters"""
    # Initialize individual models with correct parameters
    # EncConvResBlock32 expects (channels, zsize, zround)
    # For zround, we'll use a default value of 4 (decimal points for quantization)
    E = EncConvResBlock32(channels, zsize, zround=-4)
    
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
        train_loss = checkpoint.get('train_loss', checkpoint.get('loss'))
        val_loss = checkpoint.get('val_loss', None)
        
        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Train loss: {train_loss:.6f}")
        if val_loss is not None:
            print(f"Validation loss: {val_loss:.6f}")
        return epoch, train_loss, val_loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, None, None

if __name__ == "__main__":
    # Example usage with explicit parameters including validation and early stopping:
    train_mnist(
        lr=1e-4,
        batch_size=512,
        epochs_max=100,
        zsize=200,
        kl_weight=1.0,
        val_split=0.1,  # Use 10% of training data for validation
        early_stopping_patience=10,  # Stop if no improvement for 10 epochs
        min_delta=1e-6  # Minimum improvement threshold
    )
    
    # For different configurations, you can call with different parameters:
    # train_mnist(lr=5e-5, batch_size=256, epochs_max=200, 
    #            val_split=0.15, early_stopping_patience=15)
    #
    # To resume from checkpoint, you would need to modify the function
    # or create a separate resume function