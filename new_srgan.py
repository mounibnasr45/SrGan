import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed()

# Save directory
save_dir = './esrgan_output'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/samples", exist_ok=True)
os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

# Model parameters
nf1 = 64    # Number of filters
gc1 = 32    # Growth channels

# Residual Dense Block
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=nf1, gc=gc1):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=nf1, gc=gc1):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.RDB3(self.RDB2(self.RDB1(x))) * 0.2 + x

class GeneratorRRDB(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=nf1, nb=23, gc=gc1, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=False)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)

        # Upsampling layers (dynamic based on scale factor)
        upsampling_layers = []
        for _ in range(int(np.log2(scale))):
            upsampling_layers.extend([
                nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, True)
            ])
        self.upsampling = nn.Sequential(*upsampling_layers)
        
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        fea = fea + self.trunk_conv(trunk)
        out = self.conv_last(self.upsampling(fea))
        return out

# Custom Dataset
class DIV2KSRDataset(Dataset):
    def __init__(self, split="train", cache=True, hr_size=1048, lr_size=256):
        self.dataset = load_dataset("eugenesiow/Div2k", split=split, cache_dir="./div2k_data")
        self.cache = cache
        self.cached_data = {}
        self.hr_size = hr_size
        self.lr_size = lr_size

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.cache and idx in self.cached_data:
            return self.cached_data[idx]

        img_path = self.dataset[idx]["hr"]
        img = Image.open(img_path).convert("RGB")
        hr = self.hr_transform(img)
        lr = self.lr_transform(img)

        if self.cache:
            self.cached_data[idx] = (lr, hr)
        return lr, hr

# Utility functions
def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array for visualization"""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=1.0, channel_axis=2, multichannel=True)

def visualize_results(model, val_loader, device, epoch, save_dir, num_samples=3):
    """Visualize and save sample results"""
    model.eval()
    with torch.no_grad():
        for i, (lr, hr) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # Convert tensors to numpy arrays for visualization
            lr_np = tensor_to_numpy(lr[0])
            hr_np = tensor_to_numpy(hr[0])
            sr_np = tensor_to_numpy(sr[0])
            
            # Calculate metrics
            psnr_val = calculate_psnr(sr_np, hr_np)
            ssim_val = calculate_ssim(sr_np, hr_np)
            
            # Create visualization
            plt.figure(figsize=(20, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(lr_np)
            plt.title(f'LR ({lr_np.shape[0]}x{lr_np.shape[1]})')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sr_np)
            plt.title(f'SR - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(hr_np)
            plt.title(f'HR ({hr_np.shape[0]}x{hr_np.shape[1]})')
            plt.axis('off')
            
            plt.suptitle(f'Epoch {epoch}')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/samples/epoch_{epoch}_sample_{i}.png")
            plt.close()

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and compute metrics"""
    model.eval()
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for lr, hr in tqdm(test_loader, desc="Evaluating"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # Convert to numpy for metric calculation
            for i in range(sr.size(0)):
                sr_np = tensor_to_numpy(sr[i])
                hr_np = tensor_to_numpy(hr[i])
                
                # Calculate metrics
                psnr_val = calculate_psnr(sr_np, hr_np)
                ssim_val = calculate_ssim(sr_np, hr_np)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return avg_psnr, avg_ssim

def train_esrgan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with the correct scale factor for 256->1048
    scale_factor = 4  # 256 * 4 = 1024 (close to 1048)
    model = GeneratorRRDB(scale=scale_factor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # Load dataset
    full_dataset = DIV2KSRDataset(split="train", cache=True, hr_size=1048, lr_size=256)
    
    # Split dataset into train, validation, and test
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.L1Loss()

    # Training parameters
    num_epochs = 50
    best_psnr = 0
    val_freq = 1  # Validate every epoch
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})

        scheduler.step()
        
        # Validation phase
        if (epoch + 1) % val_freq == 0:
            # Visualize some results
            visualize_results(model, val_loader, device, epoch+1, save_dir)
            
            # Evaluate on validation set
            val_psnr, val_ssim = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch+1} - Validation: PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), f"{save_dir}/checkpoints/esrgan_best.pth")
                print(f"Saved best model with PSNR={val_psnr:.2f}")
        
        # Regular checkpoint saving
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/esrgan_epoch{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}")

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_psnr, test_ssim = evaluate_model(model, test_loader, device)
    print(f"Test Results: PSNR={test_psnr:.2f}, SSIM={test_ssim:.4f}")
    
    # Load best model for final inference examples
    model.load_state_dict(torch.load(f"{save_dir}/checkpoints/esrgan_best.pth"))
    visualize_results(model, test_loader, device, epoch="final", save_dir=save_dir, num_samples=5)
    
    return model

def run_inference(model_path, test_loader, device, save_dir):
    """Run inference on test samples using trained model"""
    model = GeneratorRRDB(scale=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    os.makedirs(f"{save_dir}/inference", exist_ok=True)
    
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            if i >= 10:  # Show 10 test samples
                break
                
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            
            # Convert tensors to numpy arrays for visualization
            lr_np = tensor_to_numpy(lr[0])
            hr_np = tensor_to_numpy(hr[0])
            sr_np = tensor_to_numpy(sr[0])
            
            # Calculate metrics
            psnr_val = calculate_psnr(sr_np, hr_np)
            ssim_val = calculate_ssim(sr_np, hr_np)
            
            # Create visualization
            plt.figure(figsize=(20, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(lr_np)
            plt.title(f'Low Resolution ({lr_np.shape[0]}x{lr_np.shape[1]})')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sr_np)
            plt.title(f'Super Resolution - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(hr_np)
            plt.title(f'High Resolution Ground Truth ({hr_np.shape[0]}x{hr_np.shape[1]})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/inference/test_sample_{i}.png", dpi=300)
            plt.close()
    
    print(f"Inference completed. Results saved to {save_dir}/inference/")

if __name__ == "__main__":
    # Train model
    trained_model = train_esrgan()
    
    # Load dataset for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = DIV2KSRDataset(split="validation", cache=True, hr_size=1048, lr_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Run inference with best model
    run_inference(
        model_path=f"{save_dir}/checkpoints/esrgan_best.pth", 
        test_loader=test_loader,
        device=device,
        save_dir=save_dir
    )