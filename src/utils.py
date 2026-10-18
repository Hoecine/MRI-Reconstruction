
import os
import nibabel as nib
import numpy as np
from collections import Counter
import SimpleITK as sitk
from pathlib import Path
import re
import torch; import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F



def lire_nii(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)

def slice_index(fname):
    m = re.search(r'_(\d+)\.ima$', fname.lower())
    return int(m.group(1)) if m else 10**9

def extract_patches(X_img, Y_img, patch_size=5, stride=3):
    X_patches = []
    y_targets = []
    offset = patch_size // 2
    h, w = X_img.shape

    for i in range(offset, h - offset, stride):
        for j in range(offset, w - offset, stride):
            patch = X_img[i-offset:i+offset+1, j-offset:j+offset+1]
            label = Y_img[i, j]
            X_patches.append(patch.flatten())
            y_targets.append(label)

    return np.array(X_patches), np.array(y_targets)

import cv2

def blur_metric(img):
    if img.dtype != np.uint8:
        img = ((img / 4095) * 255).astype(np.uint8)
    return cv2.Laplacian(img, cv2.CV_32F).var()

def add_rician_noise_raw(x, alpha=0.05, seed=None):
    """
    x: np array 
    alpha: intensité relative (0.02 easy, 0.05 medium, 0.08 hard)
    """
    if seed is not None:
        np.random.seed(seed)

    x = x.astype(np.float32)
    # Pencetiles
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    rng = max(p99 - p1, 1e-6)
    sigma = alpha * rng

    n1 = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    n2 = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)

    y = np.sqrt((x + n1)**2 + n2**2).astype(np.float32)
    return y


class T2noise_dataset(Dataset):
    def __init__(self, clean_nii_dir, alpha_range=(0.02, 0.08), target_shape=(320, 320)):
        self.paths = [
            os.path.join(clean_nii_dir, f)
            for f in os.listdir(clean_nii_dir)
            if f.endswith(".nii.gz")
        ]
        self.alpha_range = alpha_range
        self.target_shape = target_shape

        self.volumes = []
        self.index = []

        for p in self.paths:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)

            if arr.shape[1:] != self.target_shape:
                continue

            vol_id = len(self.volumes)
            self.volumes.append(arr)

            for k in range(arr.shape[0]):
                self.index.append((vol_id, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vol_i, k = self.index[idx]
        x = self.volumes[vol_i][k]

        alpha = np.random.uniform(*self.alpha_range)
        y = add_rician_noise_raw(x, alpha=alpha)

        x_t = torch.from_numpy(x).unsqueeze(0).float()
        y_t = torch.from_numpy(y).unsqueeze(0).float()

        return y_t, x_t
    
class T2NoiseDatasetFixed(Dataset):
    def __init__(self, clean_nii_dir, alpha=0.05, base_seed=123, target_shape=(320, 320)):
        self.paths = [
            os.path.join(clean_nii_dir, f)
            for f in os.listdir(clean_nii_dir)
            if f.endswith(".nii.gz")
        ]
        self.alpha = alpha
        self.base_seed = base_seed
        self.target_shape = target_shape

        self.volumes = []
        self.index = []

        for i, p in enumerate(self.paths):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
            if arr.shape[1:] != self.target_shape:
                continue
            self.volumes.append(arr)

            z = arr.shape[0]
            for k in range(z):
                self.index.append((i, k))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vol_i, k = self.index[idx] # idx receives always the same noise
        x = self.volumes[vol_i][k]

        seed = self.base_seed + idx
        y = add_rician_noise_raw(x, alpha=self.alpha, seed=seed)

        x_t = torch.from_numpy(x).unsqueeze(0).float()
        y_t = torch.from_numpy(y).unsqueeze(0).float()

        return y_t, x_t
    
def sobel_grad(x):
    """
    x: (B,1,H,W)
    return: gradient magnitude (B,1,H,W)
    """
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)


def edge_loss(den, clean):
    """
    L1 sur la magnitude des gradients (contours)
    """
    g_den = sobel_grad(den)
    g_cln = sobel_grad(clean)
    return torch.mean(torch.abs(g_den - g_cln))

def evaluate(model, loader, criterion, device, lam=0.0):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            pred_noise = model(noisy)
            target = noisy - clean
            
            den = noisy - pred_noise
            
            loss_noise = criterion(pred_noise, target)
            
            # Optionnal : regularization
            # if lam = 0 then it's not activated
            if lam > 0:
                loss_edge = edge_loss(den, clean)
                loss = loss_noise + lam * loss_edge
            else:
                loss = loss_noise
            
            running_loss += loss.item()

    return running_loss / len(loader)

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3, model_name="model", lam=0.0, save_dir="Results Denoising"):
    model = model.to(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            target = noisy - clean
            optimizer.zero_grad()

            pred_noise = model(noisy)
            loss_noise = criterion(pred_noise, target)

            if lam > 0:
                loss_edge = edge_loss(noisy - pred_noise, clean)
                loss = loss_noise + lam * loss_edge
            else:
                loss = loss_noise

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device, lam=lam)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / f"{model_name}.pth")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val": best_val
        }, save_dir / f"ckpt_{model_name}.pth")

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={epoch_train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    return model, train_losses, val_losses

def robust_range_per_image(x, q_low=0.01, q_high=0.99, eps=1e-6):
    """
    x: (B,1,H,W) tensor
    retourne: (B,) range robuste par image
    """
    B = x.shape[0]
    out = []
    for i in range(B):
        xi = x[i,0]
        p1 = torch.quantile(xi, q_low)
        p99 = torch.quantile(xi, q_high)
        out.append(torch.clamp(p99 - p1, min=eps))
    return torch.stack(out)

# L1 error
def mae(den, ref):
    """
        den : denoised image
        ref : clean image
    """
    return torch.mean(torch.abs(den - ref))

# PSNR
def psnr(den, ref, R):
    """
        PSNR = 10 log10(R^2 / MSE)
    """
    mse = torch.mean((den - ref) ** 2, dim=(1,2,3)) 
    if not torch.is_tensor(R):
        R = torch.tensor(R, device=den.device, dtype=den.dtype).expand_as(mse)
    elif R.ndim == 0:
        R = R.expand_as(mse)
    return (10 * torch.log10((R**2) / (mse + 1e-12))).mean()

# SSIM
def _gaussian_kernel(ws=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(ws, device=device, dtype=dtype) - ws//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g = g / g.sum()
    k2d = (g[:,None] * g[None,:]).view(1,1,ws,ws)
    return k2d

def ssim(den, ref, L, ws=11, sigma=1.5, K1=0.01, K2=0.03):
    """
    den, ref: (B,1,H,W)
    L: dynamic range (scalar or (B,))
    """
    device = den.device
    kernel = _gaussian_kernel(ws, sigma, device=device, dtype=den.dtype)
    pad = ws//2

    mu_x = F.conv2d(den, kernel, padding=pad)
    mu_y = F.conv2d(ref, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(den*den, kernel, padding=pad) - mu_x2
    sig_y2 = F.conv2d(ref*ref, kernel, padding=pad) - mu_y2
    sig_xy = F.conv2d(den*ref, kernel, padding=pad) - mu_xy

    vals = []
    for i in range(den.shape[0]):
        Li = L if not torch.is_tensor(L) else (L[i] if L.ndim==1 else L)
        C1 = (K1*Li)**2
        C2 = (K2*Li)**2
        num = (2*mu_xy[i:i+1] + C1) * (2*sig_xy[i:i+1] + C2)
        deno = (mu_x2[i:i+1] + mu_y2[i:i+1] + C1) * (sig_x2[i:i+1] + sig_y2[i:i+1] + C2) + 1e-12
        vals.append((num/deno).mean())
    return torch.stack(vals).mean()

# GSMD (contours)
def gmsd(den: torch.Tensor, ref: torch.Tensor, c=170/255) -> torch.Tensor:

    device = den.device
    dtype = den.dtype

    # Sobel
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)

    gx_d = F.conv2d(den, kx, padding=1)
    gy_d = F.conv2d(den, ky, padding=1)
    gm_d = torch.sqrt(gx_d**2 + gy_d**2 + 1e-12)

    gx_r = F.conv2d(ref, kx, padding=1)
    gy_r = F.conv2d(ref, ky, padding=1)
    gm_r = torch.sqrt(gx_r**2 + gy_r**2 + 1e-12)

    # Similarity map
    c2 = (c**2)
    gms = (2*gm_d*gm_r + c2) / (gm_d**2 + gm_r**2 + c2)

    # std pooling (par image), puis moyenne batch
    gms_flat = gms.view(gms.shape[0], -1)
    return torch.std(gms_flat, dim=1).mean()

# BLUR
def BLUR(x):
    """
        Laplacian variance : Blur(I) = Var(ΔI)
    """
    device = x.device
    dtype = x.dtype
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=device, dtype=dtype).view(1,1,3,3)
    y = F.conv2d(x, lap, padding=1)
    y_flat = y.view(y.shape[0], -1)
    return torch.var(y_flat, dim=1).mean()

def evaluate_denoising(model, loader, device, use_robust_range=True, R_fixed=None):
    model.eval()
    sums = {"MAE":0.0, "PSNR":0.0, "SSIM":0.0, "BLUR_clean":0.0, "BLUR_denoised":0.0}
    n = 0

    with torch.no_grad():
        
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
    
            pred_noise = model(noisy)
            den = noisy - pred_noise
    
            # Range for PSNR/SSIM
            if use_robust_range:
                R = robust_range_per_image(clean)      # (B,)
                L = R                                  
            else:
                R = R_fixed
                L = R_fixed
    
            sums["MAE"] += mae(den, clean).item()
            sums["PSNR"] += psnr(den, clean, R).item()
            sums["SSIM"] += ssim(den, clean, L).item()
    
            sums["BLUR_clean"] += BLUR(clean).item()
            sums["BLUR_denoised"] += BLUR(den).item()
    
            n += 1

    for k in sums:
        sums[k] /= max(1, n)
    return sums

def visualise_denoising(model, data_loader, device, n=3, batch_idx=0, show_error=True, suptitle=None):
    model.eval()

    selected_batch = None
    for i, (noisy, clean) in enumerate(data_loader):
        if i == batch_idx:
            selected_batch = (noisy, clean)
            break


    noisy, clean = selected_batch
    noisy = noisy.to(device)
    clean = clean.to(device)

    with torch.no_grad():
        pred_noise = model(noisy)
        denoised = noisy - pred_noise

    noisy = noisy.cpu()
    clean = clean.cpu()
    denoised = denoised.cpu()

    batch_size = noisy.shape[0]
    n = min(n, batch_size)

    ncols = 4 if show_error else 3
    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))

    if n == 1:
        axes = [axes]

    for j in range(n):
        clean_img = clean[j, 0].numpy()
        noisy_img = noisy[j, 0].numpy()
        denoised_img = denoised[j, 0].numpy()
        err_img = abs(denoised_img - clean_img)

        axes[j][0].imshow(clean_img, cmap="gray")
        axes[j][0].set_title("Clean")
        axes[j][0].axis("off")

        axes[j][1].imshow(noisy_img, cmap="gray")
        axes[j][1].set_title("Noisy")
        axes[j][1].axis("off")

        axes[j][2].imshow(denoised_img, cmap="gray")
        axes[j][2].set_title("Denoised")
        axes[j][2].axis("off")

        if show_error:
            axes[j][3].imshow(err_img, cmap="hot")
            axes[j][3].set_title("Abs error")
            axes[j][3].axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    plt.show()

def evaluate_all_alphas(model, test_dir, device, model_name="model", target_shape=(320,320), alphas=[0.02, 0.05, 0.08]):
    results = {}

    for a in alphas:
        ds = T2NoiseDatasetFixed(test_dir, alpha=a, base_seed=999, target_shape=target_shape)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        m = evaluate_denoising(model, ld, device)
        results[a] = m

        print(f"{model_name} | alpha={a} -> {m}")
        visualise_denoising(model, ld, device, n=3, batch_idx=0, show_error=True)

    return results

def plot_losses(train_losses, val_losses, save_path=None, model_name="model", title="Training and Validation Loss"):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()