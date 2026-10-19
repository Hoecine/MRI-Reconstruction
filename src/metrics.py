from scipy.ndimage import gaussian_filter
import numpy as np

def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    if np.any(sigma1 <= 0) or np.any(sigma2 <= 0):
        raise ValueError("Standard deviations must be positive.")
    
    kl_div = 0.5 * (-np.log(sigma1**2 / sigma2**2) + (sigma1**2 + (mu1 - mu2)**2) / (sigma2**2) - 1)
    return kl_div

def akld_image(noised_img, denoised_img, gt_img):
    mu = gt_img.astype(np.float32)
    
    P = gaussian_filter((denoised_img - mu)**2, sigma=1)
    Q = gaussian_filter((noised_img - mu)**2, sigma=1)

    # Pour avoir une seule valeur globale : on fait la moyenne du KL sur l’image
    kl_map = kl_divergence_normal(mu, np.sqrt(P), mu, np.sqrt(Q))
    return np.mean(kl_map)

import cv2


def blur_metric(img):
    if img.dtype != np.uint8:
        img = ((img / 4095) * 255).astype(np.uint8)
    return cv2.Laplacian(img, cv2.CV_32F).var()