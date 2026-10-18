import numpy as np
import random
from scipy.stats import rice


def add_gaussian_noise(image, mean=0, stddev=0.35, ratio=0.11):
    '''
    Add Gaussian noise to a portion of random pixels in the image.
    image: numpy array of shape (H, W)
    mean: mean of the Gaussian noise
    stddev: standard deviation of the Gaussian noise
    ratio: ratio of pixels to which noise is added
    '''
    noisy_image = image.copy().astype(np.float32)
    m, n = image.shape
    nbrpx = int(m * n * ratio)
    indices = np.random.choice(m * n, nbrpx, replace=False)
    coords = np.unravel_index(indices, (m, n))
    bruit = np.random.normal(mean, stddev, nbrpx)
    noisy_image[coords] += bruit
    return np.clip(noisy_image, 0, 65535).astype(np.uint16)

def add_gaussian_noise_full(image, mean=0, stddev=40, seed=None):
    """
    Add Gaussian noise to the entire image.
    """
    rng = np.random.default_rng(seed)

    image_float = image.astype(np.float32)
    noise = rng.normal(mean, stddev, image.shape)

    noisy_image = image_float + noise

    return np.clip(noisy_image, 0, 65535).astype(np.uint16)

def add_rician_noise(image, b=0, loc=0, scale=1, seed=None):
    rng = np.random.default_rng(seed)

    noise = rice.rvs(
        b,
        loc=loc,
        scale=scale,
        size=image.shape,
        random_state=rng
    )

    noisy_image = np.clip(image + noise, 0, 65535)

    return noisy_image.astype(np.uint16)

def add_salt_pepper_like(image):
    n, m = image.shape
    I_new = image.copy()
    dynamic_range = np.max(image)
    for i in range(n):
        for j in range(m):
            I_new[i, j] = image[i, j] + random.randint(0, dynamic_range - image[i, j]) 
    return I_new


def apply_noise(image,NOISE_CONFIG, noise_type, level, seed=None):
    config = NOISE_CONFIG[level]

    if noise_type == "gaussian":
        return add_gaussian_noise_full(
            image,
            stddev=config["gaussian"],
            seed=seed
        )

    elif noise_type == "rician":
        return add_rician_noise(
            image,
            scale=config["rician"],
            seed=seed
        )

    elif noise_type == "mixed":
        gaussian_std, rician_scale = config["mixed"]

        noisy = add_gaussian_noise_full(
            image,
            stddev=gaussian_std,
            seed=seed
        )

        noisy = add_rician_noise(
            noisy,
            scale=rician_scale,
            seed=None if seed is None else seed + 1
        )

        return noisy

    else:
        raise ValueError(
            "noise_type must be 'gaussian', 'rician' or 'mixed'."
        )