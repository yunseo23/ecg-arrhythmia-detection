import numpy as np
from scipy.interpolate import interp1d
import random

def add_gaussian_noise(signal, noise_level=0.01):
    """Add Gaussian noise to the signal"""
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

def time_warp(signal, sigma=0.2):
    """Apply time warping to the signal"""
    orig_steps = np.arange(signal.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(signal.shape[0],))
    warp_steps = (np.cumsum(random_warps))
    warp_steps = warp_steps * (signal.shape[0]-1) / warp_steps[-1]
    
    ret = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        interp = interp1d(orig_steps, signal[:, i])
        ret[:, i] = interp(warp_steps)
    return ret

def amplitude_scale(signal, sigma=0.1):
    """Scale the amplitude of the signal"""
    scaling_factor = np.random.normal(1.0, sigma)
    return signal * scaling_factor

def random_crop(signal, crop_ratio=0.8):
    """Randomly crop a portion of the signal"""
    crop_length = int(signal.shape[0] * crop_ratio)
    start_idx = np.random.randint(0, signal.shape[0] - crop_length)
    cropped = signal[start_idx:start_idx + crop_length]
    
    # Resize back to original length
    orig_steps = np.arange(signal.shape[0])
    crop_steps = np.linspace(0, signal.shape[0]-1, crop_length)
    ret = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        interp = interp1d(crop_steps, cropped[:, i])
        ret[:, i] = interp(orig_steps)
    return ret

def mixup(signal1, signal2, alpha=0.2):
    """Mix two signals together"""
    lam = np.random.beta(alpha, alpha)
    return lam * signal1 + (1 - lam) * signal2

def apply_augmentation(signal, augmentation_type, params=None):
    """Apply specified augmentation to the signal"""
    if params is None:
        params = {}
    
    if augmentation_type == 'gaussian_noise':
        return add_gaussian_noise(signal, params.get('noise_level', 0.01))
    elif augmentation_type == 'time_warp':
        return time_warp(signal, params.get('sigma', 0.2))
    elif augmentation_type == 'amplitude_scale':
        return amplitude_scale(signal, params.get('sigma', 0.1))
    elif augmentation_type == 'random_crop':
        return random_crop(signal, params.get('crop_ratio', 0.8))
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")

def get_augmented_data(signal, label, augmentation_config):
    """Apply multiple augmentations based on configuration"""
    augmented_signals = [signal]
    augmented_labels = [label]
    
    for aug_type, params in augmentation_config.items():
        if aug_type == 'mixup':
            # For mixup, we need another signal
            if len(augmented_signals) > 1:
                signal2 = random.choice(augmented_signals)
                augmented_signals.append(mixup(signal, signal2, params.get('alpha', 0.2)))
                augmented_labels.append(label)  # Using same label for mixup
        else:
            augmented_signals.append(apply_augmentation(signal, aug_type, params))
            augmented_labels.append(label)
    
    return np.array(augmented_signals), np.array(augmented_labels) 