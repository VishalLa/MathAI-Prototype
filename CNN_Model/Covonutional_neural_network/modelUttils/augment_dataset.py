import os 
import cv2 
import random
import numpy as np 


def add_label_noise(image: np.ndarray, label: int)->np.ndarray:
    """
    Apply label-specific Gaussian noise to the input image.

    Args:
        image (np.ndarray): Input image in range [0, 1].
        label (int): Integer label from 0 to 19.

    Returns:
        np.ndarray: Noisy image clipped to [0, 1].
    """
    image = image.astype(np.float32) / 255.0

    h, w = image.shape

    x = np.linspace(0, np.pi * (label + 1), w)
    y = np.linspace(0, np.pi * (label + 1), h)
    xv, yv = np.meshgrid(x, y)
    sinuosidal_noise = 0.1 * np.sin(xv+yv)

    means = np.linspace(0 , 0.3, 21)
    stds = np.linspace(0.05, 0.15, 21)
    noise = np.random.normal(loc=means[label], scale=stds[label], size=image.shape)

    total_noise = noise + sinuosidal_noise
    
    noisy_image = image + total_noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return (noisy_image*255).astype(np.uint8)

