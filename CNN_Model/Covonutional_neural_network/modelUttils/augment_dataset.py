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
    y = np.linspace(0, np.pi * (label+1), h)
    xv, yv = np.meshgrid(x, y)
    sinuosidal_noise = 0.1 * np.sin(xv+yv)

    means = np.linspace(0 , 0.3, 20)
    stds = np.linspace(0.05, 0.15, 20)
    noise = np.random.normal(loc=means[label], scale=stds[label], size=image.shape)

    total_noise = noise + sinuosidal_noise
    
    noisy_image = image + total_noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return (noisy_image*255).astype(np.uint8)



def add_gaussian_noise(image, mean=0, std_range=(5, 15)):
    std = np.random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noise_image = cv2.add(image, noise)
    return noise_image


def add_salt_papper_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = int(amount*image.size+0.5)
    num_pepper = int(amount*image.size*0.5)

    # Salt
    coords = tuple(np.random.randint(0, i-1, num_salt) for i in image.shape)
    noisy[coords] = 255

    # Pepper
    coords = tuple(np.random.randint(0, i - 1, num_pepper) for i in image.shape)
    noisy[coords] = 0

    return noisy


def affinee_transform(image, max_angle=10, scale_range=(0.9,1.1)):
    rows, cols = image.shape
    angle = np.random.uniform(-max_angle, max_angle)
    scale = np.random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    return cv2.warpAffine(image, M, (cols, rows), borderValue=0)


def augment_image(image):
    choice = random.choice(['gaussian', 'salt_papper', 'affine'])
    
    if choice == 'gaussian':
        return add_gaussian_noise(image)
    elif choice == 'salt_pepper':
        return add_salt_papper_noise(image)
    elif choice == 'affine':
        return affinee_transform(image)
    else:
        return image 
    