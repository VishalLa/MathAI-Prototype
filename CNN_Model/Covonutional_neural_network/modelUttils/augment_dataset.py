import os 
import cv2 
import random
import numpy as np 


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
    