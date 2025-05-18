import numpy as np 
from scipy.ndimage import gaussian_filter, map_coordinates


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

    x = np.linspace(0, np.pi * (label + 1)*0.05, w)
    y = np.linspace(0, np.pi * (label + 1)*0.05, h)
    xv, yv = np.meshgrid(x, y)
    phase_shift = label*np.pi/100
    sinuosidal_noise = 0.05 * np.sin(xv+yv+phase_shift)

    stds = np.linspace(0.00002, 0.00005, 18)
    noise = np.random.normal(loc=0.0, scale=stds[label], size=image.shape)

    total_noise = noise + sinuosidal_noise
    
    noisy_image = image + total_noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    return (noisy_image*255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.01, std: float = 0.02) -> np.ndarray:
    """
    Apply Gaussian noise to the input image.

    Args:
        image (np.ndarray): Input image in range [0, 1].
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy image clipped to [0, 1].
    """

    image = image.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return (noisy_image*255).astype(np.uint8)   


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    noisy = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Add salt (white) noise
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255

    # Add pepper (black) noise
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy


def elastic_transform(image, alpha=36, sigma=6):
    assert image.ndim == 2

    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted.astype(np.uint8)


def apply_combined_noise(image):
    image = add_salt_pepper_noise(image)
    image = elastic_transform(image, alpha=34, sigma=5)
    return image

