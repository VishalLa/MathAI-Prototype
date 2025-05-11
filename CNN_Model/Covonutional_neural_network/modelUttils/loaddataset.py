import os 
import cv2
import torch
import numpy as np
from PIL import Image
from .augment_dataset import apply_combined_noise, add_gaussian_noise


label_to_index = {
    '0': 0,
    '1': 1, 
    '2': 2, 
    '3': 3, 
    '4': 4, 
    '5': 5, 
    '6': 6, 
    '7': 7, 
    '8': 8, 
    '9': 9, 
    'add': 10,
    'dec': 11, 
    'div': 12, 
    'eq': 13, 
    'mul': 14,
    'sub': 15,
    # '(': 16, 
    # ')': 17, 
    'x': 16,  
    'y': 17, 
    # 'z': 20,
}

# Reverse mapping for predictions 

index_to_label = {v: k for k, v in label_to_index.items()}

def add_padding(image: np.ndarray, stride:int=28)-> np.ndarray:
    '''
    Adding padding to the image to make it a square with dimensions that are multiples of stride.
    '''

    h, w = image.shape[:2]
    max_side = max(h, w)
    new_size = ((max_side + stride - 1) // stride) * stride
    padded_image = np.ones((new_size, new_size), dtype=np.uint8) * 255
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return padded_image


def load_mnist_dataset(path: str)-> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    image = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)

    return image, label



def load_dataset_for_image(folder_path: str, target_size: tuple[int, int] = (28, 28))-> tuple[np.ndarray, np.ndarray]:
    
    custom_images, custom_labels = [], []

    if not os.path.exists(folder_path):
        print(f'Folder {folder_path} dose not exist!')
        return np.array(custom_images), np.array(custom_labels)
    
    for filename in os.listdir(folder_path):

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)


            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)
                img_array = cv2.convertScaleAbs(img_array) if img_array.dtype != np.uint8 else img_array

                _, binary_image = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # dected edge
                lower_thresh = max(0, 0.5*np.median(binary_image))
                upper_thersh = min(255, 1.5*np.median(binary_image))
                edges = cv2.Canny(binary_image, lower_thresh, upper_thersh)


                # Check if edges are detected
                if np.sum(edges) == 0:
                    print(f"No edges detected in {file_path}. Skipping this image.")
                    char_image_resized = cv2.resize(binary_image, target_size, interpolation=cv2.INTER_AREA)

                else:
                    x, y, w, h = cv2.boundingRect(edges)
                    char_image_resized = cv2.resize(binary_image, target_size, interpolation=cv2.INTER_AREA)
                    char_image = edges[y:y+h, x:x+w]
                    char_image = add_padding(char_image, stride=4)
                    char_image_resized = cv2.resize(char_image, target_size, interpolation=cv2.INTER_AREA)
                
                
                # Rotate 10% of images randomly
                if np.random.rand() < 0.1:
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((char_image_resized.shape[1] // 2, char_image_resized.shape[0] // 2), angle, 1)
                    char_image_resized = cv2.warpAffine(
                        char_image_resized,
                        M,
                        (char_image_resized.shape[1], char_image_resized.shape[0]), 
                        borderValue=255
                    )

                # Check if the image is of proper size
                if char_image_resized.shape != target_size:
                    char_image_resized = cv2.resize(char_image_resized, target_size, interpolation=cv2.INTER_AREA)

                char_image_resized = cv2.bitwise_not(char_image_resized)
                char_image = char_image_resized.astype(np.float32) / 255.0

                
                if '-' in filename:
                    label_str = filename.split('-')[0]
                    if label_str not in label_to_index:
                        print(f'Skipping image {file_path}: Label {label_str} not in label_to_index')
                        continue

                    numeric_label = label_to_index[label_str]
                    # binary_image = add_label_noise(char_image, numeric_label)
                    noisy_image = add_gaussian_noise(char_image_resized)
                else:
                    print(f'Skipping image {file_path}: Unable to extract label')
                    continue

                if label_str not in label_to_index:
                    print(f"Skipping image {file_path}: Label '{label_str}' not in label_to_index")
                    continue
                
                # custom_images.append(char_image)
                custom_images.append(noisy_image)
                custom_labels.append(numeric_label)

            except Exception as e:
                print(f'Error loading image {file_path}: {type(e).__name__} - {e}')
                continue
    return np.array(custom_images), np.array(custom_labels)



def load_dataset(folder_path: list[str])->tuple[np.ndarray, np.ndarray]:

    print('Loading Dataset .............')
    path_for_mnist = 'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\mnist.npz'
    mnist_images, mnist_labels = load_mnist_dataset(path_for_mnist)

    # Normalize MNIST images to [0, 1]
    mnist_images = mnist_images / 255.0

    # Add a channel dimension to MNIST images (for grayscale images)
    mnist_images = mnist_images[:, np.newaxis, :, :]

    images = []
    labels = []

    images.append(mnist_images)
    labels.append(mnist_labels)

    for path in folder_path:
        if not os.path.exists(path):
            print(f"Skipping paths: {path} (Path does not exist)")
            continue
        
        custom_images, custom_labels = load_dataset_for_image(path)

        # Skip if no valid images were found
        if len(custom_images) == 0:
            print(f"No valid images found in folder: {path}")
            continue

        # Add a channel dimension to custom images (for grayscale images)
        custom_images = np.expand_dims(np.array(custom_images), axis=1)

        # Append the images and labels
        images.append(custom_images)
        labels.append(custom_labels)
    
    # Combine all images and labels
    combined_images = np.concatenate(images, axis=0)
    combined_labels = np.concatenate(labels, axis=0)

    label_count = {}
    for i in range(20):
        label_count[i] = len(combined_images[combined_labels==i])

    # Convert from NumPy to Tensor
    combined_images = torch.tensor(combined_images, dtype=torch.float32)
    combined_labels = torch.tensor(combined_labels, dtype=torch.long)   

    print(f'Shape of imags are: {combined_images.shape}') 

    print(f'Total images lodded: {combined_images.shape[0]}')

    print('\n')
    for i, value in label_count.items():
        print(f'Label {i}: {value} number of images')

    print('\n')
    print('Dataset loaded successfully!')

    return combined_images, combined_labels

