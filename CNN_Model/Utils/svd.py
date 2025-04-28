import cv2 
import numpy as np 


def svd(canvas_array, k=50):
    # Step 1: Convert to grayscale if RGB
    if len(canvas_array.shape) == 3 and canvas_array.shape[2] == 3:
        gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_array

    print(f'[INFO] Input shape: {gray.shape}')

    # Step 2: Normalize image to [0, 1]
    gray = gray / 255.0

    # Step 3: Apply SVD
    U, S, VT = np.linalg.svd(gray, full_matrices=False)
    compressed_img = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

    # Step 4: Clip to [0, 1] and convert to uint8
    compressed_img = np.clip(compressed_img, 0, 1)
    compressed_img_uint8 = (compressed_img * 255).astype(np.uint8)

    cv2.imwrite('svd_compressed.png', compressed_img_uint8)

    # Step 5: Resize to 28x28
    resized_img = cv2.resize(compressed_img_uint8, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite('resized_image.png', resized_img)

    print(f'[INFO] Max pixel after resize: {resized_img.max()}')
    print(f'[INFO] Min pixel after resize: {resized_img.min()}')

    # Step 6: Apply thresholding
    _, binary_image = cv2.threshold(resized_img, 180, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('binary_image_svd.png', binary_image)

    return binary_image




# img = cv2.imread("C:\\Users\\visha\\OneDrive\\Documents\\images\\im2.jpg", cv2.IMREAD_GRAYSCALE)

# img = img / 255.0


# U, S, VT = np.linalg.svd(img, full_matrices=False)

# k = 50 
# compressed_img = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

# compressed_img = np.clip(compressed_img*255, 0, 255).astype(np.uint8)

# # mn_val = compressed_img.min()

# cv2.imwrite('svd_compressed.png', compressed_img)

# np_img = np.asarray(compressed_img)


# resized_img = cv2.resize(compressed_img, (28, 28), interpolation=cv2.INTER_AREA)
# cv2.imwrite('resized_image.png', resized_img)

# print(np_img.shape)

