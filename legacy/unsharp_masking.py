# Original coursework script, kept unchanged for reference.
# The maintained versions are in sharpening/filters.py; see README "Changes from the coursework code".

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image. Check file path.")
    return img

def apply_unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, alpha=1.5):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return sharpened

def process_with_unsharp_mask(image_path):
    original = load_image(image_path)
    
    if original is None:
        return

    sharp1 = apply_unsharp_mask(original, kernel_size=(3,3), sigma=0.5, alpha=1.0)
    sharp2 = apply_unsharp_mask(original, kernel_size=(5,5), sigma=1.0, alpha=1.5)
    sharp3 = apply_unsharp_mask(original, kernel_size=(9,9), sigma=2.0, alpha=2.0)

    images = [original, sharp1, sharp2, sharp3]
    titles = ['Original', 'Unsharp 3x3 α=1.0', 'Unsharp 5x5 α=1.5', 'Unsharp 9x9 α=2.0']

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Example usage
process_with_unsharp_mask("image_path.jpg")
