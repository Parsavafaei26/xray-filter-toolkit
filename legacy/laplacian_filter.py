# Original coursework script, kept unchanged for reference.
# The maintained versions are in sharpening/filters.py; see README "Changes from the coursework code".

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from scipy.signal import wiener

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image. Check file path.")
    return img

def apply_laplacian_filter(img, ksize=3, scale=1.5):
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)  # Compute Laplacian
    laplacian = np.uint8(np.clip(scale * np.absolute(laplacian), 0, 255))  # Adjust intensity
    return laplacian

def process_with_laplacian(image_path):
    original = load_image(image_path)

    if original is None:
        print("Error loading image. Check the file path.")
        return

    laplacian_filtered = apply_laplacian_filter(original, ksize=3, scale=1.5)

    # Blend Laplacian result with original for gentle enhancement
    alpha = 0.5  # control strength
    enhanced = cv2.addWeighted(original, 1.0, laplacian_filtered, alpha, 0)

    # Display Results
    images = [original, laplacian_filtered, enhanced]
    titles = ['Original', 'Laplacian Only', 'Laplacian + Original (Enhanced)']

    plt.figure(figsize=(15,5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
process_with_laplacian("image_scanned.jpg")
