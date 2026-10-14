# Original coursework script, kept unchanged for reference.
# The maintained versions are in sharpening/filters.py; see README "Changes from the coursework code".

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image. Check file path.")
    return img

def apply_butterworth_high_pass_filter(img, cutoff=100, order=1):
    img_float = np.float32(img)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)

    H = 1 / (1 + (cutoff / (D + 1e-5))**(2 * order))  # Smoother Butterworth high-pass mask
    H = np.repeat(H[:, :, np.newaxis], 2, axis=2)

    filtered_shift = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

def process_high_pass(image_path):
    original = load_image(image_path)

    if original is None:
        return

    high_passed = apply_butterworth_high_pass_filter(original, cutoff=10, order=1)

    # Display results
        # Blend high-pass result with original
    alpha = 0.5  # adjust this to control sharpness
    enhanced = cv2.addWeighted(original, 1.0, high_passed, alpha, 0)

    # Display results
    images = [original, high_passed, enhanced]
    titles = ['Original', 'High-Pass Only', 'High-Pass + Original (Enhanced)']

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# Example usage
process_high_pass("image_scanned.jpg")
