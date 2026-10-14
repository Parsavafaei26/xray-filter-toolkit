"""Image quality measures.

Two kinds:
  * full-reference (PSNR, SSIM): need a sharp reference image;
  * no-reference (edge strength, background noise, clipping): work on the
    real blurred scan, where no sharp reference exists.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference, test):
    return peak_signal_noise_ratio(reference, test, data_range=255)


def ssim(reference, test):
    return structural_similarity(reference, test, data_range=255)


def edge_strength(img, mask=None):
    """Mean Sobel gradient magnitude (Tenengrad). Higher = sharper edges."""
    f = img.astype(np.float64)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(mag[mask].mean() if mask is not None else mag.mean())


def background_noise(img, mask):
    """Standard deviation of the fine detail (image minus a 5x5 median)
    inside a flat background region. Higher = more noise amplified."""
    f = img.astype(np.float64)
    detail = f - cv2.medianBlur(img, 5).astype(np.float64)
    return float(detail[mask].std())


def clipped_fraction(img):
    """Share of pixels pushed to pure black or pure white (0 or 255)."""
    return float(np.mean((img == 0) | (img == 255)))


def flat_background_mask(img, threshold=20, margin=15):
    """Dark, flat background of an X-ray: below threshold, eroded away from
    every edge so that edge ringing is not counted as noise."""
    dark = (cv2.GaussianBlur(img, (9, 9), 3) < threshold).astype(np.uint8)
    kernel = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    return cv2.erode(dark, kernel).astype(bool)
