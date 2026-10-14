"""Image sharpening filters for 8-bit greyscale images.

Each filter takes a uint8 image and returns a uint8 image of the same size.
Arithmetic is done in float64 and clipped to 0-255 only at the end, so that
negative values in the edge detail are not lost before they are added back.
"""

import cv2
import numpy as np


def _to_float(img):
    return img.astype(np.float64)


def _to_uint8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- blurring

def gaussian_blur(img, ksize=5, sigma=1.0):
    """Low-pass Gaussian filter, B = I * G. Used to simulate scanner blur."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ---------------------------------------------------------------- Laplacian

def laplacian_sharpen(img, c=1.0):
    """g = f - c * lap(f), using the 4-neighbour kernel [0 1 0; 1 -4 1; 0 1 0].

    The kernel's centre is negative, so the Laplacian is *subtracted* to
    sharpen. The sign is kept: taking the absolute value (as the original
    coursework code did) brightens both sides of an edge instead of
    steepening it.
    """
    f = _to_float(img)
    lap = cv2.Laplacian(f, cv2.CV_64F, ksize=1)
    return _to_uint8(f - c * lap)


# ---------------------------------------------------------------- unsharp masking

def unsharp_mask(img, ksize=5, sigma=1.0, k=1.5):
    """I' = I + k (I - B), where B is a Gaussian-blurred copy of I.

    k = 1 is classic unsharp masking; k > 1 is high-boost filtering.
    """
    f = _to_float(img)
    blurred = cv2.GaussianBlur(f, (ksize, ksize), sigma)
    return _to_uint8(f + k * (f - blurred))


# ---------------------------------------------------------------- frequency domain

def butterworth_highpass(shape, d0=30.0, order=2):
    """Butterworth high-pass transfer function H(u, v) = 1 / (1 + (D0 / D)^(2n)),
    centred (DC in the middle of the array)."""
    rows, cols = shape
    u = np.arange(rows) - rows / 2
    v = np.arange(cols) - cols / 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)
    with np.errstate(divide="ignore"):
        H = 1.0 / (1.0 + (d0 / D) ** (2 * order))
    H[D == 0] = 0.0
    return H


def high_frequency_emphasis(img, d0=30.0, order=2, k=1.0):
    """G = F^-1{ [1 + k * H_hp(u, v)] F(u, v) }.

    Adds a scaled high-pass copy back to the image in the frequency domain.
    Keeps the low frequencies (overall brightness and contrast) and boosts
    the high ones (edges). Signed values are kept throughout.
    """
    f = _to_float(img)
    F = np.fft.fftshift(np.fft.fft2(f))
    H = butterworth_highpass(f.shape, d0, order)
    g = np.real(np.fft.ifft2(np.fft.ifftshift((1.0 + k * H) * F)))
    return _to_uint8(g)


# ---------------------------------------------------------------- registry

METHODS = {
    "laplacian": laplacian_sharpen,
    "unsharp": unsharp_mask,
    "highpass": high_frequency_emphasis,
}
