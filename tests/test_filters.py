"""Checks that each filter behaves as the theory says it should.
Run from the repository root:  python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sharpening import filters as F                      # noqa: E402
from sharpening import metrics as M                      # noqa: E402


def step_edge(size=64, low=60, high=180):
    img = np.full((size, size), low, np.uint8)
    img[:, size // 2:] = high
    return img


def blurred_edge():
    return F.gaussian_blur(step_edge(), 9, 2.0)


class TestFlatImage(unittest.TestCase):
    """A flat image has no edges, so no method should change it."""

    def setUp(self):
        self.flat = np.full((64, 64), 120, np.uint8)

    def test_methods_leave_flat_image_unchanged(self):
        for name, fn in F.METHODS.items():
            with self.subTest(method=name):
                out = fn(self.flat)
                self.assertTrue(np.array_equal(out, self.flat))


class TestSharpening(unittest.TestCase):

    def test_methods_steepen_a_blurred_edge(self):
        img = blurred_edge()
        before = M.edge_strength(img)
        for name, fn in F.METHODS.items():
            with self.subTest(method=name):
                self.assertGreater(M.edge_strength(fn(img)), before)

    def test_unsharp_overshoots_on_both_sides(self):
        img = blurred_edge()
        out = F.unsharp_mask(img, ksize=9, sigma=2.0, k=1.5)
        row = out[32].astype(int)
        self.assertLess(row.min(), 60)     # darker than the dark side: undershoot
        self.assertGreater(row.max(), 180) # brighter than the bright side: overshoot

    def test_stronger_k_gives_stronger_edges(self):
        img = blurred_edge()
        weak = M.edge_strength(F.unsharp_mask(img, 5, 1.0, 0.5))
        strong = M.edge_strength(F.unsharp_mask(img, 5, 1.0, 2.0))
        self.assertGreater(strong, weak)

    def test_mean_brightness_is_kept(self):
        img = blurred_edge()
        for name, fn in F.METHODS.items():
            with self.subTest(method=name):
                self.assertAlmostEqual(fn(img).mean(), img.mean(), delta=2.0)

    def test_output_is_uint8_same_shape(self):
        img = blurred_edge()
        for name, fn in F.METHODS.items():
            with self.subTest(method=name):
                out = fn(img)
                self.assertEqual(out.dtype, np.uint8)
                self.assertEqual(out.shape, img.shape)


class TestButterworth(unittest.TestCase):

    def test_blocks_dc_and_passes_high_frequencies(self):
        H = F.butterworth_highpass((128, 128), d0=20, order=2)
        self.assertEqual(H[64, 64], 0.0)            # centre (DC) removed
        self.assertGreater(H[0, 0], 0.99)           # far corner passed
        self.assertAlmostEqual(H[64, 84], 0.5, 2)   # half power at D = D0


class TestMetrics(unittest.TestCase):

    def test_identical_images_score_perfectly(self):
        img = blurred_edge()
        self.assertEqual(M.ssim(img, img), 1.0)
        with np.errstate(divide="ignore"):
            self.assertEqual(M.psnr(img, img), float("inf"))

    def test_clipped_fraction(self):
        img = np.array([[0, 128], [255, 10]], np.uint8)
        self.assertAlmostEqual(M.clipped_fraction(img), 0.5)


if __name__ == "__main__":
    unittest.main()
