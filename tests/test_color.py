"""
Tests unitarios para cvtools.color
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cvtools.color import rgb_to_hsv, rgb_to_lab, quantize_image, reduce_image_weight


class TestColorConversions(unittest.TestCase):

    def setUp(self):
        # Imagen sintética BGR de 50x50 con colores variados
        np.random.seed(0)
        self.image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    def test_rgb_to_hsv_shape(self):
        """La imagen HSV debe tener la misma forma que la original."""
        result = rgb_to_hsv(self.image)
        self.assertEqual(result.shape, self.image.shape)

    def test_rgb_to_hsv_dtype(self):
        """La imagen HSV debe ser uint8."""
        result = rgb_to_hsv(self.image)
        self.assertEqual(result.dtype, np.uint8)

    def test_rgb_to_lab_shape(self):
        """La imagen LAB debe tener la misma forma que la original."""
        result = rgb_to_lab(self.image)
        self.assertEqual(result.shape, self.image.shape)

    def test_rgb_to_lab_dtype(self):
        """La imagen LAB debe ser uint8."""
        result = rgb_to_lab(self.image)
        self.assertEqual(result.dtype, np.uint8)

    def test_hsv_values_in_range(self):
        """Valores HSV deben estar en rangos válidos de OpenCV (H:0-179, S:0-255, V:0-255)."""
        result = rgb_to_hsv(self.image)
        self.assertLessEqual(result[:, :, 0].max(), 179)  # H canal
        self.assertLessEqual(result[:, :, 1].max(), 255)
        self.assertLessEqual(result[:, :, 2].max(), 255)

    def test_lab_l_channel_in_range(self):
        """Canal L de LAB debe estar entre 0 y 255."""
        result = rgb_to_lab(self.image)
        self.assertGreaterEqual(result[:, :, 0].min(), 0)
        self.assertLessEqual(result[:, :, 0].max(), 255)


class TestQuantizeImage(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.image = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    def test_output_shape(self):
        """La imagen cuantizada debe tener la misma forma."""
        result = quantize_image(self.image, n_colors=16)
        self.assertEqual(result.shape, self.image.shape)

    def test_output_dtype(self):
        """La imagen cuantizada debe ser uint8."""
        result = quantize_image(self.image, n_colors=16)
        self.assertEqual(result.dtype, np.uint8)

    def test_reduced_unique_colors(self):
        """El número de colores únicos debe ser ≤ n_colors."""
        n_colors = 8
        result = quantize_image(self.image, n_colors=n_colors)
        pixels = result.reshape(-1, 3)
        unique_colors = len(set(map(tuple, pixels)))
        self.assertLessEqual(unique_colors, n_colors)


class TestReduceImageWeight(unittest.TestCase):

    def setUp(self):
        np.random.seed(7)
        self.image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_returns_tuple(self):
        """Debe retornar una tupla (imagen, kb)."""
        result = reduce_image_weight(self.image, n_colors=16)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_quantized_image_shape(self):
        """La imagen retornada debe tener la misma forma."""
        q_img, _ = reduce_image_weight(self.image, n_colors=16)
        self.assertEqual(q_img.shape, self.image.shape)

    def test_size_kb_positive(self):
        """El tamaño en KB debe ser un número positivo."""
        _, size_kb = reduce_image_weight(self.image, n_colors=16)
        self.assertGreater(size_kb, 0)

    def test_size_kb_type(self):
        """El tamaño en KB debe ser un número flotante."""
        _, size_kb = reduce_image_weight(self.image, n_colors=16)
        self.assertIsInstance(size_kb, float)


if __name__ == '__main__':
    unittest.main()
