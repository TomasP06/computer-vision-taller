"""
Tests unitarios para cvtools.filters
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cvtools.filters import apply_convolution, sobel_x, sobel_y, canny_detector, laplacian_filter


class TestApplyConvolution(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def test_identity_kernel_unchanged(self):
        """El kernel identidad no debe cambiar la imagen."""
        identity = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.float32)
        result = apply_convolution(self.image, identity)
        np.testing.assert_array_equal(result, self.image)

    def test_output_shape(self):
        """La forma de salida debe ser igual a la entrada."""
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = apply_convolution(self.image, kernel)
        self.assertEqual(result.shape, self.image.shape)

    def test_output_dtype(self):
        """La salida debe ser uint8."""
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        result = apply_convolution(self.image, kernel)
        self.assertEqual(result.dtype, np.uint8)


class TestSobelFilters(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.image_gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        self.image_bgr = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    def test_sobel_x_shape_gray(self):
        """Sobel X con imagen gris debe retornar misma forma."""
        result = sobel_x(self.image_gray)
        self.assertEqual(result.shape, self.image_gray.shape)

    def test_sobel_x_shape_bgr(self):
        """Sobel X con imagen BGR debe retornar forma (H, W)."""
        result = sobel_x(self.image_bgr)
        self.assertEqual(result.shape, (80, 80))

    def test_sobel_y_shape_gray(self):
        """Sobel Y con imagen gris debe retornar misma forma."""
        result = sobel_y(self.image_gray)
        self.assertEqual(result.shape, self.image_gray.shape)

    def test_sobel_y_shape_bgr(self):
        """Sobel Y con imagen BGR debe retornar forma (H, W)."""
        result = sobel_y(self.image_bgr)
        self.assertEqual(result.shape, (80, 80))

    def test_sobel_uniform_image_is_zero(self):
        """En una imagen uniforme los bordes deben ser 0."""
        uniform = np.full((50, 50), 128, dtype=np.uint8)
        result_x = sobel_x(uniform)
        result_y = sobel_y(uniform)
        self.assertEqual(result_x.max(), 0)
        self.assertEqual(result_y.max(), 0)


class TestCannyDetector(unittest.TestCase):

    def setUp(self):
        np.random.seed(3)
        self.image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_output_shape(self):
        """La salida de Canny debe ser (H, W)."""
        result = canny_detector(self.image, low=50, high=150)
        self.assertEqual(result.shape, (100, 100))

    def test_output_is_binary(self):
        """La salida de Canny solo debe contener 0 y 255."""
        result = canny_detector(self.image, low=50, high=150)
        unique_vals = set(result.ravel().tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_output_dtype(self):
        """La salida de Canny debe ser uint8."""
        result = canny_detector(self.image)
        self.assertEqual(result.dtype, np.uint8)


class TestLaplacianFilter(unittest.TestCase):

    def setUp(self):
        np.random.seed(5)
        self.image = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    def test_output_shape(self):
        """El filtro Laplaciano debe retornar imagen de misma dimensión (H, W)."""
        result = laplacian_filter(self.image)
        self.assertEqual(result.shape, (80, 80))

    def test_output_dtype(self):
        """La salida del Laplaciano debe ser uint8."""
        result = laplacian_filter(self.image)
        self.assertEqual(result.dtype, np.uint8)

    def test_uniform_image_is_zero(self):
        """En imagen uniforme el Laplaciano debe ser 0."""
        uniform = np.full((50, 50), 100, dtype=np.uint8)
        result = laplacian_filter(uniform)
        self.assertEqual(result.max(), 0)


if __name__ == '__main__':
    unittest.main()
