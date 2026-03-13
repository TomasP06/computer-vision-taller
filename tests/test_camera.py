"""
Tests unitarios para cvtools.camera
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cvtools.camera import apply_radial_distortion, vary_focal_length


class TestApplyRadialDistortion(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            [0.1, 0.2],
            [0.5, -0.3],
            [-0.4, 0.6],
            [0.0, 0.0],
        ], dtype=np.float64)

    def test_output_shape(self):
        """El resultado debe tener la misma forma que la entrada."""
        result = apply_radial_distortion(self.points, k1=0.1, k2=0.01)
        self.assertEqual(result.shape, self.points.shape)

    def test_no_distortion_when_k_zero(self):
        """Con k1=k2=0, los puntos no deben cambiar."""
        result = apply_radial_distortion(self.points, k1=0.0, k2=0.0)
        np.testing.assert_array_almost_equal(result, self.points)

    def test_origin_unchanged(self):
        """El origen (0,0) debe permanecer en (0,0) con cualquier k."""
        origin = np.array([[0.0, 0.0]])
        result = apply_radial_distortion(origin, k1=0.5, k2=0.1)
        np.testing.assert_array_almost_equal(result, origin)

    def test_barrel_distortion_moves_outward(self):
        """Distorsión en barril (k1>0) debe mover puntos hacia afuera."""
        pts = np.array([[0.5, 0.0]])
        result = apply_radial_distortion(pts, k1=0.5, k2=0.0)
        self.assertGreater(abs(result[0, 0]), abs(pts[0, 0]))

    def test_pincushion_distortion_moves_inward(self):
        """Distorsión en cojín (k1<0) debe mover puntos hacia adentro."""
        pts = np.array([[0.5, 0.0]])
        result = apply_radial_distortion(pts, k1=-0.5, k2=0.0)
        self.assertLess(abs(result[0, 0]), abs(pts[0, 0]))


class TestVaryFocalLength(unittest.TestCase):

    def setUp(self):
        self.points_3d = np.array([
            [1.0, 2.0, 5.0],
            [-1.0, 0.5, 3.0],
            [0.0, 0.0, 4.0],
        ], dtype=np.float64)
        self.focal_lengths = [100, 300, 700]

    def test_output_length(self):
        """Debe retornar una lista del mismo largo que focal_lengths."""
        result = vary_focal_length(self.points_3d, self.focal_lengths)
        self.assertEqual(len(result), len(self.focal_lengths))

    def test_each_projection_shape(self):
        """Cada proyección debe tener forma (N, 2)."""
        result = vary_focal_length(self.points_3d, self.focal_lengths)
        for proj in result:
            self.assertEqual(proj.shape, (len(self.points_3d), 2))

    def test_larger_focal_larger_projection(self):
        """Mayor focal length debe dar proyección más amplia (mayor valor de u)."""
        pt = np.array([[1.0, 0.0, 5.0]])
        result = vary_focal_length(pt, [100, 500])
        self.assertGreater(abs(result[1][0, 0]), abs(result[0][0, 0]))

    def test_center_point_at_principal(self):
        """Un punto en el eje óptico debe proyectar en el punto principal (cx, cy)."""
        pt = np.array([[0.0, 0.0, 10.0]])
        cx, cy = 320.0, 240.0
        result = vary_focal_length(pt, [500], cx=cx, cy=cy)
        np.testing.assert_array_almost_equal(result[0], [[cx, cy]])


if __name__ == '__main__':
    unittest.main()
