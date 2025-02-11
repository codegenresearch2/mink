"""Tests utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_value_error_if_shape_invalid(self):
        with self.assertRaises(ValueError):
            utils.skew(np.zeros((5,)))

    def test_skew_equals_negative(self):
        vec = np.random.randn(3)
        skew_matrix = utils.skew(vec)
        np.testing.assert_allclose(skew_matrix.T, -skew_matrix)

    def test_skew_with_valid_input(self):
        vec = np.array([1, 2, 3])
        expected_skew = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        np.testing.assert_allclose(utils.skew(vec), expected_skew)

    def test_skew_with_zero_vector(self):
        vec = np.zeros(3)
        expected_skew = np.zeros((3, 3))
        np.testing.assert_allclose(utils.skew(vec), expected_skew)

    def test_skew_with_large_input(self):
        vec = np.random.randn(10)
        with self.assertRaises(ValueError):
            utils.skew(vec)

    def test_skew_with_negative_input(self):
        vec = -np.random.randn(3)
        skew_matrix = utils.skew(vec)
        np.testing.assert_allclose(skew_matrix.T, -skew_matrix)


if __name__ == "__main__":
    absltest.main()