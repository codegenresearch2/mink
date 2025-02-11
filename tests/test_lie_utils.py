"""Tests for utils.py."""

import numpy as np
from absl.testing import absltest

from mink.lie import utils


class TestUtils(absltest.TestCase):
    def test_skew_throws_assertion_error_if_shape_invalid(self):
        with self.assertRaises(AssertionError):
            utils.skew(np.zeros((5,)))

    def test_skew_transpose_equals_negative(self):
        vec = np.random.randn(3)
        skew_mat = utils.skew(vec)
        np.testing.assert_allclose(skew_mat.T, -skew_mat)


if __name__ == "__main__":
    absltest.main()