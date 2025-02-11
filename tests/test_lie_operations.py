"""Tests for general operation definitions."""

from typing import Type

import numpy as np
from absl.testing import absltest, parameterized

from mink.lie.base import MatrixLieGroup
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3

from .utils import assert_transforms_close


def check_inverse_bijective(group: Type[MatrixLieGroup]):
    """Check inverse of inverse."""
    transform = group.sample_uniform()
    assert_transforms_close(transform, transform.inverse().inverse())


def check_matrix_bijective(group: Type[MatrixLieGroup]):
    """Check that we can convert to and from matrices."""
    transform = group.sample_uniform()
    assert_transforms_close(transform, group.from_matrix(transform.as_matrix()))


def check_log_exp_bijective(group: Type[MatrixLieGroup]):
    """Check 1-to-1 mapping for log <=> exp operations."""
    transform = group.sample_uniform()

    tangent = transform.log()
    self.assertEqual(tangent.shape, (group.tangent_dim,))

    exp_transform = group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    np.testing.assert_allclose(tangent, exp_transform.log())


def check_adjoint(group: Type[MatrixLieGroup]):
    transform = group.sample_uniform()
    omega = np.random.randn(group.tangent_dim)
    assert_transforms_close(
        transform @ group.exp(omega),
        group.exp(transform.adjoint() @ omega) @ transform,
    )


def check_rminus(group: Type[MatrixLieGroup]):
    T_a = group.sample_uniform()
    T_b = group.sample_uniform()
    T_c = T_a.inverse() @ T_b
    np.testing.assert_allclose(T_b.rminus(T_a), T_c.log())


def check_lminus(group: Type[MatrixLieGroup]):
    T_a = group.sample_uniform()
    T_b = group.sample_uniform()
    np.testing.assert_allclose(T_a.lminus(T_b), (T_a @ T_b.inverse()).log())


def check_rplus(group: Type[MatrixLieGroup]):
    T_a = group.sample_uniform()
    T_b = group.sample_uniform()
    T_c = T_a.inverse() @ T_b
    assert_transforms_close(T_a.rplus(T_c.log()), T_b)


def check_lplus(group: Type[MatrixLieGroup]):
    T_a = group.sample_uniform()
    T_b = group.sample_uniform()
    T_c = T_a @ T_b.inverse()
    assert_transforms_close(T_b.lplus(T_c.log()), T_a)


def check_jlog(group: Type[MatrixLieGroup]):
    state = group.sample_uniform()
    w = np.random.rand(state.tangent_dim) * 1e-4
    state_pert = state.plus(w).log()
    state_lin = state.log() + state.jlog() @ w
    np.testing.assert_allclose(state_pert, state_lin, atol=1e-7)


def check_so3_rpy_bijective():
    T = SO3.sample_uniform()
    assert_transforms_close(T, SO3.from_rpy_radians(*T.as_rpy_radians()))


@parameterized.named_parameters(
    ("SO3", SO3),
    ("SE3", SE3),
)
class TestOperations(parameterized.TestCase):
    def test_inverse_bijective(self, group: Type[MatrixLieGroup]):
        check_inverse_bijective(group)

    def test_matrix_bijective(self, group: Type[MatrixLieGroup]):
        check_matrix_bijective(group)

    def test_log_exp_bijective(self, group: Type[MatrixLieGroup]):
        check_log_exp_bijective(group)

    def test_adjoint(self, group: Type[MatrixLieGroup]):
        check_adjoint(group)

    def test_rminus(self, group: Type[MatrixLieGroup]):
        check_rminus(group)

    def test_lminus(self, group: Type[MatrixLieGroup]):
        check_lminus(group)

    def test_rplus(self, group: Type[MatrixLieGroup]):
        check_rplus(group)

    def test_lplus(self, group: Type[MatrixLieGroup]):
        check_lplus(group)

    def test_jlog(self, group: Type[MatrixLieGroup]):
        check_jlog(group)


class TestGroupSpecificOperations(absltest.TestCase):
    """Group specific tests."""

    def test_so3_rpy_bijective(self):
        check_so3_rpy_bijective()


if __name__ == "__main__":
    absltest.main()