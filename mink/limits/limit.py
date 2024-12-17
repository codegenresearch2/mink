"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import NamedTuple, Optional

import numpy as np

from ..configuration import Configuration


class Constraint(NamedTuple):
    """Box constraint of the form lower <= x <= upper."""

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    G: Optional[np.ndarray] = None
    h: Optional[np.ndarray] = None

    @property
    def inactive(self) -> bool:
        """Returns True if the constraint is inactive."""
        return (
            self.G is None
            and self.h is None
            and self.lower is None
            and self.upper is None
        )

    @property
    def is_box_constraint(self) -> bool:
        """Returns True if the constraint is a box constraint."""
        is_box = self.lower is not None and self.upper is not None
        return is_box and not self.is_inequality_constraint

    @property
    def is_inequality_constraint(self) -> bool:
        """Returns True if the constraint is an inequality constraint."""
        is_inequality = self.G is not None and self.h is not None
        return is_inequality and not self.is_box_constraint


class Limit(abc.ABC):
    """Abstract base class for kinematic limits.

    Subclasses must implement the :py:meth:`~Limit.compute_qp_inequalities` method
    which takes in the current robot configuration and integration time step and
    returns an instance of :class:`Constraint`.
    """

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        r"""Compute limit as linearized QP inequalities of the form:

        .. math::

            G(q) \Delta q \leq h(q)

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration time step in [s].

        Returns:
            Pair :math:`(G, h)`.
        """
        raise NotImplementedError
