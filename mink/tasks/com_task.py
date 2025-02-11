"""Center-of-mass task implementation."""

from typing import Optional

import mujoco
import numpy as np

from ..configuration import Configuration
from .exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from .task import Task


class ComTask(Task):
    """Regulate the center-of-mass (CoM) of a robot.

    Attributes:
        target_com: Target position of the CoM.
    """

    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if cost < 0.0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")

        super().__init__(
            cost=np.full((3,), cost),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_com = None

    def set_target(self, target_com: np.ndarray) -> None:
        """Set the target CoM position in the world frame.

        Args:
            target_com: Desired center-of-mass position in the world frame.
        """
        if target_com.shape != (3,):
            raise InvalidTarget(
                f"Expected target CoM to have shape (3,) but got {target_com.shape}"
            )
        self.target_com = target_com.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target CoM from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(configuration.data.subtree_com[1])

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task error.

        The error is defined as:

        .. math::

            e(q) = c^* - c

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Center-of-mass task error vector :math:`e(q)`.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        return configuration.data.subtree_com[1] - self.target_com

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{3 \times n_v}` is the
        derivative of the CoM position with respect to the current configuration
        :math:`q`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Center-of-mass task jacobian :math:`J(q)`.
        """
        if self.target_com is None:
            raise TargetNotSet(self.__class__.__name__)
        jac = np.empty((3, configuration.nv))
        mujoco.mj_jacSubtreeCom(configuration.model, configuration.data, jac, 1)
        return jac