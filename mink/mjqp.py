"""Convenience wrapper around the MuJoCo QP solver."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import mujoco

from .exceptions import IKFailure
from .configuration import Configuration

import logging


@dataclass(frozen=True)
class Problem:
    """Wrapper over `mujoco.mju_boxQP`."""

    H: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n: int
    R: np.ndarray
    index: np.ndarray

    @staticmethod
    def initialize(
        configuration: Configuration,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ):
        n = configuration.nv
        R = np.zeros((n, n + 7))
        index = np.zeros(n, np.int32)
        return Problem(H, c, lower, upper, n, R, index)

    def solve(self, prev_sol: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if prev_sol is not None:
            logging.debug("Using previous solution as initial guess.")
            dq = prev_sol
            assert dq.shape == (self.n,)
        else:
            dq = np.empty((self.n,))
        rank = mujoco.mju_boxQP(
            res=dq,
            R=self.R,
            index=self.index,
            H=self.H,
            g=self.c,
            lower=self.lower,
            upper=self.upper,
        )
        if rank == -1:
            raise IKFailure("QP solver failed to find a solution.")
        dq[np.isnan(dq)] = 0.0
        return dq
