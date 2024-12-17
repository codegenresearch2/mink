"""Convenience wrapper around the MuJoCo QP solver."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from .exceptions import IKFailure


class Problem:
    def __init__(
        self,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        noriginal: int,
    ):
        self.n = H.shape[0]
        self.H = H
        self.c = c
        self.lower = lower
        self.upper = upper
        self.R = np.zeros((self.n, self.n + 7))
        self.index = np.zeros(self.n, np.int32)

        self.noriginal = noriginal
        self._prev_sol = None

    def solve(self, warmstart: bool = False) -> Optional[np.ndarray]:
        if warmstart and self._prev_sol is not None:
            dq = self._prev_sol
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

        self._prev_sol = dq
        return dq[: self.noriginal]
