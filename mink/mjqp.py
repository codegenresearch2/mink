"""Convenience wrapper around the MuJoCo QP solver."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Problem:
    H: np.ndarray
    c: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    R: np.ndarray
    n: int
    noriginal: int

    @staticmethod
    def initialize(
        noriginal: int,
        H: np.ndarray,
        c: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Problem:
        n = H.shape[0]
        R = np.zeros((n, n + 7))
        return Problem(H, c, lower, upper, R, n, noriginal)

    def solve(self) -> np.ndarray:
        dq = np.empty((self.n,))
        rank = mujoco.mju_boxQP(
            res=dq,
            R=self.R,
            index=None,
            H=self.H,
            g=self.c,
            lower=self.lower,
            upper=self.upper,
        )
        if rank == -1:
            return np.zeros((self.noriginal,))
        return dq[: self.noriginal]
