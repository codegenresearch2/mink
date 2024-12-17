"""Build and solve the inverse kinematics problem."""

from typing import Optional, Sequence

import mujoco
import numpy as np

from .configuration import Configuration
from .limits import ConfigurationLimit, Limit
from .mjqp import Problem
from .tasks import Objective, Task


def _compute_qp_objective(
    configuration: Configuration, tasks: Sequence[Task], damping: float
) -> Objective:
    H = np.eye(configuration.nv) * damping
    c = np.zeros(configuration.nv)
    for task in tasks:
        H_task, c_task = task.compute_qp_objective(configuration)
        H += H_task
        c += c_task
    return Objective(H, c)


def _compute_qp_inequalities(
    configuration: Configuration, limits: Optional[Sequence[Limit]], dt: float
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if limits is None:
        limits = [ConfigurationLimit(configuration.model)]
    lower_list: list[np.ndarray] = []
    upper_list: list[np.ndarray] = []
    for limit in limits:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        if not inequality.inactive:
            assert (
                inequality.lower is not None and inequality.upper is not None
            )  # mypy.
            lower_list.append(inequality.lower)
            upper_list.append(inequality.upper)
    if not lower_list:
        lower = np.full(configuration.nv, -mujoco.mjMAXVAL)
        upper = np.full(configuration.nv, mujoco.mjMAXVAL)
    else:
        lower = np.max(np.vstack(lower_list), axis=0)
        upper = np.min(np.vstack(upper_list), axis=0)
    return lower, upper


def _process_inequality_constraints(
    P: np.ndarray,
    q: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    configuration: Configuration,
    inequality_constraints: Sequence[Limit],
    dt: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not inequality_constraints:
        return P, q, lower, upper

    # Combine all inequality constraints.
    A_list = []
    b_list = []
    for limit in inequality_constraints:
        inequality = limit.compute_qp_inequalities(configuration, dt)
        assert inequality.is_inequality_constraint
        A_list.append(inequality.G)
        b_list.append(inequality.h)
    A = np.vstack(A_list)
    b = np.hstack(b_list)

    # Shape checks.
    n = P.shape[0]
    n_ineq = A.shape[0]
    assert A.shape[1] == n
    assert b.shape[0] == n_ineq

    # Construct penalty terms.
    P_new = eta * np.block([[A.T @ A, A.T], [A, np.eye(n_ineq)]])
    q_new = np.concatenate([-eta * A.T @ b, -eta * b], axis=0)
    # P_new is (n + n_ineq, n + n_ineq).
    # q_new is (n + n_ineq,).

    # Incorporate the original objective.
    P_new[:n, :n] += P
    q_new[:n] += q

    # Update bounds.
    lower_new = np.hstack([lower, np.zeros(n_ineq)])
    upper_new = np.hstack([upper, np.full(n_ineq, mujoco.mjMAXVAL)])

    return P_new, q_new, lower_new, upper_new


def build_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    limits: Optional[Sequence[Limit]] = None,
) -> Problem:
    """Build quadratic program from current configuration and tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        damping: Levenberg-Marquardt damping.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.

    Returns:
        Quadratic program of the inverse kinematics problem.
    """
    # Process box constraints.
    box_constraints = [c for c in limits if c.constraint_type == "box"]
    lower, upper = _compute_qp_inequalities(configuration, box_constraints, dt)

    # Process task objectives.
    P, q = _compute_qp_objective(configuration, tasks, damping)

    # Transform inequality constraints into a box-constrained QP.
    P, q, lower, upper = _process_inequality_constraints(
        P,
        q,
        lower,
        upper,
        configuration,
        [c for c in limits if c.constraint_type == "inequality"],
        dt,
        eta=50000.0,
    )
    return Problem.initialize(configuration.nv, P, q, lower, upper)


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    safety_break: bool = False,
    limits: Optional[Sequence[Limit]] = None,
) -> np.ndarray:
    """Solve the differential inverse kinematics problem.

    Computes a velocity tangent to the current robot configuration. The computed
    velocity satisfies at (weighted) best the set of provided kinematic tasks.

    Args:
        configuration: Robot configuration.
        tasks: List of kinematic tasks.
        dt: Integration timestep in [s].
        solver: Backend quadratic programming (QP) solver.
        damping: Levenberg-Marquardt damping.
        safety_break: If True, stop execution and raise an exception if
            the current configuration is outside limits. If False, print a
            warning and continue execution.
        limits: List of limits to enforce. Set to empty list to disable. If None,
            defaults to a configuration limit.
        kwargs: Keyword arguments to forward to the backend QP solver.

    Returns:
        Velocity `v` in tangent space.
    """
    configuration.check_limits(safety_break=safety_break)
    problem = build_ik(configuration, tasks, dt, damping, limits)
    dq = problem.solve()
    v: np.ndarray = dq / dt
    return v
