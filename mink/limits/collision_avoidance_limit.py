"""Collision avoidance limit."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
Geom = Union[int, str]
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class Contact:
    """Representation of a contact between two geoms."""
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Compute the normal vector of the contact."""
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """Determine if the contact is inactive."""
        return self.dist == self.distmax and not self.fromto.any()


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact: Contact,
) -> np.ndarray:
    """Compute the contact normal Jacobian."""
    geom1_body = model.geom_bodyid[contact.geom1]
    geom2_body = model.geom_bodyid[contact.geom2]
    geom1_contact_pos = contact.fromto[:3]
    geom2_contact_pos = contact.fromto[3:]
    jac2 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac2, None, geom2_contact_pos, geom2_body)
    jac1 = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac1, None, geom1_contact_pos, geom1_body)
    return contact.normal @ (jac2 - jac1)


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Check if two geoms are welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the bodies of two geoms are parent-child."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]
    return body_weldid1 == weld_parent_weldid2 or body_weldid2 == weld_parent_weldid1


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Check if the geoms pass the contype-conaffinity check."""
    return bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit."""
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        """Compute the QP inequalities for collision avoidance."""
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            contact = self._compute_contact_with_minimum_distance(
                configuration.data, geom1_id, geom2_id
            )
            if contact.inactive:
                continue
            hi_bound_dist = contact.dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation
            jac = compute_contact_normal_jacobian(
                self.model, configuration.data, contact
            )
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> Contact:
        """Compute the contact with minimum distance."""
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.collision_detection_distance,
            fromto,
        )
        return Contact(
            dist, fromto, geom1_id, geom2_id, self.collision_detection_distance
        )

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:
        """Homogenize a list of geoms to IDs."""
        return [self.model.geom(g).id if isinstance(g, str) else g for g in geom_list]

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        """Convert collision pairs to geom ID pairs."""
        return [
            (tuple(set(self._homogenize_geom_id_list(pair[0]))), tuple(set(self._homogenize_geom_id_list(pair[1]))))
            for pair in collision_pairs
        ]

    def _construct_geom_id_pairs(self, geom_pairs):
        """Construct geom ID pairs for all possible collisions."""
        return [
            (min(geom_a, geom_b), max(geom_a, geom_b))
            for pair in self._collision_pairs_to_geom_id_pairs(geom_pairs)
            for geom_a, geom_b in itertools.product(*pair)
            if not (_is_welded_together(self.model, geom_a, geom_b) or _are_geom_bodies_parent_child(self.model, geom_a, geom_b) or not _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b))
        ]