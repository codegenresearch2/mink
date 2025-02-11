"""
This module contains the implementation of the CollisionAvoidanceLimit class.
The class is designed to handle collision avoidance between geom pairs in a MuJoCo model.
"""

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
class _Contact:
    """
    A dataclass to store contact information between two geoms.
    
    Attributes:
        dist (float): The signed distance between the two geoms.
        fromto (np.ndarray): An array containing the coordinates of the contact points.
        geom1 (int): The ID of the first geom.
        geom2 (int): The ID of the second geom.
        distmax (float): The maximum distance allowed between the geoms.
    """
    dist: float
    fromto: np.ndarray
    geom1: int
    geom2: int
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """
        Returns the normal vector of the contact.
        
        Returns:
            np.ndarray: The normal vector pointing from geom1 to geom2.
        """
        normal = self.fromto[3:] - self.fromto[:3]
        return normal / (np.linalg.norm(normal) + 1e-9)

    @property
    def inactive(self) -> bool:
        """
        Returns True if the contact is inactive.
        
        Returns:
            bool: True if the contact is inactive, False otherwise.
        """
        return self.dist == self.distmax and not self.fromto.any()


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """
    Returns true if the geoms are part of the same body, or if their bodies are welded together.
    
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        geom_id1 (int): The ID of the first geom.
        geom_id2 (int): The ID of the second geom.
    
    Returns:
        bool: True if the geoms are welded together, False otherwise.
    """
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """
    Returns true if the geom bodies have a parent-child relationship.
    
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        geom_id1 (int): The ID of the first geom.
        geom_id2 (int): The ID of the second geom.
    
    Returns:
        bool: True if the geom bodies are parent-child, False otherwise.
    """
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # body_weldid is the ID of the body's weld.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """
    Returns true if the geoms pass the contype/conaffinity check.
    
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        geom_id1 (int): The ID of the first geom.
        geom_id2 (int): The ID of the second geom.
    
    Returns:
        bool: True if the geoms pass the contype/conaffinity check, False otherwise.
    """
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """
    Normal velocity limit between geom pairs.
    
    Attributes:
        model (mujoco.MjModel): The MuJoCo model.
        geom_pairs (CollisionPairs): Set of collision pairs in which to perform active collision avoidance.
        gain (float): Gain factor that determines how fast the geoms are allowed to move towards each other.
        minimum_distance_from_collisions (float): The minimum distance to leave between any two geoms.
        collision_detection_distance (float): The distance between two geoms at which the active collision avoidance limit will be active.
        bound_relaxation (float): An offset on the upper bound of each collision avoidance constraint.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """
        Initialize collision avoidance limit.
        
        Args:
            model (mujoco.MjModel): The MuJoCo model.
            geom_pairs (CollisionPairs): Set of collision pairs in which to perform active collision avoidance.
            gain (float): Gain factor that determines how fast the geoms are allowed to move towards each other.
            minimum_distance_from_collisions (float): The minimum distance to leave between any two geoms.
            collision_detection_distance (float): The distance between two geoms at which the active collision avoidance limit will be active.
            bound_relaxation (float): An offset on the upper bound of each collision avoidance constraint.
        """
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
        """
        Compute the configuration-dependent joint position limits.
        
        Args:
            configuration (Configuration): The robot's configuration.
            dt (float): Integration timestep in seconds.
        
        Returns:
            Constraint: A tuple representing the inequality constraint as G * dq <= h.
        """
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
            jac = self._compute_contact_normal_jacobian(configuration.data, contact)
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.

    def _compute_contact_with_minimum_distance(
        self, data: mujoco.MjData, geom1_id: int, geom2_id: int
    ) -> _Contact:
        """
        Returns the smallest signed distance between a geom pair.
        
        Args:
            data (mujoco.MjData): The MuJoCo data object.
            geom1_id (int): The ID of the first geom.
            geom2_id (int): The ID of the second geom.
        
        Returns:
            _Contact: A dataclass instance containing contact information.
        """
        fromto = np.empty(6)
        dist = mujoco.mj_geomDistance(
            self.model,
            data,
            geom1_id,
            geom2_id,
            self.collision_detection_distance,
            fromto,
        )
        return _Contact(
            dist, fromto, geom1_id, geom2_id, self.collision_detection_distance
        )

    def _compute_contact_normal_jacobian(
        self, data: mujoco.MjData, contact: _Contact
    ) -> np.ndarray:
        """
        Computes the Jacobian mapping joint velocities to the normal component of the relative Cartesian linear velocity between the geom pair.
        
        Args:
            data (mujoco.MjData): The MuJoCo data object.
            contact (_Contact): A dataclass instance containing contact information.
        
        Returns:
            np.ndarray: The Jacobian matrix.
        """
        geom1_body = self.model.geom_bodyid[contact.geom1]
        geom2_body = self.model.geom_bodyid[contact.geom2]
        geom1_contact_pos = contact.fromto[:3]
        geom2_contact_pos = contact.fromto[3:]
        jac2 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac2, None, geom2_contact_pos, geom2_body)
        jac1 = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jac1, None, geom1_contact_pos, geom1_body)
        return contact.normal @ (jac2 - jac1)

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> List[int]:
        """
        Take a heterogeneous list of geoms (specified via ID or name) and return a homogenous list of IDs (int).
        
        Args:
            geom_list (GeomSequence): A list of geoms specified via ID or name.
        
        Returns:
            List[int]: A list of geom IDs.
        """
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            else:
                assert isinstance(g, str)
                list_of_int.append(self.model.geom(g).id)
        return list_of_int

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """
        Returns a set of geom ID pairs for all possible geom-geom collisions.
        
        Args:
            geom_pairs (CollisionPairs): Set of collision pairs.
        
        Returns:
            List[tuple[int, int]]: A list of geom ID pairs.
        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                weld_body_cond = not _is_welded_together(self.model, geom_a, geom_b)
                parent_child_cond = not _are_geom_bodies_parent_child(self.model, geom_a, geom_b)
                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(self.model, geom_a, geom_b)
                if weld_body_cond and parent_child_cond and contype_conaffinity_cond:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        return geom_id_pairs

This revised code snippet addresses the feedback provided by the oracle. It includes improved docstring formatting, detailed attribute descriptions, method documentation, and ensures consistent whitespace usage and method definitions. Additionally, it adds comments to clarify complex logic and includes appropriate type hints.