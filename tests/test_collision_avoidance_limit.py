"""Tests for configuration_limit.py."""

import itertools

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids


class TestCollisionAvoidanceLimit(absltest.TestCase):
    """Test collision avoidance limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")

    def test_dimensions(self):
        wrist_2_geom_ids = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        upper_arm_geom_ids = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(wrist_2_geom_ids, upper_arm_geom_ids)],
            bound_relaxation=bound_relaxation,
        )

        # Check that non-colliding geoms are correctly filtered out and that we have
        # the right number of max expected contacts.
        colliding_wrist_2_geom_ids = [
            g for g in wrist_2_geom_ids if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        colliding_upper_arm_geom_ids = [
            g for g in upper_arm_geom_ids if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_num_contacts = len(list(itertools.product(colliding_wrist_2_geom_ids, colliding_upper_arm_geom_ids)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # The upper bound should always be >= relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check that the inequality constraint dimensions are valid.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))


if __name__ == "__main__":
    absltest.main()