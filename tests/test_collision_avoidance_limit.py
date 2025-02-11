"""Tests for configuration_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

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
        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        g1_coll = [
            g for g in g1 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        g2_coll = [
            g for g in g2 if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertTrue(np.all(h >= bound_relaxation))
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        """Test if the contact normal and Jacobian match the MuJoCo implementation."""
        # Set MuJoCo options for contact normal and Jacobian computation
        self.model.opt.cone = 0
        self.model.opt.jacobian = 1
        self.model.opt.enableflags = 0

        g1 = get_body_geom_ids(self.model, self.model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(self.model, self.model.body("upper_arm_link").id)

        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Initialize MuJoCo data object
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)

        # Compute contact normal Jacobian using the limit
        G_mujoco, h_mujoco = limit.compute_contact_normal_jacobian(data, bound_relaxation)

        # Reset MuJoCo options to default
        self.model.opt.cone = 1
        self.model.opt.jacobian = 0
        self.model.opt.enableflags = 1

        # Recreate the limit with the updated model options
        limit_reset = CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        G, h = limit_reset.compute_qp_inequalities(self.configuration, 1e-3)

        # Compare the results with the MuJoCo implementation
        np.testing.assert_allclose(G, G_mujoco, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(h, h_mujoco, rtol=1e-05, atol=1e-08)


if __name__ == "__main__":
    absltest.main()