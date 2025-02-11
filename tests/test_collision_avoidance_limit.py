"""Tests for configuration_limit.py."""

import itertools
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description
from mujoco import MjModel, MjData, compute_contact_normal_jacobian
from mink.limits import CollisionAvoidanceLimit
from mink.utils import get_body_geom_ids

class Configuration:
    def __init__(self, model):
        self.model = model
        self.qpos = np.zeros(model.nv)

    def update_from_keyframe(self, keyframe_name):
        # Placeholder for actual implementation
        pass

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

        # Check that non-colliding geoms are correctly filtered out and that we have
        # the right number of max expected contacts.
        g1_coll = [
            g
            for g in g1
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        g2_coll = [
            g
            for g in g2
            if self.model.geom_conaffinity[g] != 0 and self.model.geom_contype[g] != 0
        ]
        expected_max_num_contacts = len(list(itertools.product(g1_coll, g2_coll)))
        self.assertEqual(limit.max_num_contacts, expected_max_num_contacts)

        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)

        # The upper bound should always be >= relaxation bound.
        self.assertTrue(np.all(h >= bound_relaxation))

        # Check that the inequality constraint dimensions are valid.
        self.assertEqual(G.shape, (expected_max_num_contacts, self.model.nv))
        self.assertEqual(h.shape, (expected_max_num_contacts,))

    def test_contact_normal_jac_matches_mujoco(self):
        # Set up the model with specific options
        model = self.model
        model.opt.cone = True
        model.opt.jacobian = True
        data = MjData(model)

        # Update the configuration
        self.configuration.update_from_keyframe("home")
        qpos = self.configuration.qpos
        data.qpos[:] = qpos

        # Get the geom IDs
        g1 = get_body_geom_ids(model, model.body("wrist_2_link").id)
        g2 = get_body_geom_ids(model, model.body("upper_arm_link").id)

        # Define the collision avoidance limit
        bound_relaxation = -1e-3
        limit = CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[(g1, g2)],
            bound_relaxation=bound_relaxation,
        )

        # Compute the contact normal Jacobian using MuJoCo
        contact_jacobians = []
        for geom1 in g1:
            for geom2 in g2:
                contact = Contact()
                contact.geom1 = geom1
                contact.geom2 = geom2
                contact_normal_jac = compute_contact_normal_jacobian(model, data, contact)
                contact_jacobians.append(contact_normal_jac)

        # Compute the contact normal Jacobian using the limit
        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)
        computed_jacobians = [G[i, :] for i in range(len(contact_jacobians))]

        # Compare the computed Jacobian with MuJoCo's contact normal Jacobian
        for computed_jac, contact_jac in zip(computed_jacobians, contact_jacobians):
            np.testing.assert_allclose(computed_jac, contact_jac, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    absltest.main()