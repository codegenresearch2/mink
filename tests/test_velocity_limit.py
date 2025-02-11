"""Tests for velocity_limit.py."""

import mujoco
import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.limits import LimitDefinitionError, VelocityLimit
from mink.utils import get_freejoint_dims


class TestVelocityLimit(absltest.TestCase):
    """Test velocity limit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")  # Changed to "stand"
        self.velocities = {
            self.model.joint(i).name: [3.14] * self.model.joint(i).dof
            for i in range(self.model.njnt)
        }

    def test_throws_error_if_gain_invalid(self):
        with self.assertRaises(LimitDefinitionError):
            VelocityLimit(self.model, gain=-1)
        with self.assertRaises(LimitDefinitionError):
            VelocityLimit(self.model, gain=1.1)

    def test_dimensions(self):
        limit = VelocityLimit(self.model, self.velocities)
        nv = self.configuration.nv
        nb = nv - len(get_freejoint_dims(self.model)[1])
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        limit = VelocityLimit(self.model, self.velocities)
        expected_indices = np.arange(6, self.model.nv)  # Freejoint (0-5) is not limited.
        self.assertTrue(np.array_equal(limit.indices, expected_indices))

    def test_model_with_no_limit(self):
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)
        G, h = empty_bounded.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNone(G)
        self.assertIsNone(h)

    def test_model_with_subset_of_velocities_limited(self):
        velocities = {
            self.model.joint(i).name: [3.14] * self.model.joint(i).dof
            for i in range(self.model.njnt) if self.model.joint(i).dof == 1
        }
        limit = VelocityLimit(self.model, velocities)
        nb = sum(self.model.joint(i).dof == 1 for i in range(self.model.njnt))
        nv = self.model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)

    def test_that_freejoint_raises_error(self):
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free" name="floating"/>
              <geom type="sphere" size=".1" mass=".1"/>
              <body>
                <joint type="hinge" name="hinge" range="0 1.57"/>
                <geom type="sphere" size=".1" mass=".1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml_str)
        velocities = {
            "floating": np.pi,
            "hinge": np.pi,
        }
        with self.assertRaises(LimitDefinitionError):
            VelocityLimit(model, velocities)

    def test_indices_of_limited_velocities(self):
        velocities = {
            self.model.joint(i).name: [3.14] * self.model.joint(i).dof
            for i in range(self.model.njnt) if self.model.joint(i).dof == 1
        }
        limit = VelocityLimit(self.model, velocities)
        expected_indices = np.array([j.id for j in self.model.joints if j.dof == 1])
        self.assertTrue(np.array_equal(limit.indices, expected_indices))


if __name__ == "__main__":
    absltest.main()