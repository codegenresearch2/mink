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
        cls.model = load_robot_description("g1_mj_description")  # Changed to "g1_mj_description"

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("home")  # Changed to "home"
        # NOTE(kevin): These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: 3.14 for i in range(1, self.model.njnt)
        }

    def test_ball_joint_invalid_limit_shape(self):
        """Test that invalid limit shape raises an error."""
        velocities = {
            "ball": (np.pi, np.pi / 2),
        }
        with self.assertRaises(LimitDefinitionError):
            VelocityLimit(self.model, velocities)

    def test_dimensions(self):
        limit = VelocityLimit(self.model, self.velocities)
        nv = self.configuration.nv
        free_joint_dims = get_freejoint_dims(self.model)[0]
        nb = nv - free_joint_dims
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        limit = VelocityLimit(self.model, self.velocities)
        free_joint_dims = get_freejoint_dims(self.model)[0]
        expected_indices = np.arange(free_joint_dims, self.model.nv)
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
        partial_velocities = {
            joint.name: 3.14 for joint in self.model.joint if joint.type == mujoco.mjtJoint.mjJNT_HINGE
        }
        limit = VelocityLimit(self.model, partial_velocities)
        nb = len(partial_velocities)
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
                <joint type="hinge" name="hinge" range="0 1.57" limited="true"/>
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
        with self.assertRaises(LimitDefinitionError) as cm:
            VelocityLimit(model, velocities)
        expected_error_message = "Free joint floating is not supported"
        self.assertEqual(str(cm.exception), expected_error_message)


if __name__ == "__main__":
    absltest.main()