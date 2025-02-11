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
        cls.model = load_robot_description("g1_mj_description")

    def setUp(self):
        self.configuration = Configuration(self.model)
        self.configuration.update_from_keyframe("stand")
        # NOTE(kevin): These velocities are arbitrary and do not match real hardware.
        self.velocities = {
            self.model.joint(i).name: np.pi for i in range(1, self.model.njnt)
        }

    def test_dimensions(self):
        """Test the dimensions of the velocity limit."""
        limit = VelocityLimit(self.model, self.velocities)
        nv = self.configuration.nv
        nb = nv - 6  # Freejoint (0-5) is not limited.
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))

    def test_indices(self):
        """Test the indices of the velocity limit."""
        limit = VelocityLimit(self.model, self.velocities)
        expected = np.arange(6, self.model.nv)  # Freejoint (0-5) is not limited.
        np.testing.assert_allclose(limit.indices, expected)

    def test_model_with_no_limit(self):
        """Verify that VelocityLimit behaves correctly when no limits are defined."""
        empty_model = mujoco.MjModel.from_xml_string("<mujoco></mujoco>")
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.projection_matrix)
        G, h = empty_bounded.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNone(G)
        self.assertIsNone(h)

    def test_model_with_subset_of_velocities_limited(self):
        """Test the velocity limit with a subset of velocities."""
        limit_subset = {}
        for i, (key, value) in enumerate(self.velocities.items()):
            if i > 2:
                break
            limit_subset[key] = value
        limit = VelocityLimit(self.model, limit_subset)
        nb = 3
        nv = self.model.nv
        self.assertEqual(limit.projection_matrix.shape, (nb, nv))
        self.assertEqual(len(limit.indices), nb)
        expected_limit = np.asarray([np.pi] * nb)
        np.testing.assert_allclose(limit.limit, expected_limit)

        # Check the shapes of G and h after compute_qp_inequalities
        G, h = limit.compute_qp_inequalities(self.configuration, 1e-3)
        self.assertIsNotNone(G)
        self.assertIsNotNone(h)
        self.assertEqual(G.shape, (2 * nb, nv))
        self.assertEqual(h.shape, (2 * nb,))

    def test_model_with_ball_joint(self):
        """Test the velocity limit with a ball joint."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="ball" name="ball"/>
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
            "ball": (np.pi, np.pi / 2, np.pi / 4),
            "hinge": (0.5,),
        }
        limit = VelocityLimit(model, velocities)
        nb = 3 + 1
        self.assertEqual(len(limit.indices), nb)
        self.assertEqual(limit.projection_matrix.shape, (nb, model.nv))

    def test_ball_joint_invalid_limit_shape(self):
        """Test the velocity limit with an invalid limit shape for a ball joint."""
        xml_str = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="ball" name="ball"/>
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
            "ball": (np.pi, np.pi / 2),
        }
        with self.assertRaises(LimitDefinitionError) as cm:
            VelocityLimit(model, velocities)
        expected_error_message = "Joint ball must have a limit of shape (3,). Got: (2,)"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_that_freejoint_raises_error(self):
        """Test that a free joint raises an error."""
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