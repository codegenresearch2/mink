"""Tests for configuration.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink


class TestConfiguration(absltest.TestCase):
    """Test task various configuration methods work as intended."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")
        cls.q_ref = cls.model.key("home").qpos

    def setUp(self):
        self.configuration = mink.Configuration(self.model)

    def test_nq_nv(self):
        """Test that nq and nv are correctly set."""
        self.assertEqual(self.configuration.nq, self.model.nq)
        self.assertEqual(self.configuration.nv, self.model.nv)

    def test_initialize_from_keyframe(self):
        """Test initialization from keyframe and from qpos."""
        self.configuration.update_from_keyframe("home")
        np.testing.assert_array_equal(self.configuration.q, self.q_ref)

        # Initialize from qpos
        qpos = np.zeros(self.model.nq)
        self.configuration.update(q=qpos)
        np.testing.assert_array_equal(self.configuration.q, qpos)

    def test_site_transform_world_frame(self):
        site_name = "attachment_site"
        np.random.seed(12345)
        qpos = np.random.uniform(*self.model.jnt_range.T)
        self.configuration.update(q=qpos)
        world_T_site = self.configuration.get_transform_frame_to_world(site_name, "site")
        expected_translation = self.configuration.data.site(site_name).xpos
        np.testing.assert_array_equal(world_T_site.translation(), expected_translation)
        expected_xmat = self.configuration.data.site(site_name).xmat.reshape(3, 3)
        np.testing.assert_almost_equal(world_T_site.rotation().as_matrix(), expected_xmat)

    def test_site_transform_raises_error_if_frame_name_is_invalid(self):
        """Raise an error when the requested frame does not exist."""
        with self.assertRaises(mink.InvalidFrame):
            self.configuration.get_transform_frame_to_world("invalid_name", "site")

    def test_site_transform_raises_error_if_frame_type_is_invalid(self):
        """Raise an error when the requested frame type is invalid."""
        with self.assertRaises(mink.UnsupportedFrame):
            self.configuration.get_transform_frame_to_world("name_does_not_matter", "joint")

    def test_update_raises_error_if_keyframe_is_invalid(self):
        """Raise an error when the request keyframe does not exist."""
        with self.assertRaises(mink.InvalidKeyframe):
            self.configuration.update_from_keyframe("invalid_keyframe")

    def test_inplace_integration(self):
        """Test inplace integration of velocities."""
        qvel = np.ones((self.model.nv))
        expected_qpos = self.q_ref + 1e-3 * qvel
        self.configuration.integrate_inplace(qvel, 1e-3)
        np.testing.assert_almost_equal(self.configuration.q, expected_qpos)

    def test_check_limits(self):
        """Check that an error is raised iff a joint limit is exceeded."""
        self.configuration.update(q=self.q_ref)
        self.configuration.check_limits()
        self.q_ref[0] += 1e4  # Move configuration out of bounds.
        self.configuration.update(q=self.q_ref)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            self.configuration.check_limits()


if __name__ == "__main__":
    absltest.main()