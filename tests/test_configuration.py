"""Tests for configuration.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

import mink


class TestConfiguration(absltest.TestCase):
    """Test various configuration methods work as intended."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_description("ur5e_mj_description")

    def setUp(self):
        self.q_ref = self.model.key("home").qpos

    def test_nq_nv(self):
        """Test that nq and nv are correctly set."""
        configuration = mink.Configuration(self.model)
        self.assertEqual(configuration.nq, self.model.nq)
        self.assertEqual(configuration.nv, self.model.nv)

    def test_initialize_from_keyframe(self):
        """Test initialization from keyframe."""
        configuration = mink.Configuration(self.model)
        configuration.update_from_keyframe("home")
        np.testing.assert_array_equal(configuration.q, self.q_ref)

    def test_initialize_from_qpos(self):
        """Test initialization from qpos."""
        qpos = np.zeros(self.model.nq)
        configuration = mink.Configuration(self.model)
        configuration.update(q=qpos)
        np.testing.assert_array_equal(configuration.q, qpos)

    def test_site_transform_world_frame(self):
        site_name = "attachment_site"
        qpos = np.random.uniform(*self.model.jnt_range.T)
        configuration = mink.Configuration(self.model)
        configuration.update(q=qpos)
        world_T_site = configuration.get_transform_frame_to_world(site_name, "site")
        expected_translation = configuration.data.site(site_name).xpos
        np.testing.assert_array_equal(world_T_site.translation(), expected_translation)
        expected_xmat = configuration.data.site(site_name).xmat.reshape(3, 3)
        np.testing.assert_almost_equal(world_T_site.rotation().as_matrix(), expected_xmat)

    def test_site_transform_raises_error_if_frame_name_is_invalid(self):
        """Raise an error when the requested frame does not exist."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidFrame):
            configuration.get_transform_frame_to_world("invalid_name", "site")

    def test_site_transform_raises_error_if_frame_type_is_invalid(self):
        """Raise an error when the requested frame type is invalid."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.UnsupportedFrame):
            configuration.get_transform_frame_to_world("name_does_not_matter", "joint")

    def test_update_raises_error_if_keyframe_is_invalid(self):
        """Raise an error when the request keyframe does not exist."""
        configuration = mink.Configuration(self.model)
        with self.assertRaises(mink.InvalidKeyframe):
            configuration.update_from_keyframe("invalid_keyframe")

    def test_inplace_integration(self):
        """Test inplace integration of velocities."""
        qvel = np.ones((self.model.nv))
        configuration = mink.Configuration(self.model)
        configuration.update(q=self.q_ref)
        expected_qpos = self.q_ref + 1e-3 * qvel
        configuration.integrate_inplace(qvel, 1e-3)
        np.testing.assert_almost_equal(configuration.q, expected_qpos)

    def test_check_limits(self):
        """Check that an error is raised iff a joint limit is exceeded."""
        configuration = mink.Configuration(self.model)
        configuration.update(q=self.q_ref)
        configuration.check_limits()
        q_ref = self.q_ref.copy()
        q_ref[0] += 1e4  # Move configuration out of bounds.
        configuration = mink.Configuration(self.model)
        configuration.update(q=q_ref)
        with self.assertRaises(mink.NotWithinConfigurationLimits):
            configuration.check_limits()


if __name__ == "__main__":
    absltest.main()