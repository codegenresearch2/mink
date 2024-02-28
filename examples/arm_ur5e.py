import mujoco
import mujoco.viewer
import numpy as np
import mink
from pathlib import Path
import time

_HERE = Path(__file__).resolve().parent
_XML_PATH = _HERE / "universal_robots_ur5e" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
    data = mujoco.MjData(model)

    dt = 0.002
    model.opt.timestep = dt

    model.body_gravcomp[:] = 1.0

    joints = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    keyframe_name = "home"
    configuration = mink.Configuration.initialize_from_keyframe(
        model=model, data=data, keyframe_name=keyframe_name
    )

    #
    # Tasks.
    #

    end_effector_task = mink.FrameTask.initialize(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_task = mink.PostureTask.initialize(cost=1e-3)
    posture_task.set_target(model.key(keyframe_name).qpos)

    tasks = [
        end_effector_task,
        posture_task,
    ]

    #
    # Limits.
    #

    configuration_limit = mink.ConfigurationLimit.initialize(
        model=model,
        joints=joints,
        limit_gain=0.5,
    )

    velocity_limit = mink.VelocityLimit.initialize(
        model=model,
        joint2limit={
            "shoulder_pan": np.pi,
            "shoulder_lift": np.pi,
            "elbow": np.pi,
            "wrist_1": np.pi,
            "wrist_2": np.pi,
            "wrist_3": np.pi,
        },
    )

    limits = [
        configuration_limit,
        # NOTE(kevin): The velocity limit slows down the convergence enormously
        # and the behavior make the robot move each joint in a sequence rather than
        # all at once. Commenting out until we can figure out why.
        # velocity_limit,
    ]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe_name).id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Update task target.
            end_effector_target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=data.mocap_quat[0]),
                translation=data.mocap_pos[0],
            )
            end_effector_task.set_target(end_effector_target)

            # Solve IK.
            dq = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                limits=limits,
                dt=dt,
                damping=1e-8,
            )
            data.ctrl[:] = configuration.integrate(dq, dt)

            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
