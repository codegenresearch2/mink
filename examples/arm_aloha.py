from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
from typing import Sequence, Optional

_HERE = Path(__file__).parent
_XML = _HERE / "aloha" / "scene.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_ros_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

def compensate_gravity(model: mujoco.MjModel, data: mujoco.MjData, subtree_ids: Sequence[int], grav: np.ndarray = np.array([0, 0, -9.81]), qfrc_applied: np.ndarray | None = None) -> None:
    """
    Computes forces to counteract gravity for specific subtrees.
    
    Args:
        model (mujoco.MjModel): The Mujoco model object.
        data (mujoco.MjData): The Mujoco data object.
        subtree_ids (Sequence[int]): List of subtree IDs for which gravity compensation is applied.
        grav (np.ndarray, optional): Gravitational acceleration vector. Defaults to np.array([0, 0, -9.81]).
        qfrc_applied (np.ndarray | None, optional): Array to store the computed forces. If None, the forces are applied directly to the data. Defaults to None.
    """
    if qfrc_applied is None:
        qfrc_applied = data.qfrc_applied
    qfrc_applied[:] = 0  # Reset the qfrc_applied array to avoid unintended accumulation
    for subtree_id in subtree_ids:
        total_mass = 0
        jac = np.zeros((6, model.nv))
        mujoco.mj_jacSubtree(model, data.mocap_pos, subtree_id, jac)
        for i in range(jac.shape[1]):
            total_mass += model.body(model.joint(i).parent).mass
        gravity_compensation = -grav * total_mass
        qfrc_applied[model.joint_subtree(subtree_id)[0].dofadr] = gravity_compensation

def construct_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(_XML.as_posix())

def construct_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)

def get_joint_and_actuator_ids(model: mujoco.MjModel) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, float]]:
    joint_names = []
    velocity_limits = {}
    for prefix in ["left", "right"]:
        for n in _JOINT_NAMES:
            name = f"{prefix}/{n}"
            joint_names.append(name)
            velocity_limits[name] = _VELOCITY_LIMITS[n]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    return joint_names, dof_ids, actuator_ids, velocity_limits

def construct_tasks(model: mujoco.MjModel) -> list[mink.FrameTask]:
    l_ee_task = mink.FrameTask(
        frame_name="left/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    r_ee_task = mink.FrameTask(
        frame_name="right/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=1e-4)
    return [l_ee_task, r_ee_task, posture_task]

def construct_collision_pairs(model: mujoco.MjModel) -> list[tuple[np.ndarray, list[str]]]:
    l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
    r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
    l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
    r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
    frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
    collision_pairs = [
        (l_wrist_geoms, r_wrist_geoms),
        (l_geoms + r_geoms, frame_geoms + ["table"]),
    ]
    return collision_pairs

def construct_limits(model: mujoco.MjModel, joint_names: list[str], velocity_limits: dict[str, float], collision_pairs: list[tuple[np.ndarray, list[str]]]) -> list[mink.Limit]:
    config_limit = mink.ConfigurationLimit(model=model)
    velocity_limit = mink.VelocityLimit(model, velocity_limits)
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )
    return [config_limit, velocity_limit, collision_avoidance_limit]

if __name__ == "__main__":
    model = construct_model()
    data = construct_data(model)
    joint_names, dof_ids, actuator_ids, velocity_limits = get_joint_and_actuator_ids(model)
    tasks = construct_tasks(model)
    collision_pairs = construct_collision_pairs(model)
    limits = construct_limits(model, joint_names, velocity_limits, collision_pairs)

    l_mid = model.body("left/target").mocapid[0]
    r_mid = model.body("right/target").mocapid[0]
    solver = "quadprog"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5  # Adjusted to match the gold code

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
        configuration = mink.Configuration(model)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task = tasks[2]
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

        rate = RateLimiter(frequency=200.0)
        while viewer.is_running():
            # Update task targets.
            l_ee_task = tasks[0]
            r_ee_task = tasks[1]
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    limits=limits,
                    damping=1e-5,
                )
                configuration.integrate_inplace(vel, rate.dt)

                l_err = l_ee_task.compute_error(configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= ori_threshold
                r_err = r_ee_task.compute_error(configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= ori_threshold
                if (
                    l_pos_achieved
                    and l_ori_achieved
                    and r_pos_achieved
                    and r_ori_achieved
                ):
                    break

            compensate_gravity(model, data, [model.body("left/wrist_link").id, model.body("right/wrist_link").id])
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()