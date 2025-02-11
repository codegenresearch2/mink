from typing import Tuple
import mujoco
import numpy as np

from . import constants as consts
from .exceptions import InvalidKeyframe, InvalidMocapBody


def get_freejoint_dims(model: mujoco.MjModel) -> Tuple[list[int], list[int]]:
    """
    Get all floating joint configuration and tangent indices.

    Args:
        model (mujoco.MjModel): The Mujoco model.

    Returns:
        Tuple[list[int], list[int]]: A tuple containing two lists:
            - The first list contains the indices of all free joints in the configuration space.
            - The second list contains the indices of all free joints in the tangent space.
    """
    q_ids: list[int] = []
    v_ids: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            vadr = model.jnt_dofadr[j]
            q_ids.extend(range(qadr, qadr + 7))
            v_ids.extend(range(vadr, vadr + 6))
    return q_ids, v_ids


def custom_configuration_vector(
    model: mujoco.MjModel,
    key_name: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate a configuration vector where named joints have specific values.

    Args:
        model (mujoco.MjModel): The Mujoco model.
        key_name (Optional[str]): Optional keyframe name to initialize the configuration vector from.
            Otherwise, the default pose `qpos0` is used.
        kwargs: Custom values for joint coordinates.

    Returns:
        np.ndarray: Configuration vector where named joints have the values specified in
            keyword arguments, and other joints have their neutral value or value
            defined in the keyframe if provided.
    """
    data = mujoco.MjData(model)
    if key_name is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise InvalidKeyframe(key_name, model)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    q = data.qpos.copy()
    for name, value in kwargs.items():
        jid = model.joint(name).id
        jnt_dim = consts.qpos_width(model.jnt_type[jid])
        qid = model.jnt_qposadr[jid]
        value = np.atleast_1d(value)
        if value.shape != (jnt_dim,):
            raise ValueError(
                f"Joint {name} should have a qpos value of {jnt_dim} but "
                f"got {value.shape}"
            )
        q[qid : qid + jnt_dim] = value
    return q


This revised code snippet addresses the feedback by correcting the syntax error in the comment, ensuring that the function names and structure are consistent, and adding comprehensive docstrings to improve readability and maintainability.