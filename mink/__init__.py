"""mink: MuJoCo inverse kinematics."""

from .configuration import Configuration
from .constants import SUPPORTED_FRAMES, FRAME_TO_ENUM, FRAME_TO_JAC_FUNC, FRAME_TO_POS_ATTR, FRAME_TO_XMAT_ATTR
from .exceptions import (
    MinkError,
    UnsupportedFrame,
    InvalidFrame,
    InvalidKeyframe,
    NotWithinConfigurationLimits,
    InvalidMocapBody,
)
from .lie import SE3, SO3, MatrixLieGroup
from .limits import (
    Limit,
    ConfigurationLimit,
    VelocityLimit,
    CollisionAvoidanceLimit,
    Constraint,
)
from .solve_ik import build_ik, solve_ik
from .tasks import (
    Task,
    Objective,
    PostureTask,
    FrameTask,
    ComTask,
    DampingTask,
    TargetNotSet,
)
from .utils import (
    custom_configuration_vector,
    get_body_geom_ids,
    get_freejoint_dims,
    get_subtree_geom_ids,
    move_mocap_to_frame,
)

__version__ = "0.0.2"

__all__ = (
    "ComTask",
    "Configuration",
    "build_ik",
    "solve_ik",
    "DampingTask",
    "FrameTask",
    "PostureTask",
    "Task",
    "Objective",
    "ConfigurationLimit",
    "VelocityLimit",
    "CollisionAvoidanceLimit",
    "Constraint",
    "Limit",
    "SO3",
    "SE3",
    "MatrixLieGroup",
    "MinkError",
    "UnsupportedFrame",
    "InvalidFrame",
    "InvalidKeyframe",
    "NotWithinConfigurationLimits",
    "TargetNotSet",
    "InvalidMocapBody",
    "SUPPORTED_FRAMES",
    "FRAME_TO_ENUM",
    "FRAME_TO_JAC_FUNC",
    "FRAME_TO_POS_ATTR",
    "FRAME_TO_XMAT_ATTR",
    "move_mocap_to_frame",
    "custom_configuration_vector",
    "get_freejoint_dims",
    "get_subtree_geom_ids",
    "get_body_geom_ids",
)