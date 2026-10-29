"""Pollen Robotics' Reachy Mini (experimental), converted from the official MJCF.

The USD is generated locally by ``tools/convert_assets.py`` (Isaac Lab's kit-less MJCF converter)
into ``assets_generated/reachy_mini/`` and is never committed: the Reachy Mini hardware files are
licensed CC BY-NC-SA (non-commercial).

Kinematics: a fixed foot, the ``yaw_body`` joint, six Stewart servos (``stewart_1..6``) each with
a rod on a passive ball joint (``passive_1..6``). The head (``xl_330``) hangs off rod 6 through the
ball joint ``passive_7``; the other five rods close their loops on the head through MuJoCo
``connect`` equalities (converted to ``PhysicsSphericalJoint`` with ``excludeFromArticulation``,
which Newton hands to MuJoCo-Warp as equality constraints). Two antennas sit on the head.
In the all-zero joint pose the loops are closed ("assembled zero pose").

Actuators: the real robot uses Dynamixel XL330 servos (stall torque ~0.5 N*m at 5 V). The
official MJCF uses idealised position servos (kp 10, +-20 N*m); here the actuated joints get an
XL330-like limit of 0.5 N*m and the passive ball joints get zero gains and no friction.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from .. import GENERATED_ASSETS_DIR

REACHY_MINI_USD_PATH = str(GENERATED_ASSETS_DIR / "reachy_mini" / "reachy_mini.usda")
"""Output of ``python tools/convert_assets.py reachy_mini``."""

REACHY_MINI_STEWART_JOINTS = [f"stewart_{i}" for i in range(1, 7)]
REACHY_MINI_ACTUATED_JOINTS = ["yaw_body", *REACHY_MINI_STEWART_JOINTS, "right_antenna", "left_antenna"]
REACHY_MINI_HEAD_BODY = "xl_330"

# Head (``xl_330``) pose at the assembled zero pose, in the foot (root) frame; quaternion xyzw.
# Measured with Newton forward kinematics on the converted USD (see tools/convert_assets.py --check).
REACHY_MINI_HEAD_ZERO_POS = (0.0206, -0.0218, 0.1496)
REACHY_MINI_HEAD_ZERO_QUAT = (0.7244, -0.1830, -0.4748, -0.4652)
# Head centre (the MJCF's ``Body`` frame on the head) at the zero pose, root frame.
REACHY_MINI_HEAD_CENTER_ZERO_POS = (0.0395, 0.0, 0.2021)

_XL330_STIFFNESS = 4.0  # N*m/rad: saturates the 0.5 N*m limit at ~7 deg of error
_XL330_DAMPING = 0.1  # N*m*s/rad
_XL330_EFFORT = 0.5  # N*m, XL330-M288 stall torque at 5 V
_XL330_SPEED = 8.0  # rad/s, below the ~10 rad/s no-load speed

REACHY_MINI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=REACHY_MINI_USD_PATH,
        activate_contact_sensors=False,
        articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "xl330": ImplicitActuatorCfg(
            joint_names_expr=REACHY_MINI_ACTUATED_JOINTS,
            stiffness=_XL330_STIFFNESS,
            damping=_XL330_DAMPING,
            joint_effort_limit=_XL330_EFFORT,
            actuator_velocity_limit=_XL330_SPEED,
            armature=0.005,
            friction=0.01,
        ),
        # passive_1..7 are ball joints (3 DOFs each, named passive_N:0..2)
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["passive_.*"],
            stiffness=0.0,
            damping=0.0,
            armature=0.0005,
            friction=0.0,
        ),
    },
)
"""Reachy Mini with XL330-like servos on the 9 actuated joints and free passive ball joints."""
