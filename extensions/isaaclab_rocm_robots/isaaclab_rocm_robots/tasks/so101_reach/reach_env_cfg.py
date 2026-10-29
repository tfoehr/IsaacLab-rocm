"""SO-101 reach: move the fingertip to sampled positions with joint position targets on the 5 arm joints.

Adapted from Isaac Lab's core reach task and MuammerBay/isaac_so_arm101 (BSD-3-Clause), for the
upstream SysID SO-101 USD on Newton/MuJoCo-Warp. Orientation is not tracked (5-DOF arm).
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from ... import mdp
from ...assets.so101 import SO101_ARM_JOINTS, SO101_EE_BODY, SO101_ROCM_CFG, SO101_TCP_OFFSET

# Target box in the robot base frame. The arm faces -Y and the pan axis is at x ~= +0.02 m, so
# this is 0.15-0.30 m in front of the pan axis, +-0.15 m to the side, 0.05-0.25 m above the base.
REACH_RANGES = mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(0.02 - 0.15, 0.02 + 0.15),
    pos_y=(-0.023 - 0.30, -0.023 - 0.15),
    pos_z=(0.05, 0.25),
    roll=(0.0, 0.0),
    pitch=(0.0, 0.0),
    yaw=(0.0, 0.0),
)

_EE = SceneEntityCfg("robot", body_names=[SO101_EE_BODY])


@configclass
class SO101ReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = SO101_ROCM_CFG
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )


@configclass
class CommandsCfg:
    # body_name drives the debug markers and the command's own metrics (gripper link origin);
    # the rewards track the fingertip point SO101_TCP_OFFSET instead.
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=SO101_EE_BODY,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=REACH_RANGES,
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=SO101_ARM_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_ARM_JOINTS),
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    ee_position_error = RewTerm(
        func=mdp.ee_position_command_error,
        weight=-0.2,
        params={"asset_cfg": _EE, "offset": SO101_TCP_OFFSET, "command_name": "ee_pose"},
    )
    ee_position_tanh = RewTerm(
        func=mdp.ee_position_command_error_tanh,
        weight=0.5,
        params={"asset_cfg": _EE, "offset": SO101_TCP_OFFSET, "std": 0.1, "command_name": "ee_pose"},
    )
    ee_position_fine = RewTerm(
        func=mdp.ee_position_command_error_tanh,
        weight=0.5,
        params={"asset_cfg": _EE, "offset": SO101_TCP_OFFSET, "std": 0.02, "command_name": "ee_pose"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=SO101_ARM_JOINTS)}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


@configclass
class SO101ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101ReachSceneCfg = SO101ReachSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4  # 120 Hz physics step (2 MuJoCo substeps), 30 Hz policy
        self.episode_length_s = 12.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=150, nconmax=40, cone="pyramidal", integrator="implicitfast", impratio=1, update_data_interval=2
            ),
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=True,
        )
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.6, 0.6, 0.5), lookat=(0.0, -0.2, 0.1))


@configclass
class SO101ReachEnvCfg_PLAY(SO101ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.play_mode()
