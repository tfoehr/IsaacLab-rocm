"""Reachy Mini head pose (experimental): turn the head to sampled orientations.

The policy drives yaw_body, the six Stewart servos and the two antennas with joint position
targets; the passive ball joints and the five loop closures are left to the solver. The command
rotates the head about its centre by roll/pitch in +-0.4 rad and yaw in +-1.0 rad (the body yaw
joint has to help for large yaws); the head centre should stay where it is in the zero pose.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from ...assets.reachy_mini import (
    REACHY_MINI_ACTUATED_JOINTS,
    REACHY_MINI_CFG,
    REACHY_MINI_HEAD_BODY,
    REACHY_MINI_HEAD_CENTER_ZERO_POS,
    REACHY_MINI_HEAD_ZERO_POS,
    REACHY_MINI_HEAD_ZERO_QUAT,
)

_ACTUATED = SceneEntityCfg("robot", joint_names=REACHY_MINI_ACTUATED_JOINTS, preserve_order=True)
_HEAD = SceneEntityCfg("robot", body_names=[REACHY_MINI_HEAD_BODY])


@configclass
class ReachyMiniSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = REACHY_MINI_CFG
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
    )


@configclass
class CommandsCfg:
    head_pose = mdp.HeadPoseCommandCfg(
        asset_name="robot",
        body_name=REACHY_MINI_HEAD_BODY,
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        zero_pos=REACHY_MINI_HEAD_ZERO_POS,
        zero_quat=REACHY_MINI_HEAD_ZERO_QUAT,
        center_pos=REACHY_MINI_HEAD_CENTER_ZERO_POS,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(-0.4, 0.4), pitch=(-0.4, 0.4), yaw=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=REACHY_MINI_ACTUATED_JOINTS, scale=0.5, use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ACTUATED}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ACTUATED}, noise=Unoise(n_min=-0.05, n_max=0.05))
        head_pose = ObsTerm(func=mdp.body_pose_b, params={"asset_cfg": _HEAD})
        head_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # assembled zero pose: all joints (incl. the passive ball joints) at 0, loops closed
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    head_orientation_error = RewTerm(
        func=mdp.orientation_command_error, weight=-0.5, params={"asset_cfg": _HEAD, "command_name": "head_pose"}
    )
    head_orientation_tanh = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=1.0,
        params={"asset_cfg": _HEAD, "std": 0.1, "command_name": "head_pose"},
    )
    head_position_error = RewTerm(
        func=mdp.position_command_error, weight=-10.0, params={"asset_cfg": _HEAD, "command_name": "head_pose"}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _ACTUATED})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": _ACTUATED})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ReachyMiniHeadPoseEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachyMiniSceneCfg = ReachyMiniSceneCfg(num_envs=4096, env_spacing=0.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 10  # 500 Hz physics, 50 Hz policy
        self.episode_length_s = 8.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=80,  # 5 connect equalities (15 rows) + joint limits + friction
                nconmax=8,  # self-collisions are off; the robot touches nothing else
                cone="pyramidal",
                impratio=1.0,
                iterations=30,
                ls_iterations=15,
                update_data_interval=1,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        )
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.45, 0.0, 0.3), lookat=(0.0, 0.0, 0.15))


@configclass
class ReachyMiniHeadPoseEnvCfg_PLAY(ReachyMiniHeadPoseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.play_mode()
