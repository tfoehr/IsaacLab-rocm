"""SO-101 lift: grasp a 25 mm, 30 g cube from the floor and carry it to a sampled position.

Classic Isaac Lab lift MDP (contrib/lift) with the reward shaping of MuammerBay/isaac_so_arm101
(BSD-3-Clause), on Newton/MuJoCo-Warp with the solver settings of Isaac Lab's core lift task.
Actions: joint position targets on the 5 arm joints plus a binary open/close gripper.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from ... import mdp
from ...assets.so101 import SO101_ARM_JOINTS, SO101_EE_BODY, SO101_GRASP_OFFSET, SO101_ROCM_CFG

CUBE_SIZE = 0.025  # m
CUBE_MASS = 0.03  # kg
# Cube rest position in the robot base frame (arm faces -Y, pan axis at x ~= +0.02 m): 0.20 m ahead.
CUBE_POS = (0.02, -0.22, CUBE_SIZE / 2)
LIFT_HEIGHT = 0.025  # m, cube centre above the floor that counts as lifted (12.5 mm off the floor)
# Start pose: jaws pointing down, open, the grasp point ~0.10 m above the cube area (found with
# scripts/so101_scripted_lift.py). It is also the action offset, so a zero action hovers over the
# cube instead of holding the gripper horizontal 0.24 m up, which the policy had to unlearn first.
LIFT_INIT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.1,
    "wrist_flex": 1.37,
    "wrist_roll": 0.0,
    "gripper": 0.5,
}


@configclass
class SO101LiftSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = replace(
        SO101_ROCM_CFG, init_state=replace(SO101_ROCM_CFG.init_state, joint_pos=LIFT_INIT_JOINT_POS)
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=CUBE_MASS),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.15)),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class CommandsCfg:
    # goal for the cube centre, robot base frame (body_name only feeds the command's markers/metrics)
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=SO101_EE_BODY,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.02 - 0.10, 0.02 + 0.10),
            pos_y=(-0.023 - 0.26, -0.023 - 0.14),
            pos_z=(0.08, 0.20),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=SO101_ARM_JOINTS, scale=0.5, use_default_offset=True
    )
    # open: ~49 mm jaw gap; close: -0.1 rad (~12 mm gap), which squeezes a 25 mm cube at the servo's
    # 3.35 N*m limit (closing only to 0 rad, ~22 mm, gave too weak a grasp: no lifting in 1000 iterations)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": -0.1},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # clipped so that one runaway environment cannot blow up the policy update
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-5.0, 5.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-30.0, 30.0))
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, clip=(-1.0, 1.0))
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_ARM_JOINTS),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.04, 0.04), "yaw": (-0.8, 0.8)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


_GRASP = SceneEntityCfg("robot", body_names=[SO101_EE_BODY])


@configclass
class RewardsCfg:
    reaching_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.05, "robot_cfg": _GRASP, "offset": SO101_GRASP_OFFSET}, weight=1.0
    )
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": LIFT_HEIGHT}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": LIFT_HEIGHT, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": LIFT_HEIGHT, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=SO101_ARM_JOINTS)}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    # the cube left the workspace (e.g. ejected by a contact) or its state is not finite
    object_out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds, params={"bounds": ((-0.6, 0.6), (-0.8, 0.4), (-0.05, 0.8))}
    )
    abnormal_robot = DoneTerm(func=mdp.joint_state_invalid, params={"max_joint_vel": 50.0})


@configclass
class CurriculumCfg:
    # smoothness penalties after ~830 iterations (24 steps each), once lifting is found; ten times
    # weaker than the Franka task's -1e-1, which pulled the reward down while the grasp was still forming
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 20000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 20000}
    )


@configclass
class SO101LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101LiftSceneCfg = SO101LiftSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4  # 120 Hz physics step (2 MuJoCo substeps), 30 Hz policy
        self.episode_length_s = 6.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # MuJoCo-Warp settings of Isaac Lab's core lift task (newton_mjwarp preset)
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=200,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=4000000),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=2,
            debug_mode=False,
        )
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.5, 0.4, 0.4), lookat=(0.0, -0.2, 0.05))


@configclass
class SO101LiftEnvCfg_PLAY(SO101LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.play_mode()
