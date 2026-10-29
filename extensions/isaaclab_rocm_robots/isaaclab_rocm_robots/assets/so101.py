"""SO-101 arm for the ROCm tasks, built on Isaac Lab's upstream :data:`SO101_CFG`.

The upstream asset (TheRobotStudio SO-101, NVIDIA-hosted SysID USD) is used unchanged: it
selects the SysID ``physics`` variant for Newton/MuJoCo-Warp with the USD-authored STS3215
servo model (stiffness ~48/48/15/44/55/68 N*m/rad, 3.35 N*m force limit, joint friction and
armature). The USD downloads kit-less from NVIDIA's public S3 on first use.

Frame facts (measured from the USD, zero joint pose):

* The arm points along the base's **-Y** axis; the pan axis sits at x ~= +0.02 m in the base frame.
  The tasks keep the base unrotated, so base frame == world frame (minus the env origin).
* The USD has no TCP frame. ``gripper`` is the last arm link; in its frame the jaws point along -Z.
  :data:`SO101_TCP_OFFSET` is the fingertip point (same offset as the upstream keyboard task),
  :data:`SO101_GRASP_OFFSET` is the centre between the jaw pads.
* The ``gripper`` joint closes towards its lower limit (-0.17 rad, jaw gap ~8 mm); 0 rad leaves a
  ~22 mm gap, 0.5 rad ~49 mm, 1.0 rad ~70 mm.
"""

from isaaclab.utils import replace

from isaaclab_assets.robots.so101 import SO101_CFG

SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_EE_BODY = "gripper"
SO101_TCP_OFFSET = (-0.0079, 0.0, -0.0981)
SO101_GRASP_OFFSET = (0.0, 0.0, -0.07)

SO101_ROCM_CFG = replace(
    SO101_CFG,
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=replace(
        SO101_CFG.init_state,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        # zero = upper arm up, forearm and gripper horizontal along -Y; gripper half open
        joint_pos={"shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll": 0.0, "gripper": 0.5},
        joint_vel={".*": 0.0},
    ),
)
"""SO-101 at the env origin, zero arm pose, gripper half open, USD (SysID) actuator model."""
