"""Scripted grasp-and-lift check for ``ROCm-Lift-Cube-SO101`` (no policy).

Drives the lift task's own action terms with a damped-least-squares IK on the ``gripper`` link
Jacobian: hover above the cube with the jaws pointing down, descend, close, lift. Reports the
cube height per phase, so physics problems (slipping, ejection, penetration, a weak grip) show up
before any RL is involved.

    source scripts/env.sh
    python extensions/isaaclab_rocm_robots/scripts/so101_scripted_lift.py --device cpu --num_envs 4
"""

from __future__ import annotations

import argparse

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--hover", type=float, default=0.06, help="grasp point height above the cube centre [m]")
parser.add_argument("--lift", type=float, default=0.10, help="cube centre lift target above the floor [m]")
parser.add_argument("--grasp_z", type=float, default=0.0, help="grasp point offset from the cube centre in z [m]")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--cone", choices=["pyramidal", "elliptic"], default=None, help="override the MuJoCo friction cone")
parser.add_argument("--impratio", type=float, default=None, help="override the MuJoCo impratio")
parser.add_argument("--close", type=float, default=None, help="override the gripper close target [rad]")
parser.add_argument("--friction", type=float, default=None, help="override the cube static/dynamic friction")
parser.add_argument("--init_pose", action="store_true", help="print the arm joint targets reached at the hover pose")
add_launcher_args(parser)
args = parser.parse_args()


def main():
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply

    from isaaclab_rocm_robots.assets.so101 import SO101_ARM_JOINTS, SO101_EE_BODY, SO101_GRASP_OFFSET
    from isaaclab_rocm_robots.tasks.so101_lift.lift_env_cfg import SO101LiftEnvCfg

    cfg = SO101LiftEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 1.0
    cfg.observations.policy.enable_corruption = False
    cfg.commands.object_pose.debug_vis = False
    cfg.episode_length_s = 8.0
    solver = cfg.sim.physics.solver_cfg
    if args.cone is not None:
        solver.cone = args.cone
    if args.impratio is not None:
        solver.impratio = args.impratio
    if args.close is not None:
        cfg.actions.gripper_action.close_command_expr = {"gripper": args.close}
    if args.friction is not None:
        cfg.scene.object.spawn.physics_material.static_friction = args.friction
        cfg.scene.object.spawn.physics_material.dynamic_friction = args.friction
    print(f"[scripted] cone={solver.cone} impratio={solver.impratio} "
          f"close={cfg.actions.gripper_action.close_command_expr} "
          f"cube_mu={cfg.scene.object.spawn.physics_material.static_friction}")

    with launch_simulation(cfg, args):
        env = ManagerBasedRLEnv(cfg)
        env.reset()
        robot = env.scene["robot"]
        cube = env.scene["object"]
        dev = env.device
        n = env.num_envs
        dt = env.step_dt

        arm_ids, _ = robot.find_joints(SO101_ARM_JOINTS, preserve_order=True)
        grip_id = robot.find_joints(["gripper"])[0][0]
        body_id = robot.find_bodies([SO101_EE_BODY])[0][0]
        jac_body = body_id - 1 if robot.is_fixed_base else body_id
        jac_cols = [j + robot.num_base_dofs for j in arm_ids]
        arm_term = env.action_manager.get_term("arm_action")
        scale, offset = arm_term._scale, arm_term._offset
        off_b = torch.tensor(SO101_GRASP_OFFSET, device=dev).expand(n, 3)
        down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)
        jaw_axis_b = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)

        q_cmd = robot.data.joint_pos.torch[:, arm_ids].clone()
        phases = [("hover", 1.5), ("descend", 1.5), ("close", 1.0), ("lift", 2.0), ("hold", 1.5)]
        t_end = []
        acc = 0.0
        for _, d in phases:
            acc += d
            t_end.append(acc)
        steps = int(round(acc / dt))
        cube0 = cube.data.root_pos_w.torch - env.scene.env_origins
        target_xy = cube0[:, :2].clone()
        cube_z0 = cube0[:, 2].clone()
        log = {}
        actions = torch.zeros(n, env.action_manager.total_action_dim, device=dev)

        for k in range(steps):
            t = k * dt
            ph = next(i for i, te in enumerate(t_end) if t < te + 1e-9)
            name = phases[ph][0]
            pos = robot.data.body_pos_w.torch[:, body_id] - env.scene.env_origins
            quat = robot.data.body_quat_w.torch[:, body_id]
            grasp = pos + quat_apply(quat, off_b)
            axis = quat_apply(quat, jaw_axis_b)
            cube_p = cube.data.root_pos_w.torch - env.scene.env_origins

            goal = torch.zeros(n, 3, device=dev)
            goal[:, :2] = target_xy
            if name == "hover":
                goal[:, 2] = cube_z0 + args.hover
            elif name == "descend":
                a = min(1.0, (t - t_end[0]) / (0.8 * phases[1][1]))
                goal[:, 2] = cube_z0 + args.grasp_z + (1 - a) * args.hover
            elif name == "close":
                goal[:, 2] = cube_z0 + args.grasp_z
            else:
                a = min(1.0, (t - t_end[2]) / (0.8 * phases[3][1]))
                goal[:, 2] = cube_z0 + args.grasp_z + a * (args.lift - cube_z0)

            jac = robot.data.body_link_jacobian_w.torch[:, jac_body][:, :, jac_cols]  # (n, 6, 5)
            r = grasp - pos
            jv = jac[:, :3] - torch.cross(r.unsqueeze(-1).expand(-1, -1, 5), jac[:, 3:], dim=1)
            e_pos = (goal - grasp).clamp(-0.02, 0.02)
            e_rot = torch.cross(axis, down, dim=1)
            J = torch.cat([jv, 0.3 * jac[:, 3:]], dim=1)
            e = torch.cat([e_pos, 0.3 * e_rot], dim=1)
            lam = 0.01
            JJt = J @ J.transpose(1, 2) + lam * torch.eye(6, device=dev)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
            q_cmd = q_cmd + dq.clamp(-0.1, 0.1)
            lo = robot.data.soft_joint_pos_limits.torch[:, arm_ids, 0]
            hi = robot.data.soft_joint_pos_limits.torch[:, arm_ids, 1]
            q_cmd = torch.maximum(torch.minimum(q_cmd, hi), lo)

            actions[:, :5] = (q_cmd - offset) / scale
            actions[:, 5] = -1.0 if name in ("close", "lift", "hold") else 1.0
            env.step(actions)

            if args.trace and k % 5 == 0:
                print(
                    f"t={t:5.2f} {name:8s} grasp={grasp[0].tolist()} goal={goal[0].tolist()} "
                    f"cube={cube_p[0].tolist()} axis={axis[0].tolist()} grip={robot.data.joint_pos.torch[0, grip_id]:.3f}"
                )
            if args.init_pose and name == "hover" and t + dt >= t_end[0] - 1e-9:
                print("[scripted] hover joint targets (mean over envs):", [round(v, 3) for v in q_cmd.mean(0).tolist()])
                print("[scripted] hover grasp point (env 0):", [round(v, 4) for v in grasp[0].tolist()])
            if t + dt >= t_end[ph] - 1e-9:
                log[name] = dict(
                    cube_z=cube_p[:, 2].clone(),
                    cube_xy_err=torch.linalg.norm(cube_p[:, :2] - target_xy, dim=1),
                    grasp_err=torch.linalg.norm(goal - grasp, dim=1),
                    gripper=robot.data.joint_pos.torch[:, grip_id].clone(),
                    gripper_effort=robot.data.applied_torque.torch[:, grip_id].clone()
                    if hasattr(robot.data, "applied_torque")
                    else torch.zeros(n, device=dev),
                )

        print("\nphase     cube_z[mm] (mean/min/max)    cube_xy_drift[mm]  grasp_err[mm]  gripper[rad]  grip_torque")
        for name, v in log.items():
            z = v["cube_z"] * 1000
            print(
                f"{name:8s}  {z.mean():6.1f} / {z.min():6.1f} / {z.max():6.1f}      {v['cube_xy_err'].mean() * 1000:6.1f}"
                f"           {v['grasp_err'].mean() * 1000:6.1f}        {v['gripper'].mean():+.3f}       {v['gripper_effort'].mean():+.3f}"
            )
        final_z = log["hold"]["cube_z"]
        # where a held cube sits in the gripper-link frame (to place the grasp point of the rewards)
        from isaaclab.utils.math import quat_apply_inverse

        pos = robot.data.body_pos_w.torch[:, body_id]
        quat = robot.data.body_quat_w.torch[:, body_id]
        rel = quat_apply_inverse(quat, cube.data.root_pos_w.torch - pos)
        held = final_z > 0.04
        if held.any():
            print("held cube centre in the gripper frame [m] (mean over lifted envs):",
                  [round(v, 4) for v in rel[held].mean(0).tolist()])
        print("per-env final cube z [mm]:", [round(v * 1000, 1) for v in final_z.tolist()])
        ok = final_z > 0.04
        print(f"\nlifted (cube centre > 40 mm at the end): {int(ok.sum())}/{n} = {ok.float().mean() * 100:.0f}%")
        env.close()


if __name__ == "__main__":
    main()
