"""Evaluate a ``ROCm-Lift-Cube-SO101`` checkpoint headless: cube height and goal distance over one episode.

    source scripts/env.sh
    python extensions/isaaclab_rocm_robots/scripts/so101_eval_lift.py --device cuda:0 --num_envs 64 \\
        --checkpoint isaaclab/logs/rsl_rl/rocm_so101_lift/<run>/model_1499.pt
"""

from __future__ import annotations

import argparse

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=1, help="consecutive episodes per environment")
parser.add_argument("--seed", type=int, default=123)
add_launcher_args(parser)
args = parser.parse_args()


def main():
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import combine_frame_transforms, quat_apply

    from isaaclab_rocm_robots.assets.so101 import SO101_EE_BODY, SO101_GRASP_OFFSET

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_rocm_robots.tasks.so101_lift.agents.rsl_rl_ppo_cfg import SO101LiftPPORunnerCfg
    from isaaclab_rocm_robots.tasks.so101_lift.lift_env_cfg import LIFT_HEIGHT, SO101LiftEnvCfg_PLAY

    cfg = SO101LiftEnvCfg_PLAY()
    cfg.scene.num_envs = args.num_envs
    cfg.commands.object_pose.debug_vis = False
    cfg.seed = args.seed
    agent_cfg = SO101LiftPPORunnerCfg()
    agent_cfg.device = args.device

    with launch_simulation(cfg, args):
        base = ManagerBasedRLEnv(cfg)
        env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=base.device)

        n = base.num_envs
        T = int(base.max_episode_length)
        robot, cube = base.scene["robot"], base.scene["object"]
        body_id = robot.find_bodies([SO101_EE_BODY])[0][0]
        grip_id = robot.find_joints(["gripper"])[0][0]
        off_b = torch.tensor(SO101_GRASP_OFFSET, device=base.device).expand(n, 3)
        results = []
        for ep in range(args.episodes):
            # start every environment on a fresh episode
            base.episode_length_buf[:] = T  # forces a time-out reset on the next step
            obs = env.get_observations()
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            alive = torch.ones(n, dtype=torch.bool, device=base.device)
            heights = torch.zeros(n, T, device=base.device)
            goal_d = torch.full((n, T), float("nan"), device=base.device)
            valid = torch.zeros(n, T, dtype=torch.bool, device=base.device)
            grasp_d = torch.zeros(n, T, device=base.device)
            grip_q = torch.zeros(n, T, device=base.device)
            for k in range(T - 1):
                with torch.inference_mode():
                    obs, _, dones, extras = env.step(policy(obs))
                alive &= ~dones.bool()
                z = cube.data.root_pos_w.torch[:, 2] - base.scene.env_origins[:, 2]
                cmd = base.command_manager.get_command("object_pose")
                goal_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, cmd[:, :3])
                heights[:, k] = z
                goal_d[:, k] = torch.linalg.norm(goal_w - cube.data.root_pos_w.torch, dim=1)
                valid[:, k] = alive
                gp = robot.data.body_pos_w.torch[:, body_id] + quat_apply(robot.data.body_quat_w.torch[:, body_id], off_b)
                grasp_d[:, k] = torch.linalg.norm(gp - cube.data.root_pos_w.torch, dim=1)
                grip_q[:, k] = robot.data.joint_pos.torch[:, grip_id]
            lifted = (heights > LIFT_HEIGHT) & valid
            frac = lifted.sum(1).float() / valid.sum(1).clamp(min=1).float()
            last = valid.sum(1) - 1
            idx = torch.arange(n, device=base.device)
            final_d = goal_d[idx, last.clamp(min=0)]
            final_lifted = lifted[idx, last.clamp(min=0)]
            # goal distance over the last second, for envs holding the cube then
            tail = slice(T - 1 - int(1.0 / base.step_dt), T - 1)
            tail_lift = lifted[:, tail].all(1)
            tail_d = torch.nanmean(torch.where(lifted[:, tail], goal_d[:, tail], torch.nan), dim=1)
            res = dict(
                ever_lifted=(lifted.any(1)).float().mean().item(),
                lifted_half=(frac >= 0.5).float().mean().item(),
                mean_frac=frac.mean().item(),
                final_lifted=final_lifted.float().mean().item(),
                held_last_second=tail_lift.float().mean().item(),
                goal_dist_last_second_mm=(tail_d[tail_lift].mean().item() * 1000) if tail_lift.any() else float("nan"),
                # while lifted: cube centre to the grasp point between the jaws, and the gripper angle
                grasp_dist_lifted_mm=(grasp_d[lifted].mean().item() * 1000) if lifted.any() else float("nan"),
                gripper_lifted_rad=grip_q[lifted].mean().item() if lifted.any() else float("nan"),
                max_height_mm=(heights.max(1).values.mean().item() * 1000),
                terminated_early=(~alive).float().mean().item(),
            )
            results.append(res)
            print(f"[eval] episode {ep}: " + ", ".join(f"{k}={v:.3f}" for k, v in res.items()))
        keys = results[0].keys()
        print("[eval] mean over episodes: " + ", ".join(f"{k}={sum(r[k] for r in results) / len(results):.3f}" for k in keys))
        base.close()


if __name__ == "__main__":
    main()
