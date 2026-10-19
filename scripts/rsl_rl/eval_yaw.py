"""Fixed (v_x, w_z) command protocol: how well does a checkpoint turn while running?

Per command pair the command is held constant, a warm-up transient is discarded, then over the window:
  yaw_mean_err   |w_cmd - mean(w_z)|  (bias of the stride-averaged yaw rate)   [rad/s]
  yaw_inst_err   mean |w_cmd - w_z(t)| (instantaneous, includes gait oscillation) [rad/s]
  yaw_osc_std    std of w_z(t) around its mean (gait-induced yaw oscillation)    [rad/s]
  heading_err    |psi_cmd - psi_actual| with psi = integral of w_z (world) over the window [rad]
  lin_err        mean |v_cmd - v_xy(t)| in body frame                              [m/s]
"""
import argparse
from isaaclab.app import AppLauncher
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/kist/unitree_rl_lab/scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Running")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--pairs", type=str, default="1.0:0.0,1.0:0.25,1.0:0.5,1.0:0.75,1.0:1.0,2.0:0.0,2.0:0.25,2.0:0.5,2.0:0.75,2.0:1.0")
parser.add_argument("--warmup_steps", type=int, default=150)
parser.add_argument("--measure_steps", type=int, default=500)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, torch, csv
import isaaclab_tasks  # noqa
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
import unitree_rl_lab.tasks  # noqa
from unitree_rl_lab.algorithms.registry import get_runner_class
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    for term_name in list(vars(env_cfg.curriculum).keys()):
        setattr(env_cfg.curriculum, term_name, None)
    env_cfg.scene.terrain.terrain_type = "plane"; env_cfg.scene.terrain.terrain_generator = None
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9); cmd.rel_standing_envs = 0.0; cmd.heading_command = False; cmd.debug_vis = False
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.checkpoint and os.path.sep in args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = get_runner_class(getattr(agent_cfg, "class_name", "OnPolicyRunner"))(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base_env = env.unwrapped; robot = base_env.scene["robot"]; device = base_env.device; dt = base_env.step_dt
    term = base_env.command_manager.get_term("base_velocity")
    rows = []
    for pair in args_cli.pairs.split(","):
        vx, wz = (float(x) for x in pair.split(":"))
        term.cfg.ranges.lin_vel_x = (vx, vx); term.cfg.ranges.lin_vel_y = (0.0, 0.0); term.cfg.ranges.ang_vel_z = (wz, wz)
        N = base_env.num_envs; n = float(args_cli.measure_steps)
        wz_sum = torch.zeros(N, device=device); wz_sq = torch.zeros(N, device=device); inst = torch.zeros(N, device=device)
        psi = torch.zeros(N, device=device); lin = torch.zeros(N, device=device); spd = torch.zeros(N, device=device)
        tau2 = torch.zeros(N, device=device); ppow = torch.zeros(N, device=device); valid = torch.ones(N, dtype=torch.bool, device=device)
        with torch.inference_mode():
            obs, _ = env.reset()
            for step in range(args_cli.warmup_steps + args_cli.measure_steps):
                actions = policy(obs); out = env.step(actions); obs, dones = out[0], out[2]
                if step < args_cli.warmup_steps: continue
                valid &= ~dones.bool()
                w = robot.data.root_ang_vel_b[:, 2]; ww = robot.data.root_ang_vel_w[:, 2]
                c = base_env.command_manager.get_command("base_velocity")
                wz_sum += w; wz_sq += w * w; inst += torch.abs(c[:, 2] - w); psi += ww * dt
                lin += torch.linalg.norm(c[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=-1)
                spd += torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
                torque = robot.data.applied_torque
                tau2 += torch.square(torque).sum(-1); ppow += torch.clamp_min(torque * robot.data.joint_vel, 0.0).sum(-1)
        if valid.sum() == 0:
            print(f"[WARN] all envs fell at vx={vx} wz={wz}"); rows.append({"vx": vx, "wz": wz, "survival": 0.0}); continue
        m = lambda t: t[valid].mean().item()
        wz_mean = wz_sum / n; wz_std = torch.sqrt(torch.clamp_min(wz_sq / n - wz_mean ** 2, 0.0))
        T = n * dt
        row = {"vx": vx, "wz": wz, "survival": valid.float().mean().item(), "speed_mps": m(spd) / n,
               "wz_actual": m(wz_mean), "yaw_mean_err": m(torch.abs(wz - wz_mean)), "yaw_inst_err": m(inst) / n,
               "yaw_osc_std": m(wz_std), "heading_err_rad": m(torch.abs(wz * T - psi)), "heading_cmd_rad": wz * T,
               "lin_err": m(lin) / n, "torque_sq_raw": m(tau2) / n, "pos_power_w": m(ppow) / n}
        rows.append(row)
        print(f"vx {vx:3.1f} wz {wz:4.2f} | wz_act {row['wz_actual']:5.2f} | mean_err {row['yaw_mean_err']:4.2f} inst {row['yaw_inst_err']:4.2f} osc {row['yaw_osc_std']:4.2f} | "
              f"heading err {row['heading_err_rad']:5.2f}/{row['heading_cmd_rad']:5.2f} rad | lin_err {row['lin_err']:4.2f} | tau2 {row['torque_sq_raw']:6.0f} | alive {row['survival']:.2f}")
    if args_cli.out:
        with open(args_cli.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()) if rows else ["vx"]); w.writeheader(); w.writerows(rows)
        print("[INFO] wrote", args_cli.out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
