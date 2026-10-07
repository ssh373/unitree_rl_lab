# AMP reference motions

Reference clips for the `Unitree-G1-29dof-Running-AMP` discriminator, retargeted to the G1 29dof.
Only the **csv** files are committed; the **npz** files (Isaac Sim forward kinematics, tied to the robot
USD, 21–24 MB each) are gitignored and rebuilt per machine:

    conda activate env_isaaclab
    bash scripts/mimic/build_amp_npz.sh        # ~30 s per clip headless, ~12 min for all 22

| directory | clips | source | csv quaternion | use |
|---|---|---|---|---|
| `lafan1_run/` | run1/run2/sprint1 × 2 subjects (6) | LAFAN1 retarget, 30 fps | xyzw | running band 1.0–2.5 m/s |
| `lafan1_walk/` | walk1–4 (12) | LAFAN1 retarget, 30 fps | xyzw | walking band 0.2–1.0 m/s |
| `kimodo/` | kimodo_1–4 (10 s each) | Kimodo generated, 30 fps | xyzw (converted; originals in `kimodo/orig_wxyz/`) | 0.2–2.5 m/s |

csv layout (36 columns, no header): root xyz, root quaternion, 29 joint angles in `joint_sdk_names` order.
`csv_to_npz.py` expects **xyzw**; Kimodo exports wxyz, so its clips were re-ordered once — do not convert
`orig_wxyz/` directly.

Which clips a run uses is set in `agents/rsl_rl_ppo_cfg.py` (`LOWCOT_CLIPS`, `LAFAN1_BANDED_CLIPS`,
`LAFAN1_RUN_SINGLE_CLIP`). The low-CoT selection (inverse-dynamics CoT of each clip, 2026-10-06) is
walk3_subject4, walk1_subject1 / run2_subject4, run1_subject5 / kimodo_1–4.
