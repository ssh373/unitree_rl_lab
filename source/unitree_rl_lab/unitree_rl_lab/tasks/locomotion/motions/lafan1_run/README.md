# LAFAN1 running clips, retargeted to G1 29dof

Reference motions for the AMP style reward of `Unitree-G1-29dof-Running-AMP`.

Source: LAFAN1 retargeting dataset (Unitree G1 29dof), `run1/run2/sprint1` takes.

Format (verified against the mimic dance clips, which the npz converter is known to handle):
36 columns, no header, one row per frame at **30 fps** -- note the mimic dance clips are 60 fps.

| columns | content |
|---------|---------|
| 0-2     | root position x, y, z (world, m) |
| 3-6     | root quaternion **x, y, z, w** (converted to wxyz by the loader) |
| 7-35    | 29 joint angles (rad) in `joint_sdk_names` order |

## Regenerating the npz files

The npz files are gitignored: `csv_to_npz.py` runs forward kinematics in Isaac Sim, so its
output is tied to the robot USD and must be rebuilt rather than committed.

    python scripts/mimic/csv_to_npz.py -f <clip>.csv --input_fps 30 --headless

Clips are converted in full. Idle and over-speed segments are interspersed throughout the
takes rather than sitting at the ends, so they are dropped per transition by the speed gate
in the AMP data loader, not by trimming frame ranges here.

## Clip statistics (from the CSV root trajectory, 30 fps)

| clip | length | v_mean | v_p90 | v_max | in 0.3-2.5 m/s | idle | over 2.5 |
|------|-------:|-------:|------:|------:|---------------:|-----:|---------:|
| run1_subject2    | 237.8 s | 1.21 | 2.33 | 3.53 | 77.8% | 15.6% |  6.6% |
| run1_subject5    | 237.8 s | 1.53 | 2.40 | 3.75 | 88.7% |  3.2% |  8.0% |
| run2_subject1    | 244.8 s | 1.38 | 2.47 | 3.96 | 78.4% | 12.6% |  9.0% |
| run2_subject4    | 244.8 s | 1.50 | 2.71 | 4.43 | 79.5% |  5.4% | 15.1% |
| sprint1_subject2 | 273.1 s | 0.97 | 3.09 | 4.86 | 38.6% | 44.5% | 16.9% |
| sprint1_subject4 | 273.1 s | 0.96 | 3.43 | 4.85 | 25.1% | 55.8% | 19.1% |

The sprint takes are mostly standing between efforts, but they hold the only data above
3 m/s, so they are kept and gated rather than dropped.
