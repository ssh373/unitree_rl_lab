#!/usr/bin/env bash
# Rebuild the AMP reference npz files from the committed 30 fps csv clips (Isaac Sim forward kinematics).
#
# The npz files are gitignored (they depend on the robot USD), so run this once on every machine that
# trains an AMP task. csv_to_npz.py keeps its sim loop alive after np.savez, so each conversion is
# killed as soon as its npz appears. ~30 s per 4-minute clip headless; ~12 min for all 22 clips.
#
#   conda activate env_isaaclab
#   bash scripts/mimic/build_amp_npz.sh                 # every csv under motions/{lafan1_run,lafan1_walk,kimodo}
#   bash scripts/mimic/build_amp_npz.sh motions/kimodo  # one directory
#   FORCE=1 bash scripts/mimic/build_amp_npz.sh         # rebuild even where an npz exists
set -u
cd "$(dirname "$0")/../.."
MOTIONS=source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/motions

# Same interpreter resolution as unitree_rl_lab.sh: an explicit PYTHON, else the active conda env,
# else the python bundled with Isaac Lab (the docker image has no conda and no `python` on PATH).
if [ -n "${PYTHON:-}" ]; then :
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -n "${ISAACLAB_PATH:-}" ] && [ -x "${ISAACLAB_PATH}/_isaac_sim/python.sh" ]; then PYTHON="${ISAACLAB_PATH}/_isaac_sim/python.sh"
else PYTHON=python
fi
echo "python: $PYTHON"
DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("$MOTIONS/lafan1_run" "$MOTIONS/lafan1_walk" "$MOTIONS/kimodo")

for dir in "${DIRS[@]}"; do
  for csv in "$dir"/*.csv; do
    npz="${csv%.csv}.npz"
    if [ -f "$npz" ] && [ -z "${FORCE:-}" ]; then echo "keep  $npz"; continue; fi
    rm -f "$npz"
    echo "build $npz"
    "$PYTHON" scripts/mimic/csv_to_npz.py -f "$csv" --input_fps 30 --headless > "${csv%.csv}.convert.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 240); do            # up to 8 min per clip
      if [ -f "$npz" ]; then sleep 3; break; fi
      kill -0 $pid 2>/dev/null || break
      sleep 2
    done
    pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 2; kill -9 $pid 2>/dev/null
    if [ -f "$npz" ]; then
      rm -f "${csv%.csv}.convert.log"
      chown --reference="$csv" "$npz" 2>/dev/null  # inside docker we are root; hand the file to the checkout's owner
    else
      echo "FAILED $csv (see ${csv%.csv}.convert.log)"
    fi
  done
done
echo "done: $(ls "${DIRS[@]/%//*.npz}" 2>/dev/null | wc -l) npz files"
