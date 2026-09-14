#!/usr/bin/env bash
# Run a training job inside the container and hand the results back to the host user.
#
# The image runs as root (Isaac Sim needs OMNI_KIT_ALLOW_ROOT=1), so everything it writes under
# logs/ and wandb/ would belong to root on the host. On a shared server without sudo that makes
# the files impossible to move or delete, so this wrapper chowns them to the host uid/gid when the
# job ends -- also when it fails or is interrupted.
#
# Usage (from the host, all extra args go to `unitree_rl_lab.sh -t`):
#   docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab \
#       docker/train.sh $(id -u) $(id -g) --task Unitree-G1-29dof-Running --max_iterations 30000
set -uo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <host-uid> <host-gid> [train args...]" >&2
    exit 2
fi
HOST_UID=$1; HOST_GID=$2; shift 2

REPO_DIR="${UNITREE_RL_LAB_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

restore_owner() {
    for d in logs wandb outputs; do
        [[ -d "$d" ]] && chown -R "${HOST_UID}:${HOST_GID}" "$d" 2>/dev/null
    done
}
trap restore_owner EXIT

./unitree_rl_lab.sh -t "$@"
