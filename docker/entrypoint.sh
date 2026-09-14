#!/usr/bin/env bash
# Container entrypoint: run the requested command, then hand written files back to the host user.
#
# The image must run as root (Isaac Sim's /isaac-sim tree is drwxr-x--- and OMNI_KIT_ALLOW_ROOT=1),
# so everything written under logs/, wandb/ and outputs/ would stay root-owned on the host. On a
# shared server without sudo those files cannot be deleted, moved or written into again. There is no
# uid to pass: the target owner is simply whoever owns the bind-mounted checkout (dev image) or the
# bind-mounted logs directory (self-contained image), so this works with no extra flags.
set -uo pipefail

REPO=${UNITREE_RL_LAB_PATH:-/workspace/unitree_rl_lab}

owner_of() { stat -c '%u %g' "$1" 2>/dev/null; }
HOST_OWNER=""
for probe in "${REPO}" "${REPO}/logs"; do
    read -r u g <<<"$(owner_of "${probe}")"
    if [[ -n "${u:-}" && "${u}" != "0" ]]; then HOST_OWNER="${u}:${g}"; break; fi
done

restore_owner() {
    [[ -n "${HOST_OWNER}" ]] || return 0
    for d in logs wandb outputs; do
        [[ -d "${REPO}/${d}" ]] && chown -R "${HOST_OWNER}" "${REPO}/${d}" 2>/dev/null
    done
    return 0
}
trap restore_owner EXIT

cd "${REPO}" || exit 1
if [[ $# -eq 0 ]]; then bash; else "$@"; fi
