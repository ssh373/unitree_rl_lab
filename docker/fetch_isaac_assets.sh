#!/usr/bin/env bash
# Mirror the handful of Isaac Sim / Isaac Lab cloud assets that unitree_rl_lab needs,
# so the container can run with no network access.
#
# Usage: fetch_isaac_assets.sh <destination-dir> [asset-root-url] [file-list]
#
# The destination directory becomes the value of the Kit setting
# `persistent.isaac.asset_root.cloud`, i.e. Isaac Lab's ISAAC_NUCLEUS_DIR is
# <destination-dir>/Isaac.
set -euo pipefail

DEST=${1:?"usage: $0 <destination-dir> [asset-root-url] [file-list]"}
ROOT=${2:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1}
LIST=${3:-"$(dirname "$(readlink -f "$0")")/isaac_assets.txt"}

mkdir -p "${DEST}"
while read -r rel; do
    # skip blank lines and comments
    [[ -z "${rel}" || "${rel}" == \#* ]] && continue
    mkdir -p "${DEST}/$(dirname "${rel}")"
    echo "[fetch] ${rel}"
    curl -fSL --retry 3 --retry-delay 2 -o "${DEST}/${rel}" "${ROOT}/${rel}"
done < "${LIST}"

echo "[fetch] done: $(du -sh "${DEST}" | cut -f1) in ${DEST}"
