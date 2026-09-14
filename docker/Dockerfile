# syntax=docker/dockerfile:1
#
# Image for unitree_rl_lab. Two build targets:
#
#   base  - repository + python dependencies. Robot description files and the Isaac
#           cloud assets stay outside the image (bind-mounted / downloaded at run time).
#   full  - self-contained: additionally bakes the Unitree robot description files and
#           a mirror of the Isaac cloud assets, so the container runs with `--network none`.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile --target base -t unitree-rl-lab:2.3.2 .
#   docker build -f docker/Dockerfile --target full -t unitree-rl-lab:2.3.2-selfcontained .
# or via compose (see docker/docker-compose.yaml). Full guide: docker/README.md

# Isaac Lab base image. The public NGC image ships Isaac Sim 5.1.0 + Isaac Lab 2.3.x for
# both linux/amd64 and linux/arm64, so no locally built `isaac-lab-base` image is needed.
ARG ISAACLAB_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-lab:2.3.2
# Where the `full` target takes the robot description files from: `clone` (download during
# the build) or `local` (copy from the named build contexts unitree_model / unitree_ros).
ARG ASSETS_SOURCE=clone


# ---------------------------------------------------------------------------------------
# base: repository + python dependencies
# ---------------------------------------------------------------------------------------
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_UNITREE_RL_LAB_PATH_ARG=/workspace/unitree_rl_lab
ENV UNITREE_RL_LAB_PATH=${DOCKER_UNITREE_RL_LAB_PATH_ARG}

# Locations of the Unitree robot description files inside the container. They are read by
# source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py, which honours these
# environment variables. Bind-mount them (target `base`) or bake them in (target `full`).
ENV UNITREE_MODEL_DIR=/workspace/unitree_model
ENV UNITREE_ROS_DIR=/workspace/unitree_ros
# Local mirror of the Isaac cloud assets, populated by the `full` target.
ENV ISAAC_ASSETS_DIR=/workspace/isaac_assets

USER root
SHELL ["/bin/bash", "-c"]

# git-lfs is needed to fetch the USD files of the unitree_model dataset.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && git lfs install --system \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the repository (exclusions are listed in .dockerignore at the repository root).
COPY . ${UNITREE_RL_LAB_PATH}

# Install the extension into the python environment that ships with Isaac Sim / Isaac Lab.
# `isaaclab.sh -p` resolves to that interpreter; ISAACLAB_PATH is set by the base image.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
        -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab \
        argcomplete wandb matplotlib \
    && ${ISAACLAB_PATH}/isaaclab.sh -p -c "import unitree_rl_lab, argcomplete, prettytable, psutil; print('[OK] unitree_rl_lab installed')" \
    # task-name autocompletion for ./unitree_rl_lab.sh in interactive shells
    && echo "source ${UNITREE_RL_LAB_PATH}/unitree_rl_lab.sh" >> ${HOME}/.bashrc

# The base image starts Isaac Sim headless by default; this image is driven by explicit
# commands instead (see docker/README.md).
ENTRYPOINT []
CMD ["bash"]
WORKDIR ${UNITREE_RL_LAB_PATH}


# ---------------------------------------------------------------------------------------
# Robot description files for the `full` target: one of these two stages is selected by
# the ASSETS_SOURCE build argument. Only the selected stage is built.
# ---------------------------------------------------------------------------------------

# ASSETS_SOURCE=clone (default): download during the build. Needs network access.
FROM base AS assets-clone
ARG UNITREE_MODEL_REPO=https://huggingface.co/datasets/unitreerobotics/unitree_model
ARG UNITREE_ROS_REPO=https://github.com/unitreerobotics/unitree_ros.git
RUN set -eux; \
    if [ -n "${UNITREE_MODEL_REPO}" ]; then \
        git clone --depth 1 "${UNITREE_MODEL_REPO}" "${UNITREE_MODEL_DIR}"; \
        rm -rf "${UNITREE_MODEL_DIR}/.git"; \
    fi; \
    if [ -n "${UNITREE_ROS_REPO}" ]; then \
        git clone --depth 1 "${UNITREE_ROS_REPO}" "${UNITREE_ROS_DIR}"; \
        rm -rf "${UNITREE_ROS_DIR}/.git"; \
    fi

# ASSETS_SOURCE=local: copy from directories already present on the build machine, e.g.
#   docker build -f docker/Dockerfile --target full \
#     --build-arg ASSETS_SOURCE=local \
#     --build-context unitree_model=$HOME/unitree_model \
#     --build-context unitree_ros=$HOME/unitree_ros -t unitree-rl-lab:2.3.2-selfcontained .
FROM base AS assets-local
COPY --from=unitree_model --exclude=.git . ${UNITREE_MODEL_DIR}
COPY --from=unitree_ros --exclude=.git . ${UNITREE_ROS_DIR}


# ---------------------------------------------------------------------------------------
# full: self-contained image, runs without network access
# ---------------------------------------------------------------------------------------
FROM assets-${ASSETS_SOURCE} AS full

# Mirror the Isaac cloud assets the tasks reference (markers, terrain material, sky texture)
# and point Kit's asset root at the mirror, so nothing is fetched from the internet.
# To use the online asset server again, run with:
#   --kit_args "--/persistent/isaac/asset_root/cloud=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
ARG ISAAC_ASSET_SERVER=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1
RUN ${UNITREE_RL_LAB_PATH}/docker/fetch_isaac_assets.sh "${ISAAC_ASSETS_DIR}" "${ISAAC_ASSET_SERVER}" \
        "${UNITREE_RL_LAB_PATH}/docker/isaac_assets.txt" \
    && sed -i -E "s#^(persistent\.isaac\.asset_root\.(default|cloud|nvidia)) = .*#\1 = \"${ISAAC_ASSETS_DIR}\"#" \
        ${ISAACLAB_PATH}/apps/*.kit \
    && grep -h "asset_root" ${ISAACLAB_PATH}/apps/isaaclab.python.headless.kit

WORKDIR ${UNITREE_RL_LAB_PATH}
