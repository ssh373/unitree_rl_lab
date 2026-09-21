# Docker 가이드 (unitree_rl_lab)

Isaac Lab 기반 `unitree_rl_lab`을 Docker 이미지로 빌드하고, 네트워크 없이도 그대로 돌아가는
**self-contained 이미지**를 만드는 방법을 정리한 문서입니다.

---

## 1. 처음에 났던 빌드 에러와 원인

```
ERROR: failed to build: failed to solve: base name (${ISAACLAB_BASE_IMAGE_ARG}) should not be blank
```

원인은 두 가지가 겹쳐 있었습니다.

**(1) `docker/.env.base`는 compose가 자동으로 읽지 않습니다.**
`docker-compose.yaml`의 `env_file:`은 *컨테이너 안의 환경 변수*만 채웁니다.
`build.args`에 쓰인 `${ISAACLAB_BASE_IMAGE}` 같은 값은 compose가 파일을 파싱하는 시점에
셸 환경 또는 프로젝트 디렉터리의 `.env`에서만 치환합니다. 그래서 값이 빈 문자열이 되고,
`FROM ${ISAACLAB_BASE_IMAGE_ARG}`가 비어 있다는 에러가 납니다.
compose가 자동으로 읽는 파일 이름은 `.env`이지 `.env.base`가 아닙니다.

**(2) 값을 채워도 그 이미지가 없습니다.**
`--env-file docker/.env.base`로 값을 넣으면 `isaac-lab-base`로 치환되는데, 이 이미지는
IsaacLab 저장소에서 직접 빌드해야 생기는 로컬 전용 태그입니다. 없으면 이렇게 실패합니다.

```
failed to solve: isaac-lab-base: pull access denied, repository does not exist ...
```

**해결 방법**은 공개 NGC 이미지를 기본값으로 박아두는 것입니다. `nvcr.io/nvidia/isaac-lab`은
로그인 없이 받을 수 있고 `linux/amd64`, `linux/arm64` 두 아키텍처를 모두 제공합니다.

- `docker/Dockerfile`: `ARG ISAACLAB_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-lab:2.3.2` (기본값 부여)
- `docker/docker-compose.yaml`: 모든 변수를 `${VAR:-기본값}` 형태로 변경

이제 `.env` 파일이 없어도 아래 명령이 바로 동작합니다.

```bash
docker compose -f docker/docker-compose.yaml build
```

---

## 2. 사전 요구사항

| 항목 | 요구 사항 |
| --- | --- |
| NVIDIA 드라이버 | 570.169 이상 (Isaac Sim 5.1 기준) |
| 런타임 | `nvidia-container-toolkit` 설치 후 `docker info`에 `nvidia` 런타임이 보여야 함 |
| Docker | BuildKit이 있는 버전 (Docker 23+, Compose v2) |
| 디스크 | 개발용 26 GB, self-contained 추가 약 2 GB |

검증에 사용한 환경: Docker 28.5.1 / Compose v2.40.0 / buildx 0.29.1, `aarch64`(NVIDIA GB10),
드라이버 580.95.05. `x86_64`에서도 같은 명령이 그대로 동작합니다.

---

## 3. 이미지 두 종류

`docker/Dockerfile`은 멀티스테이지이고 타깃이 두 개입니다.

| 타깃 | 태그(기본) | 내용 | 로봇 파일 | 네트워크 |
| --- | --- | --- | --- | --- |
| `base` | `unitree-rl-lab:2.3.2` | 저장소 + 파이썬 의존성 | 호스트에서 bind mount | 실행 중 NVIDIA 에셋 서버 접근 필요 |
| `full` | `unitree-rl-lab:2.3.2-selfcontained` | 위 + 로봇 파일 + Isaac 에셋 미러 | 이미지에 포함 | 불필요 (`--network none` 가능) |

`base`는 코드 수정이 즉시 반영되는 개발용, `full`은 그대로 배포·반출하는 실행용입니다.

---

## 4. 개발용 이미지 (`base`)

### 4.1 로봇 description 파일 받기

```bash
git lfs install
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model ~/unitree_model
git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros
```

compose는 기본적으로 이 두 경로를 컨테이너의 `/workspace/unitree_model`,
`/workspace/unitree_ros`로 마운트합니다. 다른 위치에 두었다면 `docker/.env`에서 바꾸세요.

```bash
cp docker/.env.base docker/.env   # 필요할 때만. 값은 전부 기본값과 동일합니다.
```

컨테이너 안에서는 `UNITREE_MODEL_DIR`, `UNITREE_ROS_DIR` 환경 변수가 이 경로를 가리키고,
`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`가 이 변수를 읽습니다.
따라서 소스 코드를 직접 고칠 필요가 없습니다.

### 4.2 빌드와 실행

```bash
# 빌드 (베이스 이미지 내려받기 제외 약 5분)
docker compose -f docker/docker-compose.yaml build

# 셸 열기
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab

# 태스크 목록
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab ./unitree_rl_lab.sh -l

# 학습
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 4096

# 추론 (헤드리스). 가장 최근 체크포인트를 자동으로 찾습니다.
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab \
  ./unitree_rl_lab.sh -p --headless --task Unitree-G1-29dof-Velocity --num_envs 16
```

저장소 전체가 bind mount되므로 호스트에서 코드를 고치면 컨테이너에 즉시 반영되고,
학습 로그와 체크포인트는 호스트의 `logs/`에 남습니다.

Isaac Sim의 셰이더 캐시는 named volume에 보존되므로 두 번째 실행부터 기동이 빨라집니다.

---

## 5. Self-contained 이미지 (`full`)

이미지 하나에 다음이 모두 들어갑니다.

1. 저장소 코드와 `pip install -e`로 설치된 `unitree_rl_lab` 패키지
2. Unitree 로봇 description 파일 (`unitree_model`, `unitree_ros`)
3. Isaac Sim이 실행 중 내려받는 클라우드 에셋의 로컬 미러 (약 23 MB)
4. 그 미러를 바라보도록 수정된 Kit 설정

4번이 핵심입니다. 태스크는 지형 재질, 하늘 HDR, 화살표 마커 USD를 NVIDIA 에셋 서버에서
받아옵니다. 이 파일이 없으면 오프라인에서 서버 접속을 300초 동안 기다리다 실패합니다.
빌드 시 `docker/isaac_assets.txt`에 적힌 파일만 내려받아 `/workspace/isaac_assets`에 두고,
`${ISAACLAB_PATH}/apps/*.kit`의 `persistent.isaac.asset_root.{default,cloud,nvidia}`를
그 경로로 바꿉니다.

### 5.1 빌드 방식 A: 빌드할 때 내려받기 (기본)

```bash
docker compose -f docker/docker-compose.yaml build unitree-rl-lab-selfcontained
```

또는 compose 없이:

```bash
docker build -f docker/Dockerfile --target full -t unitree-rl-lab:2.3.2-selfcontained .
```

빌드 중에 Hugging Face와 GitHub, NVIDIA 에셋 서버에 접근합니다.

### 5.2 빌드 방식 B: 이미 받아둔 디렉터리 사용

에셋을 이미 갖고 있거나 빌드 머신이 외부망에 못 나갈 때 씁니다.
`--build-context`로 넘긴 디렉터리를 그대로 복사하며, `.git`은 제외합니다.

```bash
docker build -f docker/Dockerfile --target full \
  --build-arg ASSETS_SOURCE=local \
  --build-context unitree_model=$HOME/unitree_model \
  --build-context unitree_ros=$HOME/unitree_ros \
  -t unitree-rl-lab:2.3.2-selfcontained .
```

Isaac 에셋 미러(23 MB)는 이 방식에서도 빌드 중에 내려받습니다. 완전 오프라인 빌드가
필요하면 미리 받아두고 사내 미러 주소를 넘기세요.

```bash
./docker/fetch_isaac_assets.sh /srv/isaac_assets          # 인터넷 되는 곳에서 한 번
docker build ... --build-arg ISAAC_ASSET_SERVER=http://내부미러/Assets/Isaac/5.1 ...
```

### 5.3 실행

마운트가 하나도 필요 없습니다.

```bash
docker run --rm --gpus all --network none \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  unitree-rl-lab:2.3.2-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 4096
```

학습 결과를 꺼내려면 로그 디렉터리만 연결하면 됩니다.

```bash
docker run --rm --gpus all --network none \
  -e OMNI_KIT_ALLOW_ROOT=1 -v $PWD/logs:/workspace/unitree_rl_lab/logs \
  unitree-rl-lab:2.3.2-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

compose로도 같은 동작을 합니다. 이때는 `logs/`만 bind mount됩니다.

```bash
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity
```

### 5.4 다시 온라인 에셋을 쓰고 싶을 때

미러에 없는 Isaac 에셋이 필요한 태스크를 추가했다면, 두 가지 중 하나를 하세요.

- `docker/isaac_assets.txt`에 경로를 추가하고 다시 빌드
- 실행할 때만 원래 서버로 되돌리기

```bash
docker run --rm --gpus all -e OMNI_KIT_ALLOW_ROOT=1 unitree-rl-lab:2.3.2-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity \
  --kit_args "--/persistent/isaac/asset_root/cloud=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
```

### 5.5 폐쇄망으로 반출

```bash
docker save unitree-rl-lab:2.3.2-selfcontained | zstd -T0 -o unitree-rl-lab-sc.tar.zst
# 옮긴 뒤
zstd -d -c unitree-rl-lab-sc.tar.zst | docker load
```

압축 전 기준 약 28 GB입니다. 대상 장비의 드라이버가 570.169 이상인지 먼저 확인하세요.

---

## 6. 측정값과 검증 결과

빌드 시간과 크기 (베이스 이미지는 이미 받아둔 상태, GB10 기준):

| 항목 | 값 |
| --- | --- |
| `base` 타깃 빌드 | 4분 50초 |
| `full` 타깃 추가 빌드 | 2분 22초 |
| `unitree-rl-lab:2.3.2` | 26 GB |
| `unitree-rl-lab:2.3.2-selfcontained` | 27.7 GB |
| Isaac 에셋 미러 | 23 MB |

실제로 확인한 동작:

- 태스크 5개 목록 출력 (`./unitree_rl_lab.sh -l`)
- G1-29dof 속도 태스크 학습, 체크포인트와 텐서보드 로그 생성
- self-contained 이미지에서 `--network none`, 마운트 없이 G1 학습 성공
- 같은 조건에서 Go2 학습 성공 (방식 B로 빌드한 이미지)
- `play.py`로 체크포인트 로드 후 `policy.pt`, `policy.onnx` 내보내기

빠르게 다시 확인하는 방법:

```bash
docker run --rm --gpus all --network none -e OMNI_KIT_ALLOW_ROOT=1 \
  unitree-rl-lab:2.3.2-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 64 --max_iterations 2
```

`Learning iteration 1/2`까지 찍히고 종료 코드가 0이면 정상입니다.

---

## 7. 설정 항목 정리

빌드 인자 (`--build-arg`):

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `ISAACLAB_BASE_IMAGE_ARG` | `nvcr.io/nvidia/isaac-lab:2.3.2` | 베이스 이미지 |
| `DOCKER_UNITREE_RL_LAB_PATH_ARG` | `/workspace/unitree_rl_lab` | 컨테이너 안 저장소 경로 |
| `ASSETS_SOURCE` | `clone` | `full` 타깃의 로봇 파일 출처 (`clone` 또는 `local`) |
| `UNITREE_MODEL_REPO` | Hugging Face `unitree_model` | 빈 문자열이면 건너뜀 |
| `UNITREE_ROS_REPO` | GitHub `unitree_ros` | 빈 문자열이면 건너뜀 |
| `ISAAC_ASSET_SERVER` | NVIDIA S3 주소 | 사내 미러로 교체 가능 |

컨테이너 환경 변수:

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `UNITREE_MODEL_DIR` | `/workspace/unitree_model` | USD 로봇 파일 경로 |
| `UNITREE_ROS_DIR` | `/workspace/unitree_ros` | URDF 로봇 파일 경로 |
| `ISAAC_ASSETS_DIR` | `/workspace/isaac_assets` | Isaac 에셋 미러 경로 (`full`만) |
| `ISAACLAB_PATH` | `/workspace/isaaclab` | 베이스 이미지가 설정 |

compose 변수는 `docker/.env.base`에 주석과 함께 정리되어 있습니다. 쓰려면 `docker/.env`로
복사하세요. 이 파일은 `.gitignore`와 `.dockerignore`에 들어 있습니다.

---

## 8. 문제 해결

**`base name (${ISAACLAB_BASE_IMAGE_ARG}) should not be blank`**
1장 참고. 예전 파일을 쓰고 있거나 변수를 직접 넘기다 비운 경우입니다.
`docker compose -f docker/docker-compose.yaml config`로 치환 결과를 먼저 확인하세요.

**`failed to unmarshal the received message: string field contains invalid UTF-8`**
빌드 컨텍스트에 파일 이름이 UTF-8이 아닌 파일이 있으면 buildx의 `docker-container` 드라이버가
실패합니다. 이 저장소 루트에 그런 빈 파일 두 개가 남아 있으면 `--build-context`를 쓰는 빌드가
막힙니다. 확인과 삭제:

```bash
# 1) 어떤 파일인지 먼저 확인 (이름, 크기)
python3 -c "
import os
for n in os.listdir(b'.'):
    try: n.decode()
    except UnicodeDecodeError: print(repr(n), os.stat(n).st_size, 'bytes')
"

# 2) 내용이 없는 쓰레기 파일이 맞다면 삭제
python3 -c "
import os
for n in os.listdir(b'.'):
    try: n.decode()
    except UnicodeDecodeError:
        if os.path.isfile(n) and os.path.getsize(n) == 0:
            os.remove(n); print('removed', repr(n))
"
```

크기가 0이 아니면 삭제하지 말고 무엇인지 먼저 확인하세요.

**오프라인인데 300초씩 멈춘다**
`Checking server availability for USD path: https://...` 로그가 보이면 Kit이 아직 클라우드
에셋 서버를 보고 있다는 뜻입니다. `full` 타깃으로 빌드했는지, 또는 `--kit_args`로 asset_root를
로컬 미러로 지정했는지 확인하세요.

**`ModuleNotFoundError: isaaclab.utils.pretrained_checkpoint`**
Isaac Lab 2.3.2에서 이 헬퍼가 `isaaclab_rl.utils.pretrained_checkpoint`로 옮겨졌습니다.
`scripts/rsl_rl/play.py`는 두 위치를 모두 시도하도록 되어 있어 2.3.1과 2.3.2에서 함께 동작합니다.

**로그 파일이 root 소유로 생긴다**
컨테이너가 root로 돌기 때문입니다 (Isaac Sim은 `OMNI_KIT_ALLOW_ROOT=1`을 요구).

```bash
sudo chown -R $(id -u):$(id -g) logs
```

**GUI로 보고 싶다**
헤드리스만 검증했습니다. 화면이 있는 장비라면 아래를 추가하고 `--headless`를 빼면 됩니다.
드라이버와 X 설정에 따라 추가 조정이 필요할 수 있습니다.

```bash
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
```

**베이스 이미지 태그를 바꾸고 싶다**
`ISAACLAB_BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.1`처럼 지정하면 됩니다. 태그마다 Isaac Lab
API가 조금씩 다르니 바꾼 뒤에는 학습 한 번을 짧게 돌려 확인하세요.

---

## 9. 파일 목록

| 파일 | 역할 |
| --- | --- |
| `docker/Dockerfile` | `base` / `full` 두 타깃의 멀티스테이지 정의 |
| `docker/docker-compose.yaml` | `unitree-rl-lab`, `unitree-rl-lab-selfcontained` 서비스 |
| `docker/.env.base` | compose 변수 템플릿 (`docker/.env`로 복사해서 사용) |
| `docker/isaac_assets.txt` | 미러링할 Isaac 클라우드 에셋 목록 |
| `docker/fetch_isaac_assets.sh` | 그 목록을 내려받는 스크립트 (호스트에서도 사용 가능) |

---

## 공용 서버(sudo 없음)에서 돌리기

컨테이너는 root로 돌아야 합니다(이미지의 `/isaac-sim`이 `drwxr-x---`라 `--user`로는 실행조차 되지 않습니다).
그래서 이미지 ENTRYPOINT(`docker/entrypoint.sh`)가 명령이 끝날 때 `logs/`, `wandb/`, `outputs/`의 소유자를
호스트 사용자로 되돌립니다. 대상 uid는 바인드 마운트된 체크아웃(개발용) 또는 `logs/`(self-contained)의
소유자에서 읽으므로 **추가 플래그가 필요 없습니다**. 학습이 실패하거나 중단되어도 실행됩니다.
아래처럼 평소 명령을 그대로 쓰면 되고, `docker/train.sh $(id -u) $(id -g) ...`로 uid를 명시해도 됩니다.

```bash
# 개발용 이미지(코드 바인드 마운트) — 서버에 저장소·로봇 모델이 있고 외부 네트워크가 열린 경우
docker compose -f docker/docker-compose.yaml run --rm unitree-rl-lab \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Running --max_iterations 30000

# self-contained 이미지 — 서버에 네트워크가 없거나 저장소를 두지 않을 경우(logs만 마운트)
docker run --rm --gpus all -e OMNI_KIT_ALLOW_ROOT=1 \
  -v $HOME/rl_logs:/workspace/unitree_rl_lab/logs \
  unitree-rl-lab:2.3.2-selfcontained \
  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Running --max_iterations 30000
```

- wandb 로거를 쓰면 `-e WANDB_API_KEY=...`를 넘기고, 외부 네트워크가 없으면 `-e WANDB_MODE=offline`으로
  돌린 뒤 `wandb/offline-run-*` 디렉터리를 로그와 함께 옮겨 나중에 `wandb sync` 합니다.
- 서버 전제: `docker ps`가 되고(docker 그룹), `docker info | grep -i nvidia`에 런타임이 보여야 합니다.
- 이미지에는 `wandb`, `matplotlib`이 추가로 설치되어 있습니다(`scripts/rsl_rl/train.py`, `plot_energy.py`).

### 학습 영상을 wandb에 올리기 (`--video`)

컨테이너에서 헤드리스 카메라 렌더링과 `WandbRecordVideo` 업로드가 동작하는 것을 확인했습니다
(`NVIDIA_DRIVER_CAPABILITIES=all`은 compose에 들어 있음). wandb 인증은 호스트의 `~/.netrc`를 읽기 전용으로
마운트하거나 `-e WANDB_API_KEY=...`로 넘깁니다. 프로젝트는 `--log_project_name`으로 지정합니다(기본 `isaaclab`).

```bash
docker compose -f docker/docker-compose.yaml run --rm \
  -v $HOME/.netrc:/root/.netrc:ro \
  unitree-rl-lab ./unitree_rl_lab.sh -t \
  --task Unitree-G1-29dof-Running-Reward --max_iterations 30000 \
  --logger wandb --log_project_name isaaclab \
  --video --video_interval 4800 --video_length 500
```

- `--video_interval`/`--video_length`는 환경 스텝 단위입니다(1 이터레이션 = 24 스텝). 지금까지의 러닝 실행은
  모두 4800/500을 썼습니다: 200 이터레이션마다 10초 클립이 `logs/.../videos/train/`에 저장되고 wandb의
  `Videos/train`에 올라갑니다(30000 이터레이션에 150개). train.py 기본값(2000/200)과 다르니 명시하세요.
- rsl_rl의 wandb 로거는 체크포인트(`model_*.pt`)도 wandb 실행 디렉터리에 복사합니다. 런당 수 GB가 `wandb/` 아래에
  쌓이므로 self-contained 이미지에서는 **`wandb/`도 반드시 마운트**해야 컨테이너 종료 시 사라지지 않습니다
  (`-v $HOME/rl_wandb:/workspace/unitree_rl_lab/wandb`). 오프라인 모드(`WANDB_MODE=offline`)에서는 이 디렉터리를
  나중에 `wandb sync`하면 됩니다. `train.sh`가 종료 시 `wandb/`의 소유자도 되돌립니다.

### 이미 root 소유가 된 로그 되돌리기

ENTRYPOINT가 들어가기 전에 돌린 실행이 남긴 root 소유 파일은 컨테이너 안에서 한 번 정리하면 됩니다.

```bash
docker compose -f docker/docker-compose.yaml run --rm --entrypoint bash unitree-rl-lab \
  -c "chown -R $(id -u):$(id -g) logs wandb"
```
