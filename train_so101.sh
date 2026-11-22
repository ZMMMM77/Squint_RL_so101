#!/usr/bin/env bash
# Train the SO-101 pick-cube RL policy (MuJoCo Playground + RSL-RL PPO) on the
# RTX 3060 in WSL2. Code, logs and checkpoints all stay in this VM folder:
# each run lives in runs/<run>/ (train.log, model_*.pt, TensorBoard events).
#
#   ./train_so101.sh start place  [iterations]   pick the cube and drop it into the box (wrist camera on top, ~2.6 h)
#   ./train_so101.sh start vision [iterations]   pick the cube and lift it to a point (wrist camera on the side)
#   ./train_so101.sh start state  [iterations]   state-based policy (sim-only sanity check, ~25 min)
#   ./train_so101.sh status                      progress of the latest run
#   ./train_so101.sh log                         follow the training log (Ctrl+C only stops viewing)
#   ./train_so101.sh stop                        stop training (checkpoints are kept)
#   ./train_so101.sh resume [run] [iterations]   continue a run from its latest checkpoint
#   ./train_so101.sh list                        list runs
#   ./train_so101.sh board [stop]                live training curves in the browser (TensorBoard, port 6006)
#
# Real robot (runs here in the VM, where the arm and cameras are attached; CPU only):
#   ./train_so101.sh calibrate                   calibrate the follower arm (hold the middle pose, press ENTER)
#   ./train_so101.sh hover [seconds] [pick|place] real arm: go to the task's start pose, hold, then park (no policy)
#   ./train_so101.sh deploy sim|real [run]       run a trained policy (default: latest run's latest checkpoint;
#                                                the task comes from the run, a place run aims above the box)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ---- Training parameters tuned for the RTX 3060 Laptop (6 GB, ~0.9 GB used by Windows) ----
# Benchmarked 2026-11-21:
#   vision nworld=1024 -> 4.2 GB peak, ~6.4k steps/s, 2.6 s/iteration
#   vision nworld=2048 -> 6.0 GB, spills into system RAM, 14x slower; 4096 (repo default) -> out of memory
#   state  nworld=8192 -> 5.1 GB peak, ~132k steps/s, 1.5 s/iteration (repo default works)
VISION_ENV=SO101PickCubeVision
VISION_NWORLD=1024
VISION_ITERS=6000   # 6000 x 16 steps x 1024 envs = 98M steps, ~4.3 h (repo default: 3000 iters at 4096 envs)
PLACE_ENV=SO101PlaceCubeVision   # cube into the 12x12x7 cm box (mujoco_playground/.../so101/place_task.py)
PLACE_NWORLD=1024
PLACE_ITERS=3000    # ~2.6 h at ~5.3k steps/s (smoke-tested 2026-11-22); the pick task converged in ~1000 iterations
STATE_ENV=SO101PickCube
STATE_NWORLD=8192
STATE_ITERS=1000    # repo default, ~25 min
WARP_CACHE=/home/sam/.cache/warp_kernels   # compiled GPU kernels, kept in WSL2 across runs

# ---- Real robot (VM side) ----
DEPLOY_VENV=$HOME/.venvs/playground-deploy  # CPU env: playground + lerobot[feetech] + rsl-rl
# The InnoMaker wrist camera has no 320x240 mode (the deploy default): the driver would pick
# 640x360 and deploy would squash it to 4:3. Use the native 4:3 mode instead.
WRIST_CAM_W=640
WRIST_CAM_H=480
# MUJOCO_GL=egl: offscreen rendering that works both from the desktop and over SSH.
vmpy() { (cd mujoco_playground && MUJOCO_GL=${MUJOCO_GL:-egl} "$DEPLOY_VENV/bin/python" "$@"); }
DEPLOY_MOD=mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run
wrist_cam_free() {  # a camera preview window holding the wrist camera makes the real deploy fail
  local cam; cam=$(sed -n 's/.*"wrist": "\(.*\)".*/\1/p' mujoco_playground/.local/so101_cameras.json 2>/dev/null)
  if [[ -n "$cam" ]] && fuser "$cam" >/dev/null 2>&1; then
    echo "腕部相机正被别的程序占用 (e.g. a preview window): close it first."; return 1
  fi
}

RUNS=runs
mkdir -p "$RUNS"

latest_run() { cat "$RUNS/.latest" 2>/dev/null || true; }
# The deploy task (pick|place) a run was trained for, from its run.cfg.
run_task() { local e; e=$(sed -n 's/^env=//p' "$RUNS/$1/run.cfg" 2>/dev/null); [[ $e == "$PLACE_ENV" ]] && echo place || echo pick; }
is_running() { local p; p=$(cat "$RUNS/$1/vm.pid" 2>/dev/null) && kill -0 "$p" 2>/dev/null; }

launch() {  # run env nworld iters [resume_ckpt]
  local run=$1 env=$2 nworld=$3 iters=$4 ckpt=${5:-}
  # --alsologtostderr: put absl INFO lines (config, "Loading from checkpoint") in train.log
  local args=(python -m mujoco_playground.rl.rsl_rl.train --alsologtostderr "--env=$env"
              "--logdir=../$RUNS/$run" "--wp_kernel_cache_dir=$WARP_CACHE"
              "--overrides={\"nworld\": $nworld, \"max_iterations\": $iters}")
  [[ -n "$ckpt" ]] && args+=("--resume_path=../$RUNS/$run/$ckpt")
  mkdir -p "$RUNS/$run"
  printf 'env=%s\nnworld=%s\n' "$env" "$nworld" > "$RUNS/$run/run.cfg"
  echo "=== $(date '+%F %T') start: ${args[*]}" >> "$RUNS/$run/train.log"
  # setsid+nohup: training keeps going if this terminal or Claude disconnects.
  WSL_GPU_JOB_ID=$run setsid nohup ./wsl-gpu "${args[@]}" >> "$RUNS/$run/train.log" 2>&1 < /dev/null &
  echo $! > "$RUNS/$run/vm.pid"
  echo "$run" > "$RUNS/.latest"
  echo "Started $run  ($env, $nworld envs, $iters iterations)"
  echo "  log:    ./train_so101.sh log"
  echo "  status: ./train_so101.sh status"
  echo "  stop:   ./train_so101.sh stop"
}

case "${1:-}" in
  start)
    kind=${2:-vision}
    r=$(latest_run)
    if [[ -n "$r" ]] && is_running "$r"; then echo "$r is still running; stop it first."; exit 1; fi
    case "$kind" in
      place)  launch "so101_place_$(date +%Y%m%d_%H%M%S)"  $PLACE_ENV  $PLACE_NWORLD  "${3:-$PLACE_ITERS}" ;;
      vision) launch "so101_vision_$(date +%Y%m%d_%H%M%S)" $VISION_ENV $VISION_NWORLD "${3:-$VISION_ITERS}" ;;
      state)  launch "so101_state_$(date +%Y%m%d_%H%M%S)"  $STATE_ENV  $STATE_NWORLD  "${3:-$STATE_ITERS}" ;;
      *) echo "usage: $0 start place|vision|state [iterations]"; exit 1 ;;
    esac ;;
  resume)
    run=${2:-$(latest_run)}
    [[ -d "$RUNS/$run" ]] || { echo "no run '$run'"; exit 1; }
    if is_running "$run"; then echo "$run is already running"; exit 1; fi
    ckpt=$(ls "$RUNS/$run"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    [[ -n "$ckpt" ]] || { echo "no checkpoint in $RUNS/$run"; exit 1; }
    source "$RUNS/$run/run.cfg"
    case $env in $VISION_ENV) def=$VISION_ITERS ;; $PLACE_ENV) def=$PLACE_ITERS ;; *) def=$STATE_ITERS ;; esac
    echo "Resuming $run from model_$ckpt.pt"
    launch "$run" "$env" "$nworld" "${3:-$def}" "model_$ckpt.pt" ;;
  status)
    set +e +o pipefail   # missing log lines are normal here, not errors
    run=$(latest_run); [[ -n "$run" ]] || { echo "no runs yet"; exit 0; }
    if is_running "$run"; then state=RUNNING; else state=stopped; fi
    echo "run: $run  [$state]"
    log="$RUNS/$run/train.log"
    grep -E "Learning iteration" "$log" | tail -1 | sed 's/\x1b\[[0-9;]*m//g; s/^ *//'
    for k in "Metrics/success" "Metrics/in_box" "Metrics/cube_lifted" "Metrics/box_xy_err" "Mean reward" "Mean episode length" "Steps per second" "ETA"; do
      grep -E "^\s*$k:" "$log" | tail -1 | sed 's/^ */  /'
    done
    grep -iE "error|Traceback|out of memory" "$log" | tail -3
    echo "  checkpoints: $(ls "$RUNS/$run"/model_*.pt 2>/dev/null | wc -l)  (latest: $(ls -t "$RUNS/$run"/model_*.pt 2>/dev/null | head -1 | xargs -r basename))"
    if [[ $state == RUNNING ]]; then
      ./wsl-gpu nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
    fi ;;
  log)
    run=$(latest_run); [[ -n "$run" ]] || { echo "no runs yet"; exit 0; }
    tail -n 40 -f "$RUNS/$run/train.log" ;;
  stop)
    run=$(latest_run); [[ -n "$run" ]] || { echo "no runs yet"; exit 0; }
    ./wsl-gpu --stop "$run"
    p=$(cat "$RUNS/$run/vm.pid" 2>/dev/null) && kill "$p" 2>/dev/null || true
    echo "Stopped $run. Checkpoints kept in $RUNS/$run/" ;;
  board)
    # TensorBoard runs here in the VM (CPU) and reads the event files training writes into runs/.
    pidf=$RUNS/.tensorboard.pid
    if [[ ${2:-} == stop ]]; then
      p=$(cat "$pidf" 2>/dev/null) && kill "$p" 2>/dev/null; rm -f "$pidf"; echo "TensorBoard stopped"; exit 0
    fi
    if ! { p=$(cat "$pidf" 2>/dev/null) && kill -0 "$p" 2>/dev/null; }; then
      setsid nohup "$DEPLOY_VENV/bin/tensorboard" --logdir "$RUNS" --host 0.0.0.0 --port 6006 \
        --reload_interval 30 > "$RUNS/.tensorboard.log" 2>&1 < /dev/null &
      echo $! > "$pidf"
    fi
    echo "TensorBoard (refreshes every 30 s):"
    echo "  in this VM:   http://localhost:6006"
    echo "  from Windows: http://$(hostname -I | awk '{print $1}'):6006"
    echo "  stop:         ./train_so101.sh board stop" ;;
  list)
    set +e +o pipefail
    for d in "$RUNS"/*/; do
      [[ -d "$d" ]] || continue; r=$(basename "$d")
      it=$(grep -E "Learning iteration" "$d/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
      s=$(grep -E "^\s*Metrics/success:" "$d/train.log" 2>/dev/null | tail -1 | awk '{print $NF}')
      printf '%-34s iter %-12s success %-8s %s\n' "$r" "${it:--}" "${s:--}" "$(is_running "$r" && echo RUNNING)"
    done ;;
  calibrate)
    # The official get_ports --identify needs a leader AND a follower; there is only a follower here,
    # so write its port config directly from the one USB serial device that is attached.
    vmpy - <<'PY' || exit 1
import sys
from mujoco_playground._src import paths
from mujoco_playground._src.real import find_port
cfg = paths.local_dir() / "so101_ports.json"
usb = [p for p in find_port.list_ports() if p.serial_number]
present = {p.serial_number for p in usb}
if cfg.exists():
    import json
    serial = json.loads(cfg.read_text()).get("follower")
    if serial in present:
        sys.exit(0)
    print(f"Port config has follower serial {serial}, which is not attached; re-detecting.")
if len(usb) == 1:
    path = find_port.write_port_config("so101", {"follower": usb[0].serial_number})
    print(f"Follower arm: {usb[0].device} (serial {usb[0].serial_number}) -> {path}")
    sys.exit(0)
nameless = [p.device for p in find_port.list_ports() if not p.serial_number and ("ttyACM" in p.device or "ttyUSB" in p.device)]
if not usb and nameless:
    print(f"Found {nameless} but it reports no USB serial number, which playground needs to identify the arm.")
elif not usb:
    print("找不到机械臂的 USB 串口 (no USB serial device in this VM).\n"
          "  1. 舵机控制板接上电源, USB 插到笔记本上\n"
          "  2. VMware 菜单: 虚拟机 -> 可移动设备 -> 选择串口设备 (QinHeng / USB Single Serial) -> 连接\n"
          "  3. 再运行 ./train_so101.sh calibrate")
else:
    print("More than one USB serial device; unplug the others and retry:")
    print(find_port.format_ports(usb))
sys.exit(1)
PY
    echo "提醒: 中间姿势时腕部相机要朝【左】、两指上下排列 (mujoco_playground/mujoco_playground/robots/so101/real/assets/middle_pose.png)。"
    echo "      相机朝上会让手腕旋转差 90 度 (2026-11-22 修过一次, 旧文件 .local/so101_calibration.before_roll_fix_20261122.json)。"
    vmpy -m mujoco_playground.robots.so101.real.get_calibration --roles=follower ;;
  hover)
    wrist_cam_free || exit 1
    r=$(latest_run); task=${3:-$([[ -n "$r" ]] && run_task "$r" || echo pick)}
    [[ $task == pick || $task == place ]] || { echo "usage: $0 hover [seconds] [pick|place]"; exit 1; }
    echo "Real arm -> $task start pose for ${2:-10}s, then park and release. Keep a hand near the power switch."
    vmpy -m $DEPLOY_MOD --mode=real --task="$task" --noviewer --num_episodes=1 --episode_length_secs="${2:-10}" \
      --homing_duration_secs=3 --camera_width=$WRIST_CAM_W --camera_height=$WRIST_CAM_H \
      --mcap_log_dir="../$RUNS/.hover_test" ;;
  deploy)
    mode=${2:-}; [[ $mode == sim || $mode == real ]] || { echo "usage: $0 deploy sim|real [run] [extra flags]"; exit 1; }
    [[ $mode == real ]] && { wrist_cam_free || exit 1; }
    run=${3:-$(latest_run)}; [[ -n "$run" && -d "$RUNS/$run" ]] || { echo "no run '$run'"; exit 1; }
    ls "$RUNS/$run"/model_*.pt >/dev/null 2>&1 || { echo "no checkpoint in $RUNS/$run"; exit 1; }
    task=$(run_task "$run")
    echo "Deploying $run ($mode, task=$task); episode logs go to $RUNS/$run/deploy_logs"
    vmpy -m $DEPLOY_MOD --mode="$mode" --task="$task" \
      --checkpoint="../$RUNS/$run" --camera_width=$WRIST_CAM_W --camera_height=$WRIST_CAM_H \
      --mcap_log_dir="../$RUNS/$run/deploy_logs" "${@:4}" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
