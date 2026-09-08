#!/usr/bin/env bash
# 方案C 主机 GUI 一键启动
#   ./run_gui.sh              进程内数字孪生 (默认场景 point_press)
#   ./run_gui.sh udp          先起仿真 FPGA (sim_device), GUI 走 UDP 回环 127.0.0.1:5000
#   ./run_gui.sh board        接真实板子 192.168.2.128:5000
#   ./run_gui.sh replay x.npz 回放录制
# 可加环境变量: SCENE=sweep NOISE=sig2_matched ./run_gui.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"          # fem-2
source "$ROOT/venv/bin/activate"
cd "$HERE"
SCENE="${SCENE:-point_press}"; NOISE="${NOISE:-hardware}"
case "${1:-twin}" in
  twin)   exec python -m honeycomb_host.gui.main --source twin --scene "$SCENE" --noise "$NOISE" ;;
  udp)    python -m honeycomb_host.sim_device --scene "$SCENE" --noise "$NOISE" --drop "${DROP:-0}" &
          DEV=$!; trap "kill $DEV 2>/dev/null" EXIT; sleep 0.5
          python -m honeycomb_host.gui.main --source udp --device 127.0.0.1:5000 ;;
  board)  exec python -m honeycomb_host.gui.main --source udp --device "${DEVICE:-192.168.2.128:5000}" ;;
  replay) exec python -m honeycomb_host.gui.main --source replay --file "$2" ;;
  *) echo "用法: $0 [twin|udp|board|replay <file.npz>]"; exit 1 ;;
esac
