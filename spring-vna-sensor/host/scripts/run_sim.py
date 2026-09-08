#!/usr/bin/env python3
"""一键演示: 进程内孪生 → GUI (默认), 或 --udp: 起 sim_device 子进程 + GUI 走 UDP 回环.
用法: python scripts/run_sim.py [--udp] [--scene point_press] [--noise hardware]"""
import argparse, subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument('--udp', action='store_true'); ap.add_argument('--scene', default='point_press')
ap.add_argument('--noise', default='hardware'); ap.add_argument('--drop', type=float, default=0.0)
a = ap.parse_args()
from honeycomb_host.gui.main import main
if a.udp:
    dev = subprocess.Popen([sys.executable, '-m', 'honeycomb_host.sim_device', '--scene', a.scene, '--noise', a.noise,
                            '--drop', str(a.drop)], cwd=ROOT)
    time.sleep(0.5)
    try:
        sys.exit(main(['--source', 'udp', '--device', '127.0.0.1:5000']))
    finally:
        dev.terminate()
else:
    sys.exit(main(['--source', 'twin', '--scene', a.scene, '--noise', a.noise]))
