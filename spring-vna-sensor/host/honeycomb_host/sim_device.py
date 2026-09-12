"""仿真 FPGA: UDP 服务端, 按帧率发 0x11 驻留帧, 响应 adda_project 风格命令. 可注入丢包/乱序/饱和.
用法: python -m honeycomb_host.sim_device [--scene point_press] [--rate 77] [--drop 0.0] [--host 127.0.0.1]
"""
from __future__ import annotations
import argparse
import asyncio
import socket
import time
import numpy as np
from . import geometry as G
from .twin import Twin, Scenes, Environment, NoiseModel
from . import protocol as P

class SimDevice:
    def __init__(self, twin: Twin, host='127.0.0.1', cmd_port=P.FPGA_PORT, data_host='127.0.0.1',
                 data_port=P.DATA_PORT, drop=0.0, reorder=0.0, rate_hz: float | None = None, verbose=False):
        self.twin = twin; self.host = host; self.cmd_port = cmd_port
        self.data_addr = (data_host, data_port)
        self.drop = drop; self.reorder = reorder; self.verbose = verbose
        self.period = 1.0 / rate_hz if rate_hz else twin.env.frame_period_s
        self.running = True
        self.regs = {P.REG['DEVICE_ID']: P.DEVICE_ID, P.REG['FRAME_CTRL']: P.FRAME_CTRL_RUN,
                     P.REG['DWELL_NSAMP']: twin.env.dwell_nsamp, P.REG['NCO_FREQ_WORD']: twin.env.nco_word,
                     P.REG['N_DWELL']: len(twin.dwell_table)}
        self._tbl_full = np.zeros(64, np.uint16); self._tbl_full[:len(twin.dwell_table)] = twin.dwell_table
        self._tbl_addr = 0
        self.rng = np.random.default_rng(7)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.stats = dict(sent=0, dropped=0, cmds=0)

    def handle_cmd(self, b: bytes, addr):
        try:
            op, _, seq, reg, ch, ln, data = P.unpack_cmd(b)
        except Exception:
            return
        self.stats['cmds'] += 1
        st = 0; payload = 0
        if op == P.OP_IDENTIFY:
            payload = P.DEVICE_ID
        elif op == P.OP_REG_READ:
            payload = self.regs.get(reg, 0)
        elif op == P.OP_REG_WRITE:
            self.regs[reg] = data
            if reg == P.REG['DWELL_TABLE_ADDR']:
                self._tbl_addr = data
            elif reg == P.REG['DWELL_TABLE_DATA']:
                self._tbl_full[self._tbl_addr] = data & 0xFFFF
                if self._tbl_addr < len(self.twin.dwell_table):
                    self.twin.dwell_table[self._tbl_addr] = data & 0xFFFF
                self._tbl_addr = (self._tbl_addr + 1) & 63       # 固件语义: 写后地址自增
            elif reg == P.REG['N_DWELL']:
                n = max(1, min(64, int(data)))
                self.twin.dwell_table = self._tbl_full[:n].copy()
            elif reg == P.REG['DWELL_NSAMP']:
                self.twin.env.dwell_nsamp = int(data)
            elif reg == P.REG['HOST_PORT']:
                self.data_addr = (self.data_addr[0], int(data))
        else:
            st = 1
        self.sock.sendto(P.pack_rsp(op, st, seq, payload), addr)
        if self.verbose:
            print(f'cmd op={op:#x} reg={reg:#x} data={data:#x} from {addr}')

    async def run(self):
        loop = asyncio.get_running_loop()
        self.sock.bind((self.host, self.cmd_port))
        print(f'sim_device: cmd {self.host}:{self.cmd_port} → data {self.data_addr}, period {self.period*1e3:.1f}ms')
        next_t = time.perf_counter()
        pending = []
        while self.running:
            # 命令
            while True:
                try:
                    b, addr = self.sock.recvfrom(2048)
                except BlockingIOError:
                    break
                self.handle_cmd(b, addr)
            if self.regs.get(P.REG['FRAME_CTRL'], 1) != P.FRAME_CTRL_STOP:
                fr = self.twin.step(self.period)
                pkt = P.encode_frame(fr)
                if self.rng.random() < self.drop:
                    self.stats['dropped'] += 1
                else:
                    if self.rng.random() < self.reorder and pending:
                        pending.append(pkt); pkt = pending.pop(0)
                    self.sock.sendto(pkt, self.data_addr); self.stats['sent'] += 1
                    if self.regs[P.REG['FRAME_CTRL']] == P.FRAME_CTRL_SINGLE:
                        self.regs[P.REG['FRAME_CTRL']] = P.FRAME_CTRL_STOP
            next_t += self.period
            dt = next_t - time.perf_counter()
            if dt > 0:
                await asyncio.sleep(dt)
            else:
                next_t = time.perf_counter()
                await asyncio.sleep(0)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='point_press', choices=Scenes.ALL)
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--data-host', default='127.0.0.1')
    ap.add_argument('--cmd-port', type=int, default=P.FPGA_PORT); ap.add_argument('--data-port', type=int, default=P.DATA_PORT)
    ap.add_argument('--rate', type=float, default=None); ap.add_argument('--drop', type=float, default=0.0)
    ap.add_argument('--noise', default='hardware', choices=['hardware', 'sig2_matched', 'off'])
    ap.add_argument('--shim', default=None, help='垫片场景 unit:gap_mm[:tilt_deg[:tilt_dir_deg[:dx[:dy]]]] (其余单元无环), 覆盖 --scene')
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args(argv)
    env = Environment(noise=NoiseModel(preset=a.noise))
    if a.shim:
        v = [float(x) for x in a.shim.split(':')] + [0.0] * 6
        scene = Scenes.shim(int(v[0]), v[1], v[2], v[3], v[4], v[5])
    else:
        scene = getattr(Scenes, a.scene)()
    tw = Twin(env=env, scene=scene)
    dev = SimDevice(tw, a.host, a.cmd_port, a.data_host, a.data_port, a.drop, rate_hz=a.rate, verbose=a.v)
    try:
        asyncio.run(dev.run())
    except KeyboardInterrupt:
        print('stats', dev.stats)

if __name__ == '__main__':
    main()
