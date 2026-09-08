"""数据源 (QThread): TwinSource (进程内孪生) / UdpSource (FPGA 或 sim_device) / ReplaySource (npz).
统一接口: start()/stop()/send_command(); 帧经 frame_ready 信号或 queue 交给 PipelineWorker."""
from __future__ import annotations
import queue
import socket
import threading
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from .twin import Twin, Scenes, Environment, Frame
from . import protocol as P
from .pipeline import Pipeline, FrameResult
from . import recorder

class DataSource(QThread):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
    def stop(self):
        self._stop.set(); self.wait(2000)
    def send_command(self, cmd: P.Command):
        pass

class TwinSource(DataSource):
    def __init__(self, twin: Twin, realtime: bool = True, parent=None):
        super().__init__(parent)
        self.twin = twin; self.realtime = realtime
        self._setters = queue.Queue()
        self.lock = threading.Lock()
    def set_scene(self, scene):
        self._setters.put(lambda: setattr(self.twin, 'scene', scene) or setattr(self.twin, 't', 0.0))
    def set_param(self, fn):
        """fn(twin) 在孪生线程内执行."""
        self._setters.put(lambda: fn(self.twin))
    def send_command(self, cmd: P.Command):
        if cmd.name == 'dwell_table':
            tbl = np.asarray(cmd.value, np.uint16).copy()
            self._setters.put(lambda: setattr(self.twin, 'dwell_table', tbl))
        elif cmd.name == 'DWELL_NSAMP':
            v = int(cmd.value); self._setters.put(lambda: setattr(self.twin.env, 'dwell_nsamp', v))
    def run(self):
        period = self.twin.env.frame_period_s
        nxt = time.perf_counter()
        while not self._stop.is_set():
            while not self._setters.empty():
                try:
                    self._setters.get_nowait()()
                except Exception as e:
                    self.status.emit(f'setter error: {e}')
            with self.lock:
                fr = self.twin.step(period)
            self.frame_ready.emit(fr)
            if self.realtime:
                nxt += period
                dt = nxt - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    nxt = time.perf_counter()

class UdpSource(DataSource):
    def __init__(self, listen=('0.0.0.0', P.DATA_PORT), device=(P.FPGA_IP, P.FPGA_PORT), parent=None):
        super().__init__(parent)
        self.listen = listen; self.device = device
        self.cmd_seq = 0
        self.n_bad = 0
        self.sock = None
    def send_command(self, cmd: P.Command):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(0.2)
        for pk in cmd.to_packets(self.cmd_seq):
            s.sendto(pk, self.device); self.cmd_seq += 1
            try:
                s.recvfrom(64)
            except socket.timeout:
                self.status.emit(f'cmd {cmd.name}: no ack')
        s.close()
    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.1)
        try:
            self.sock.bind(self.listen)
        except OSError as e:
            self.status.emit(f'bind failed: {e}'); return
        self.status.emit(f'listening {self.listen}')
        while not self._stop.is_set():
            try:
                b, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                fr = P.decode_frame(b)
            except P.ProtocolError as e:
                self.n_bad += 1
                if self.n_bad % 100 == 1:
                    self.status.emit(f'bad packet: {e}')
                continue
            self.frame_ready.emit(fr)
        self.sock.close()

class ReplaySource(DataSource):
    def __init__(self, path: str, period_s: float | None = None, loop: bool = True, parent=None):
        super().__init__(parent)
        self.frames = recorder.load(path); self.loop = loop
        self.period = period_s
    def run(self):
        if not self.frames:
            return
        k = 0
        nxt = time.perf_counter()
        while not self._stop.is_set():
            fr = self.frames[k]
            self.frame_ready.emit(fr)
            k += 1
            if k >= len(self.frames):
                if not self.loop:
                    break
                k = 0
            period = self.period or 63 * (fr.dwell_nsamp / 62.5e6 + 5e-6)
            nxt += period
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.perf_counter()

class PipelineWorker(QThread):
    """有界队列 (丢最旧) → pipeline.process → result_ready."""
    result_ready = pyqtSignal(object)
    def __init__(self, pipeline: Pipeline, maxsize: int = 8, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self.n_dropped = 0
        self.recorder = None
    def push(self, fr: Frame):
        if self.recorder is not None:
            self.recorder.add(fr)
        if self.q.full():
            try:
                self.q.get_nowait(); self.n_dropped += 1
            except queue.Empty:
                pass
        self.q.put_nowait(fr)
    def stop(self):
        self._stop.set(); self.wait(2000)
    def run(self):
        while not self._stop.is_set():
            try:
                fr = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                res = self.pipeline.process(fr, time.perf_counter())
            except Exception as e:      # 不让一帧异常杀掉工作线程
                import traceback; traceback.print_exc()
                continue
            self.result_ready.emit(res)
