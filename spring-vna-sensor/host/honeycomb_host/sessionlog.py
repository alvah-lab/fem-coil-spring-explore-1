"""会话日志: 记录操作事件 + 每帧主界面背后的原始数据 (跟踪器/掩码/观测), 供离线复现与分析.

目录 logs/session_<时间>/
  meta.json      启动参数, 环境, git 版本
  events.jsonl   一行一个操作事件 {t, seq, name, ...}
  frames.jsonl   一行一帧 {t, seq, present, no_rings, q, chi2, dq, reset, relin, clamped, refl, sat, gain, fps, dropped}
  raw_frames.npz 原始帧 (dwell I/Q), 关闭日志时写出; 可用 ReplaySource 回放
用法: GUI 工具栏"日志"按钮 (或 --log 启动参数); 分析: scripts/analyze_session.py <目录>
"""
from __future__ import annotations
import json
import os
import subprocess
import time
import numpy as np
from . import geometry as G
from .recorder import Recorder


def _git_rev(path: str) -> str:
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], text=True, timeout=5).strip()
    except Exception:
        return '?'


class SessionLogger:
    def __init__(self, root: str | None = None, meta: dict | None = None):
        root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        self.dir = os.path.join(root, time.strftime('session_%Y%m%d_%H%M%S'))
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.perf_counter()
        self._ev = open(os.path.join(self.dir, 'events.jsonl'), 'w')
        self._fr = open(os.path.join(self.dir, 'frames.jsonl'), 'w')
        self.recorder = Recorder()
        self.n_frames = 0
        self.last_seq = None
        m = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), git=_git_rev(os.path.dirname(os.path.abspath(__file__))), **(meta or {}))
        json.dump(m, open(os.path.join(self.dir, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
        self.event('log_start', **m)

    def event(self, name: str, **kw):
        rec = dict(t=round(time.perf_counter() - self.t0, 4), seq=self.last_seq, name=name)
        rec.update({k: _jsonable(v) for k, v in kw.items()})
        self._ev.write(json.dumps(rec, ensure_ascii=False) + '\n'); self._ev.flush()

    def frame(self, res, fr=None, extra: dict | None = None):
        """res: pipeline.FrameResult (主界面每帧看到的一切); fr: 原始 Frame (录入 raw_frames)."""
        self.last_seq = res.seq
        tr = res.track
        rec = dict(t=round(time.perf_counter() - self.t0, 4), seq=int(res.seq), t_s=round(res.t_s, 6),
                   present=(res.present.astype(int).tolist() if res.present is not None else None),
                   no_rings=bool(getattr(res, 'no_rings', False)),
                   refl=np.round(res.refl_nH, 4).tolist(), R=np.round(res.R_ohm, 4).tolist(),
                   sat=int(res.sat.sum()), gain=[round(float(np.abs(res.link_gain)), 6), round(float(np.degrees(np.angle(res.link_gain))), 4)],
                   fps=round(res.fps, 1), seq_gap=int(res.seq_gap))
        if tr is not None:
            rec.update(q=np.round(tr.q * 1e3, 1).tolist(), chi2=round(float(tr.chi2), 3), dq=round(float(tr.dq_norm), 4),
                       reset=bool(tr.reset), relin=bool(tr.relin), clamped=bool(tr.clamped), ms=round(float(tr.ms), 2),
                       weak=np.round(tr.weak_amp, 5).tolist(), resid_max=round(float(np.abs(tr.resid).max()), 3))
        if res.truth is not None:
            rec['truth_q'] = np.round(np.asarray(res.truth.q) * 1e3, 1).tolist()
        if extra:
            rec.update({k: _jsonable(v) for k, v in extra.items()})
        self._fr.write(json.dumps(rec) + '\n')
        self.n_frames += 1
        if self.n_frames % 50 == 0:
            self._fr.flush()
        if fr is not None:
            self.recorder.add(fr)

    def close(self):
        self.event('log_stop', n_frames=self.n_frames)
        self._ev.close(); self._fr.close()
        if self.recorder.frames:
            self.recorder.save(os.path.join(self.dir, 'raw_frames.npz'))
        return self.dir


_ACTIVE: 'SessionLogger | None' = None


def set_active(logger):
    global _ACTIVE
    _ACTIVE = logger


def emit(name: str, **kw):
    """任何面板都可调用: 有活动日志则记事件, 否则无操作."""
    if _ACTIVE is not None:
        _ACTIVE.event(name, **kw)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v
