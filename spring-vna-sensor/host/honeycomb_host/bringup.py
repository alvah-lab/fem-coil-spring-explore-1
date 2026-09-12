"""回板测试模式的非 GUI 核心: 任意驻留表帧 → 阻抗, 无环基线, 垫片标定记录, 噪声统计.

与 Pipeline 的分工: Pipeline 只处理默认 63 驻留表并跑跟踪器; 这里处理回板阶段的所有"非标准"用法
(单驻留、基线、单环垫片) 而且不依赖模型正确 — 模型只用来给出对照预测。"""
from __future__ import annotations
import csv
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
import numpy as np
from . import geometry as G
from .twin import (decode_dwells, word_fields, obs_of_word, default_dwell_table, FLAG_SAT, FLAG_REF, FLAG_ISENSE,
                   FLAG_LINK_TIMEOUT, FLAG_LINK_FAULT, Environment)
from .fastmodel import FastModel


def frame_to_Z(fr, env: Environment):
    """任意驻留表的帧 → 每驻留复阻抗 Z_k = V_k/PGA_k/(I_k/isense) (Ω) 与元数据.
    参考/电流标定/无线圈驻留的 Z 无意义 (meta['obs'] 为 None), V/I 原值保留在 meta 里。"""
    d = fr.dwells
    n_eff = fr.dwell_nsamp - env.link.blank_nsamp
    V, I = decode_dwells(d, n_eff)
    words = d['dwell_word'].astype(int)
    pga = np.array(env.link.pga_gains)[(words >> 4) & 3]
    Iamp = I / env.link.isense_V_per_A
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.where(np.abs(Iamp) > 0, V / pga / np.where(np.abs(Iamp) > 0, Iamp, 1), np.nan + 0j)
    obs = [obs_of_word(int(w)) for w in words]
    fl = d['flags'].astype(int)
    meta = dict(obs=obs, words=words, pga=pga, V=V, I=I, flags=fl,
                sat=(fl & FLAG_SAT) != 0, ref=(fl & FLAG_REF) != 0, isense=(fl & FLAG_ISENSE) != 0,
                link_timeout=(fl & FLAG_LINK_TIMEOUT) != 0, link_fault=(fl & FLAG_LINK_FAULT) != 0,
                n_eff=n_eff, seq=fr.seq, t_ticks=fr.t_ticks)
    return Z, meta


def obs61_from_frame(fr, env: Environment):
    """默认 63 表的帧 → (61,) 复阻抗 (按观测索引排好); 非默认表 → None."""
    Z, meta = frame_to_Z(fr, env)
    out = np.full(G.NOBS, np.nan + 0j)
    for k, n in enumerate(meta['obs']):
        if n is not None:
            out[n] = Z[k]
    return (out if not np.isnan(out).any() else None), meta


@dataclass
class Baseline:
    """无环基线: 61 观测的 L (nH) 与 R (Ω), 用来替换模型载波."""
    L61: np.ndarray
    R61: np.ndarray
    n_frames: int = 0
    f0: float = 0.0
    when: str = ''
    note: str = ''
    sigma_L61: np.ndarray | None = None

    @classmethod
    def from_frames(cls, frames, env: Environment, note: str = '') -> 'Baseline':
        w = 2 * np.pi * env.f0
        Zs = []
        for fr in frames:
            z, _ = obs61_from_frame(fr, env)
            if z is not None:
                Zs.append(z)
        if not Zs:
            raise ValueError('no default-table frames')
        Zs = np.array(Zs)
        Zm = Zs.mean(axis=0)
        L = np.imag(Zs) / w * 1e9
        return cls(L61=np.imag(Zm) / w * 1e9, R61=np.real(Zm), n_frames=len(Zs), f0=env.f0,
                   when=time.strftime('%Y-%m-%d %H:%M:%S'), note=note,
                   sigma_L61=L.std(axis=0) if len(Zs) > 1 else None)

    def to_dict(self) -> dict:
        d = dict(L61_nH=self.L61.tolist(), R61_ohm=self.R61.tolist(), n_frames=self.n_frames, f0=self.f0,
                 when=self.when, note=self.note, obs=[list(o) for o in G.OBS])
        if self.sigma_L61 is not None:
            d['sigma_L61_nH'] = self.sigma_L61.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Baseline':
        return cls(L61=np.array(d['L61_nH'], float), R61=np.array(d['R61_ohm'], float), n_frames=d.get('n_frames', 0),
                   f0=d.get('f0', 0.0), when=d.get('when', ''), note=d.get('note', ''),
                   sigma_L61=np.array(d['sigma_L61_nH']) if 'sigma_L61_nH' in d else None)

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, 'w'), ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> 'Baseline':
        return cls.from_dict(json.load(open(path)))


def model_shim(model: FastModel, unit: int, gap_mm: float, tilt_deg: float = 0.0, tilt_dir_deg: float = 0.0,
               dx_mm: float = 0.0, dy_mm: float = 0.0, others: bool = False, r_on: float = 0.0):
    """模型对"单元 unit 放一个环 (垫片) 其余无环"的预测: 返回 (dL61 nH 相对载波, 该单元自反射, 6 邻边反射 dict)."""
    poses = np.zeros((G.NU, 5))
    th = np.radians(tilt_deg); ph = np.radians(tilt_dir_deg)
    poses[unit] = [dx_mm, dy_mm, gap_mm - model.cfg.gap, th * np.cos(ph), th * np.sin(ph)]
    poses, _ = G.clamp_pose(poses, model.cfg.gap)
    present = np.full(G.NU, bool(others)); present[unit] = True
    Rr = np.where(present, model.ring.R, 1e12)
    Z = model.fold(*model.blocks(poses), Rr, None)
    from .twin import obs_of
    dL = (np.imag(obs_of(Z)) - np.imag(model.carrier_Z())) / model.w * 1e9
    edges = {}
    for n, (kind, i, j) in enumerate(G.OBS):
        if kind == 'edge' and (i == unit or j == unit):
            edges[j if i == unit else i] = dL[n]
    return dL, dL[unit], edges


@dataclass
class CalibRow:
    unit: int
    gap_mm: float
    tilt_deg: float
    tilt_dir_deg: float
    dx_mm: float
    dy_mm: float
    n_frames: int
    dL_self_nH: float
    sigma_self_nH: float
    dL_edges_nH: dict            # neighbour unit → ΔL
    model_self_nH: float
    model_edges_nH: dict
    note: str = ''
    when: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))


class CalibLog:
    """垫片标定记录: 每行一个 (单元, 间隙, 倾角...) 工况的实测 ΔL 与模型预测."""
    def __init__(self):
        self.rows: list[CalibRow] = []

    def add_from_frames(self, frames, env: Environment, baseline_L61: np.ndarray, model: FastModel,
                        unit: int, gap_mm: float, tilt_deg=0.0, tilt_dir_deg=0.0, dx_mm=0.0, dy_mm=0.0, note='') -> CalibRow:
        w = 2 * np.pi * env.f0
        Ls = []
        for fr in frames:
            z, _ = obs61_from_frame(fr, env)
            if z is not None:
                Ls.append(np.imag(z) / w * 1e9)
        if not Ls:
            raise ValueError('no default-table frames')
        Ls = np.array(Ls)
        dL = Ls.mean(axis=0) - baseline_L61
        sig = Ls.std(axis=0) if len(Ls) > 1 else np.zeros(G.NOBS)
        edges = {}
        for n, (kind, i, j) in enumerate(G.OBS):
            if kind == 'edge' and (i == unit or j == unit):
                edges[j if i == unit else i] = float(dL[n])
        _, ms, me = model_shim(model, unit, gap_mm, tilt_deg, tilt_dir_deg, dx_mm, dy_mm)
        row = CalibRow(unit, gap_mm, tilt_deg, tilt_dir_deg, dx_mm, dy_mm, len(Ls), float(dL[unit]), float(sig[unit]),
                       edges, float(ms), {k: float(v) for k, v in me.items()}, note)
        self.rows.append(row)
        return row

    def save_csv(self, path: str):
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['when', 'unit', 'gap_mm', 'tilt_deg', 'tilt_dir_deg', 'dx_mm', 'dy_mm', 'n_frames',
                         'dL_self_nH', 'sigma_self_nH', 'model_self_nH', 'edges(nb:meas/model)', 'note'])
            for r in self.rows:
                e = ' '.join(f'{k}:{v:.3f}/{r.model_edges_nH.get(k, float("nan")):.3f}' for k, v in sorted(r.dL_edges_nH.items()))
                wr.writerow([r.when, r.unit, r.gap_mm, r.tilt_deg, r.tilt_dir_deg, r.dx_mm, r.dy_mm, r.n_frames,
                             f'{r.dL_self_nH:.4f}', f'{r.sigma_self_nH:.4f}', f'{r.model_self_nH:.4f}', e, r.note])

    def save_json(self, path: str):
        json.dump([asdict(r) for r in self.rows], open(path, 'w'), ensure_ascii=False, indent=1)


class RunningStats:
    """滑动窗口内每驻留 |V| 的均值/标准差、帧率、饱和/链路计数."""
    def __init__(self, window: int = 200):
        self.window = window
        self.absV = deque(maxlen=window)
        self.L61 = deque(maxlen=window)
        self.t = deque(maxlen=window)
        self.n_sat = 0; self.n_timeout = 0; self.n_fault = 0; self.n_frames = 0
        self.last_seq = None; self.n_gap = 0

    def push(self, fr, env: Environment):
        Z, meta = frame_to_Z(fr, env)
        self.absV.append(np.abs(meta['V']))
        z61, _ = obs61_from_frame(fr, env)
        if z61 is not None:
            self.L61.append(np.imag(z61) / (2 * np.pi * env.f0) * 1e9)
        self.t.append(time.perf_counter())
        self.n_frames += 1
        self.n_sat += int(meta['sat'].sum()); self.n_timeout += int(meta['link_timeout'].sum()); self.n_fault += int(meta['link_fault'].sum())
        if self.last_seq is not None:
            gap = (fr.seq - self.last_seq - 1) & 0xFFFF
            if 0 < gap < 0x8000:
                self.n_gap += gap
        self.last_seq = fr.seq

    def sigma_V(self) -> np.ndarray | None:
        if len(self.absV) < 2:
            return None
        a = np.array(self.absV)
        if a.ndim != 2 or (a.shape[1] != len(self.absV[-1])):
            return None
        return a.std(axis=0)

    def sigma_L61(self) -> np.ndarray | None:
        return np.array(self.L61).std(axis=0) if len(self.L61) >= 2 else None

    def fps(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-9)

    def reset(self):
        self.__init__(self.window)


def single_dwell_table(drv: int, sns: int, pga: int, ref: int = 0, vna: int = 0) -> np.ndarray:
    from .twin import dwell_word
    return np.array([dwell_word(drv, sns, pga, ref, vna)], np.uint16)
