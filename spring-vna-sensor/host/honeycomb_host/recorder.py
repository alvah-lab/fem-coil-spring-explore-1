"""录制 / 回放 (npz): frames 结构化数组 + 头字段 + 可选真值."""
from __future__ import annotations
import numpy as np
from .twin import Frame, DWELL_DTYPE, N_DWELL

class Recorder:
    def __init__(self):
        self.frames: list[Frame] = []
    def add(self, fr: Frame):
        self.frames.append(fr)
    def clear(self):
        self.frames.clear()
    def save(self, path: str):
        n = len(self.frames)
        if n == 0:
            return
        dw = np.stack([f.dwells for f in self.frames])
        hdr = np.array([(f.seq, f.t_ticks, f.dwell_nsamp, f.nco_word, f.frame_id, f.flags) for f in self.frames],
                       dtype=[('seq', 'i8'), ('t_ticks', 'i8'), ('dwell_nsamp', 'i4'), ('nco_word', 'u4'), ('frame_id', 'i8'), ('flags', 'u1')])
        extra = {}
        if self.frames[0].truth is not None:
            extra['truth_q'] = np.stack([f.truth.q for f in self.frames])
            extra['truth_L'] = np.stack([f.truth.L61_nH for f in self.frames])
        np.savez_compressed(path, dwells=dw, hdr=hdr, **extra)

def load(path: str) -> list[Frame]:
    d = np.load(path, allow_pickle=False)
    dw, hdr = d['dwells'], d['hdr']
    out = []
    for k in range(len(hdr)):
        h = hdr[k]
        out.append(Frame(int(h['seq']), int(h['t_ticks']), int(h['dwell_nsamp']), int(h['nco_word']),
                         dw[k].astype(DWELL_DTYPE), int(h['frame_id']), int(h['flags']), None))
    return out
