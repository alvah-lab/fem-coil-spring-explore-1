import numpy as np
from honeycomb_host import geometry as G
from honeycomb_host.invert import Tracker, TrackerConfig

def test_weak_modes_and_convergence(model):
    tr = Tracker(model, TrackerConfig(iters_per_frame=1))
    S = tr.lin['S']; ss = np.sort(S)
    assert ss[3] / ss[2] > 5 and ss[2] / ss[0] < 20        # 3 弱模式 (共模 u/v + 旋转)
    tr.async_relin = False
    w_true = -0.3 * np.exp(-(G.XY[:, 0] ** 2 + G.XY[:, 1] ** 2) / (2 * 6.2 ** 2))
    q_true = np.concatenate([w_true, np.zeros(2 * G.NU)])
    y = model.observe_L(G.pose_of(q_true))
    errs = []
    for k in range(12):
        out = tr.update(y)
        errs.append(np.abs(out.q[:G.NU] - w_true).max())
    assert errs[-1] < 3e-3 and errs[-1] < errs[0]          # <3µm, 单调收敛到位

def test_clamp_and_reset(model):
    tr = Tracker(model); tr.async_relin = False
    y = model.observe_L(np.zeros((G.NU, 5))) + 1e4          # 荒谬观测
    for _ in range(8):
        out = tr.update(y)
    assert np.all(np.isfinite(out.q)) and np.all(np.abs(out.q) <= tr.box + 1e-12)

def test_async_relin_double_buffer(model):
    tr = Tracker(model)
    q1 = np.zeros(57); q1[G.CI] = -0.2
    assert tr.request_relinearize(q1)
    tr._relin_thread.join(5)
    y = model.observe_L(np.zeros((G.NU, 5)))
    out = tr.update(y)
    assert out.relin and np.allclose(tr.lin['q_lin'], q1)
