"""闭环: 孪生真值 q → 帧 → pipeline → q̂. 阈值与 honeycomb_scheme_c.json e2e_v2 (w 1σ 1.83µm, uv 3.10µm) 一致."""
import numpy as np, pytest
from honeycomb_host import geometry as G
from honeycomb_host.twin import Twin, Scenes, Environment, NoiseModel
from honeycomb_host.pipeline import Pipeline
from honeycomb_host.invert import TrackerConfig

def run(scene, seed, nframes=30, preset='sig2_matched'):
    env = Environment(noise=NoiseModel(preset=preset, seed=seed))
    tw = Twin(env=env, scene=scene)
    pl = Pipeline(env, model=tw.model, tracker_cfg=TrackerConfig(iters_per_frame=2))
    pl.tracker.async_relin = False
    res = None
    for _ in range(nframes):
        res = pl.process(tw.step())
    q_true = res.truth.q; tr = res.track
    weak = tr and pl.tracker.lin['weak']
    qt_s = q_true / G.RANGE_Q
    qt_obs = (qt_s - weak.T @ (weak @ qt_s)) * G.RANGE_Q       # 可观测子空间真值
    qh_s = tr.q / G.RANGE_Q
    qh_obs = (qh_s - weak.T @ (weak @ qh_s)) * G.RANGE_Q
    return qh_obs - qt_obs, tr, q_true

@pytest.mark.parametrize('scene', ['point_press', 'tilt', 'impact'])
def test_strong_subspace_noise(scene):
    sc = {'point_press': Scenes.point_press(ramp_s=0), 'tilt': Scenes.tilt(), 'impact': Scenes.impact(t0=0.0, tau=1e9)}[scene]
    errs = np.array([run(sc, s)[0] for s in range(6)])
    w_sd = np.std(errs[:, :G.NU]); uv_sd = np.std(errs[:, G.NU:])
    w_bias = np.abs(errs[:, :G.NU].mean(axis=0)).max()
    assert w_sd < 2.0e-3, w_sd                 # w 1σ < 2.0 µm
    assert uv_sd < 4.0e-3, uv_sd               # 面内 1σ < 4.0 µm
    assert w_bias < 15e-3, w_bias              # 满深系统误差 < 15 µm

def test_converges_within_15_frames():
    env = Environment(noise=NoiseModel(preset='sig2_matched', seed=0))
    tw = Twin(env=env, scene=Scenes.point_press(ramp_s=0)); pl = Pipeline(env, model=tw.model)
    pl.tracker.async_relin = False
    for k in range(15):
        res = pl.process(tw.step())
    e = res.track.q[:G.NU] - res.truth.q[:G.NU]
    assert np.abs(e).max() < 15e-3

def test_hardware_preset_reports():
    err, tr, qt = run(Scenes.point_press(ramp_s=0), 1, preset='hardware')
    print(f'\nhardware preset: w max err {np.abs(err[:G.NU]).max()*1e3:.2f}µm, uv max {np.abs(err[G.NU:]).max()*1e3:.2f}µm, chi2 {tr.chi2:.2f}')
    assert np.isfinite(err).all()
