"""回板测试模式的孪生支持: 驻留表驱动、无环/垫片场景、单驻留、基线替换."""
import numpy as np
from honeycomb_host import geometry as G
from honeycomb_host import protocol as P
from honeycomb_host.twin import (Twin, Environment, NoiseModel, Scenes, default_dwell_table, dwell_word,
                                 decode_dwells, word_fields, obs_of_word, FLAG_REF, FLAG_ISENSE, OBS_REF, OBS_ISENSE)
from honeycomb_host.pipeline import Pipeline
from honeycomb_host.bringup import frame_to_Z, obs61_from_frame, Baseline


def _twin(scene=None):
    env = Environment(noise=NoiseModel(preset='off'))
    return Twin(env=env, scene=scene or Scenes.rest())


def test_default_table_unchanged():
    tw = _twin()
    fr = tw.step()
    d = fr.dwells
    assert len(d) == 63 and d['flags'][OBS_REF] & FLAG_REF and d['flags'][OBS_ISENSE] & FLAG_ISENSE
    assert not (d['flags'][:61] & (FLAG_REF | FLAG_ISENSE)).any()
    n_eff = tw.env.dwell_nsamp - tw.env.link.blank_nsamp
    V, I = decode_dwells(d, n_eff)
    z, zc, _, _ = tw.z61(np.zeros(3 * G.NU))
    zs = zc if tw.env.coff_enable else z
    pga = np.array(tw.env.link.pga_gains)[(d['dwell_word'] >> 4) & 3]
    Zm = V[:61] / pga[:61] / (I[:61] / tw.env.link.isense_V_per_A)
    assert np.allclose(Zm, zs, rtol=2e-4, atol=1e-4)


def test_no_rings_gives_carrier():
    tw = _twin(Scenes.no_rings())
    fr = tw.step()
    Z, meta = obs61_from_frame(fr, tw.env)
    car = tw.model.carrier_Z()
    # 虚部 = 模型载波 (C_off 端接的影响 ~1e-4 Ω 可忽略); 实部多出开关 Ron (自观测) → 只比虚部
    assert np.allclose(np.imag(Z), np.imag(car), rtol=2e-4, atol=1e-3)
    assert np.allclose(np.real(Z[:19]), tw.model.coil.R + tw.env.r_on_ohm, atol=0.05)
    # rest with rings must differ (reflection present)
    Zr, _ = obs61_from_frame(_twin().step(), tw.env)
    assert np.abs(Zr[:19] - car[:19]).max() > 1.0


def test_shim_only_affects_unit_and_edges():
    u = 9
    tw = _twin(Scenes.shim(unit=u, gap_mm=1.75, others=False))
    Z, _ = obs61_from_frame(tw.step(), tw.env)
    car = tw.model.carrier_Z()
    dL = np.imag(Z - car) / tw.w * 1e9
    touched = {u} | set(G.ADJ[u])
    for n, (kind, i, j) in enumerate(G.OBS):
        if kind == 'self':
            assert (abs(dL[n]) > 1.0) == (i == u), (n, dL[n])
        else:
            expect = (i == u or j == u)
            assert (abs(dL[n]) > 0.05) == expect, (n, i, j, dL[n])
    # gap sensitivity: closer → larger |reflection|
    tw2 = _twin(Scenes.shim(unit=u, gap_mm=1.35))
    Z2, _ = obs61_from_frame(tw2.step(), tw.env)
    assert abs(np.imag(Z2[u] - car[u])) > abs(np.imag(Z[u] - car[u]))
    # tilt changes edge asymmetry but not the sign of the self term
    tw3 = _twin(Scenes.shim(unit=u, gap_mm=1.75, tilt_deg=5, tilt_dir_deg=0))
    Z3, _ = obs61_from_frame(tw3.step(), tw.env)
    edges = [n for n, (k, i, j) in enumerate(G.OBS) if k == 'edge' and (i == u or j == u)]
    assert np.std(np.imag(Z3[edges])) > np.std(np.imag(Z[edges]))


def test_single_dwell_table():
    tw = _twin()
    w = dwell_word(4, 4, 0)
    tw.dwell_table = np.array([w], np.uint16)
    fr = tw.step()
    assert len(fr.dwells) == 1 and fr.dwells['dwell_word'][0] == w
    assert obs_of_word(w) == 4 and word_fields(w)['pga'] == 0
    Z, meta = frame_to_Z(fr, tw.env)
    z, zc, _, _ = tw.z61(np.zeros(3 * G.NU))
    assert abs(Z[0] - (zc if tw.env.coff_enable else z)[4]) < 1e-3 * abs(z[4])
    assert meta['obs'][0] == 4


def test_pipeline_rejects_short_table_and_baseline():
    tw = _twin()
    pipe = Pipeline(tw.env, model=tw.model)
    tw.dwell_table = np.array([dwell_word(0, 0, 0)], np.uint16)
    try:
        pipe.process(tw.step()); raise AssertionError('should reject')
    except ValueError:
        pass
    tw.dwell_table = default_dwell_table()
    fr = tw.step()
    res0 = pipe.process(fr)
    bl = Baseline.from_frames([_twin(Scenes.no_rings()).step() for _ in range(3)], tw.env)
    assert bl.L61.shape == (61,) and np.allclose(bl.L61, tw.model.carrier_L(), rtol=2e-4, atol=1e-3)
    pipe.set_baseline(bl.L61)
    res1 = pipe.process(fr)
    assert np.allclose(res1.refl_nH, res0.refl_nH, atol=5e-3)   # 基线来自量化后的帧, 与模型载波差 ~1e-3 nH
    d = bl.to_dict(); bl2 = Baseline.from_dict(d)
    assert np.allclose(bl2.L61, bl.L61)
