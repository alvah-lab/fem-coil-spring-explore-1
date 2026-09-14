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


def test_no_rings_pauses_tracker():
    """无环帧: 跟踪器暂停 (track=None, 只复位一次), 有环后恢复."""
    tw = _twin()
    pipe = Pipeline(tw.env, model=tw.model)
    for _ in range(2):
        pipe.process(tw.step())
    tw.scene = Scenes.no_rings(); tw.t = 0
    res = [pipe.process(tw.step()) for _ in range(7)]
    assert all(r.no_rings and r.track is None for r in res[3:])      # 前 3 帧为切换去抖
    assert pipe.tracker.n_reset <= 1 and np.all(pipe.tracker.q == 0)
    tw.scene = Scenes.rest(); tw.t = 0
    for _ in range(4):
        r = pipe.process(tw.step())
    assert not r.no_rings and r.track is not None


def test_partial_rings_tracker():
    """单环垫片 (其余无环): 跟踪器自动只拟合在位单元, 其余冻结为 0, 不再发散/复位."""
    tw = _twin()
    pipe = Pipeline(tw.env, model=tw.model)
    for _ in range(2):
        pipe.process(tw.step())
    gap = 1.75; w_true = gap - tw.env.gap                                                 # 相对静息 2.53 → -0.78 mm
    tw.scene = Scenes.shim(unit=9, gap_mm=gap, model_gap=tw.env.gap); tw.t = 0
    res = None
    for n in range(25):
        res = pipe.process(tw.step())
    assert res.present is not None and res.present[9] and res.present.sum() == 1
    assert pipe.tracker.n_reset == 0 and res.track is not None
    q = res.track.q
    others = np.ones(G.NU, bool); others[9] = False
    assert np.all(q[:G.NU][others] == 0) and np.all(q[G.NU:].reshape(2, G.NU)[:, others] == 0)
    assert abs(q[9] - w_true) < 0.03, (q[9], w_true)


def test_plates_layout_and_prediction():
    """整板布局表: 转位映射自洽, 孪生场景与模型整板预测一致, 全阵列板在跟踪器下可运行."""
    from honeycomb_host.twin import load_plates, plate_cells_on_board
    from honeycomb_host.bringup import plate_expected, plate_record_from_frames, Baseline
    P = load_plates()
    assert set(c for c in P if not c.startswith('_')) == {'A0', 'A1C', 'A2C', 'B1', 'B2', 'B3', 'D', 'E1'}
    # B2 全阵列: 任何取向都一样; A1C: 类 0 在任何取向都落在类 0
    for k in range(6):
        assert sorted(plate_cells_on_board(P, 'B2', k)) == list(range(G.NU))
        assert all(G.SUNK[v] for v in plate_cells_on_board(P, 'A1C', k))
        c2 = plate_cells_on_board(P, 'A2C', k)
        assert not any(G.SUNK[v] for v in c2) and len(c2) == 6
    # 孪生场景 (板 D, k=2) 经帧 → 与模型整板预测一致
    tw = _twin(Scenes.plate('D', 2, P, model_gap=Environment().gap))
    bl = Baseline.from_frames([_twin(Scenes.no_rings()).step() for _ in range(2)], tw.env)
    rec = plate_record_from_frames([tw.step() for _ in range(3)], tw.env, bl.L61, tw.model, P, 'D', 2)
    sm = rec.summary()
    assert sm['self_max'] < 0.05 and sm['edge_max'] < 0.02 and sm['absent_self_max'] < 0.01, sm
    assert len(rec.units) == 8
    # 偏移板 A2C: 偏移向量随取向旋转 (k=1 时 60°)
    c0 = plate_cells_on_board(P, 'A2C', 0); c1 = plate_cells_on_board(P, 'A2C', 1)
    off0 = [s for s in c0.values() if s['dx_mm'] or s['dy_mm']][0]
    assert any(abs(np.hypot(s['dx_mm'], s['dy_mm']) - 0.5) < 1e-6 for s in c1.values())


def test_rigid_mode_on_stepped_plate():
    """D 板 (相邻阶差对): 部分放环 → 刚性模式, 跟踪器无复位收敛到 ≤ 20 µm."""
    from honeycomb_host.twin import load_plates
    P = load_plates()
    tw = _twin()
    pipe = Pipeline(tw.env, model=tw.model)
    for _ in range(3):
        pipe.process(tw.step())
    tw.scene = Scenes.plate('D', 3, P, model_gap=tw.env.gap); tw.t = 0
    truth = tw.scene.poses_of_t(0)[:, 2]
    for n in range(120):
        res = pipe.process(tw.step())
        if n == 40:
            r40 = pipe.tracker.n_reset          # 切换去抖期间的复位可接受, 之后不得再复位
    assert pipe.tracker.rigid and res.present.sum() == 8 and pipe.tracker.n_reset == r40
    pres = np.where(res.present)[0]
    assert np.abs(res.track.q[:G.NU][pres] - truth[pres]).max() < 0.02


def test_session_logger_roundtrip(tmp_path):
    from honeycomb_host.sessionlog import SessionLogger, emit, set_active
    import json, os
    tw = _twin(); pipe = Pipeline(tw.env, model=tw.model)
    lg = SessionLogger(root=str(tmp_path), meta=dict(source='twin'))
    set_active(lg); emit('unit_test', a=1, arr=np.arange(3))
    for _ in range(3):
        fr = tw.step(); lg.frame(pipe.process(fr), fr=fr)
    set_active(None); d = lg.close()
    ev = [json.loads(l) for l in open(os.path.join(d, 'events.jsonl'))]
    fr_ = [json.loads(l) for l in open(os.path.join(d, 'frames.jsonl'))]
    assert [e['name'] for e in ev] == ['log_start', 'unit_test', 'log_stop'] and ev[1]['arr'] == [0, 1, 2]
    assert len(fr_) == 3 and len(fr_[0]['q']) == 57 and len(fr_[0]['refl']) == 61 and 'truth_q' in fr_[0]
    assert os.path.exists(os.path.join(d, 'raw_frames.npz'))
