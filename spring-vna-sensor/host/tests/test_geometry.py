import json, os, numpy as np
from honeycomb_host import geometry as G

def test_lattice_counts():
    assert G.NU == 19 and G.NE == 42 and G.CI == 9 and G.NOBS == 61
    assert list(np.where(G.SUNK)[0]) == [1, 3, 6, 9, 12, 15, 17]
    assert G.OBS[0] == ('self', 0, 0) and G.OBS[19] == ('edge', 0, 1)
    assert G.EDGES == sorted(G.EDGES)

def test_layout_json_matches():
    lay = json.load(open(os.path.join(G.DATA_DIR, 'layout.json')))
    for r, s in zip(G.layout_records(), lay):
        assert r['id'] == s['id'] and r['sunk'] == s['sunk']
        assert abs(r['x_mm'] - s['x_mm']) < 1e-3 and abs(r['y_mm'] - s['y_mm']) < 1e-3

def test_prior_plane_gradient():
    a, b = 0.3, -0.7
    wp = a * G.XY[:, 0] + b * G.XY[:, 1]
    assert np.allclose(G.GX @ wp, a, atol=1e-9) and np.allclose(G.GY @ wp, b, atol=1e-9)
    q = np.concatenate([wp, np.zeros(2 * G.NU)])
    p = G.pose_of(q)
    assert np.allclose(p[:, 2], wp) and np.allclose(p[:, 3], b) and np.allclose(p[:, 4], -a)

def test_series_scheme_fixture():
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'series_scheme_T.npz')
    if not os.path.exists(path):
        import pytest; pytest.skip('fixture 未生成 (scripts/make_fixtures.py)')
    d = np.load(path)
    assert np.allclose(d['T_phys'], G.T_PHYS) and np.allclose(d['T_s'], G.T_S) and np.allclose(d['RANGE_q'], G.RANGE_Q)

def test_clamp_never_raises():
    p = np.full((G.NU, 5), 1e6)
    q, m = G.clamp_pose(p, G.GAP_NOM)
    assert m.all() and np.all(np.abs(q[:, :2]) <= 1.0) and np.all(np.abs(q[:, 3:]) <= 0.35)
    q2, m2 = G.clamp_pose(np.zeros((G.NU, 5)), G.GAP_NOM)
    assert not m2.any()
