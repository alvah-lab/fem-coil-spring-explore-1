import os, time, numpy as np, pytest
from honeycomb_host import geometry as G
from honeycomb_host.fastmodel import FastModel, ModelConfig, obs_of

def test_nominal_scale(model):
    refl = model.observe_L(np.zeros((G.NU, 5))) - model.carrier_L()
    nom, sunk = refl[:G.NU][~G.SUNK], refl[:G.NU][G.SUNK]
    assert -66 < nom.mean() < -52 and np.ptp(nom) < 0.5      # Neumann 口径 62nH (L2 在 -0.1 → 略小)
    assert -25 < sunk.mean() < -18 and np.ptp(sunk) < 0.5
    assert 0.8 < refl[G.NU:].min() and refl[G.NU:].max() < 4.0   # 边反射 1.2~3.7 nH 量级
    car = model.carrier_L()
    assert 1050 < car[:G.NU].mean() < 1170                        # 载波 ≈ L_self 1.11µH

@pytest.mark.parametrize('seed', [1, 2, 3])
def test_tables_vs_direct_neumann(model, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1, 1, (G.NU, 5)) * np.array([0.3, 0.3, 0.4, 0.15, 0.15])
    Mcr_d, Mrr_d = model.neumann_direct(p)
    Mcr_t, Mrr_t = model.blocks(p)
    row_ref = np.abs(Mcr_d).max(axis=1, keepdims=True)
    assert np.max(np.abs(Mcr_t - Mcr_d) / row_ref) < 3e-3          # 相对本行最强 (近零项不按相对算)
    m = model.near & ~np.eye(G.NU, dtype=bool)
    nn = np.abs(Mrr_d[m]).max()
    assert np.max(np.abs(Mrr_t - Mrr_d)[m]) / nn < 1e-2        # 72 段多边形 vs 表 (表更准, 见 fastmodel 文档)
    y_t = np.imag(obs_of(model.fold(Mcr_t, Mrr_t))); y_d = np.imag(obs_of(model.fold(Mcr_d, Mrr_d)))
    assert np.max(np.abs(y_t - y_d) / np.abs(y_d)) < 2e-3

def test_symmetry(model):
    Mcr, Mrr = model.blocks(np.zeros((G.NU, 5)))
    assert np.allclose(Mrr, Mrr.T, atol=1e-15)
    nn = [Mrr[G.CI, j] for j in G.ADJ[G.CI]]
    assert np.ptp(nn) / abs(np.mean(nn)) < 1e-3        # 六重对称

def test_row_update_equals_full(model):
    rng = np.random.default_rng(5)
    p = rng.uniform(-1, 1, (G.NU, 5)) * np.array([0.2, 0.2, 0.3, 0.1, 0.1])
    Mcr, Mrr = model.blocks(p)
    p2 = p.copy(); p2[7, 2] += 0.05; p2[7, 0] -= 0.03
    Mcr2, Mrr2 = model.blocks(p2); Mcr3, Mrr3 = model.update_ring(Mcr, Mrr, 7, p2)
    assert np.allclose(Mcr2, Mcr3, rtol=0, atol=1e-14) and np.allclose(Mrr2, Mrr3, rtol=0, atol=5e-14)

def test_jacobian_structure(model):
    J = model.jacobian()
    Jc = (J / np.maximum(1e-3 * np.abs(model.observe_L(np.zeros((G.NU, 5))) - model.carrier_L()), 0.005)[:, None]) @ G.T_S
    S = np.linalg.svd(Jc, compute_uv=False)
    ss = np.sort(S)
    assert np.sum(S > 1e-8) == 57                                  # 柔性先验下满秩
    assert ss[3] / ss[2] > 5                                        # 3 个弱模式与强子空间分离

def test_clamp_never_raises(model):
    y = model.observe_L(np.full((G.NU, 5), 100.0))
    assert np.all(np.isfinite(y))

@pytest.mark.skipif(os.environ.get('CI') == '1', reason='timing')
def test_runtime(model):
    p = np.zeros((G.NU, 5)); model.blocks(p)
    t = time.perf_counter(); [model.blocks(p) for _ in range(10)]; dt = (time.perf_counter() - t) / 10
    assert dt < 0.015
    t = time.perf_counter(); model.jacobian(); assert time.perf_counter() - t < 0.5
