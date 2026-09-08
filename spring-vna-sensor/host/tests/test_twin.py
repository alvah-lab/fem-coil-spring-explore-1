import numpy as np
from honeycomb_host import geometry as G
from honeycomb_host.twin import Twin, Scenes, Environment, NoiseModel, decode_dwells, LSB, FLAG_SAT

def test_level0_vs_level1_sign():
    tw = Twin(env=Environment(noise=NoiseModel(preset='off')), scene=Scenes.rest())
    V = np.array([0.3 + 0.1j, 0.02 - 0.05j, -0.1 + 0.2j])
    s = tw.render_samples(V)
    I, Q = tw.nco_accumulate(s)
    Vrec = (I - 1j * Q) * 2 * LSB / s.shape[1]
    assert np.allclose(Vrec, V, atol=2 * LSB)            # 量化误差量级内, 符号约定钉死
    VI, VQ, _, _ = tw.accumulate(V, V, s.shape[1])
    assert np.allclose(VI, I, atol=1.5 * s.shape[1]) and np.allclose(VQ, Q, atol=1.5 * s.shape[1])

def test_frame_decode_roundtrip():
    env = Environment(noise=NoiseModel(preset='off'))
    tw = Twin(env=env, scene=Scenes.point_press(ramp_s=0))
    fr = tw.step()
    assert fr.dwells.shape == (63,) and fr.dwell_nsamp == 12500
    V, I = decode_dwells(fr.dwells, env.dwell_nsamp - env.link.blank_nsamp)
    gp = np.array(env.link.pga_gains)[(fr.dwells['dwell_word'] >> 4) & 3]
    Z = V[:61] / gp[:61] / (I[:61] / env.link.isense_V_per_A)
    L = np.imag(Z) / tw.w * 1e9
    Lt = np.imag(fr.truth.z61_coff) / tw.w * 1e9
    assert np.abs(L - Lt).max() < 0.02                 # 12-bit 量化 (noise off) 以内
    assert not (fr.dwells['flags'] & FLAG_SAT).any()

def test_hardware_noise_sigma():
    env = Environment(noise=NoiseModel(preset='hardware', seed=3))
    tw = Twin(env=env, scene=Scenes.rest())
    Ls = []
    for _ in range(40):
        fr = tw.step()
        V, I = decode_dwells(fr.dwells, env.dwell_nsamp - env.link.blank_nsamp)
        gp = np.array(env.link.pga_gains)[(fr.dwells['dwell_word'] >> 4) & 3]
        Ls.append(np.imag(V[:61] / gp[:61] / (I[:61] / env.link.isense_V_per_A)) / tw.w * 1e9)
    sd = np.std(Ls, axis=0)
    assert sd[:19].max() < 0.3 and sd[19:].max() < 0.02   # 自观测 <0.3nH, 边观测 <20pH

def test_coff_residual_scaling():
    env = Environment(noise=NoiseModel(preset='off'))
    tw = Twin(env=env, scene=Scenes.rest())
    z0, zc0, _, _ = tw.z61(np.zeros(57))
    bias0 = zc0 - z0
    tw.env.c_off_F[:] = 250e-15
    z1, zc1, _, _ = tw.z61(np.zeros(57))
    bias1 = zc1 - z1
    r = np.abs(bias1[19:]).mean() / np.abs(bias0[19:]).mean()
    assert 1.6 < r < 2.4                                   # 干扰 ∝ C_off
    assert np.abs(np.imag(bias0[19:])).max() / tw.w * 1e9 < 0.05   # 125fF 时边偏置 <50pH

def test_temperature_channel():
    env = Environment(noise=NoiseModel(preset='off'))
    tw = Twin(env=env, scene=Scenes.rest())
    z0, _, _, _ = tw.z61(np.zeros(57))
    z1, _, _, _ = tw.z61(np.zeros(57), dT_ring=np.full(G.NU, 10.0))
    dIm = np.abs(np.imag(z1 - z0)[:19]) / tw.w * 1e9
    dRe = np.real(z1 - z0)[:19]
    assert dIm.max() < 0.05                                # Im 免疫: <50pH (≈0.1µm 等效以内)
    assert dRe.min() > 0                                   # Re 随环温单调增
