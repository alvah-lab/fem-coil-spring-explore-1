"""固件契约 (v0.1): 定点 NCO 累加参考、寄存器表、flags 位、驻留表自增."""
import numpy as np
from honeycomb_host import protocol as P
from honeycomb_host.twin import Twin, Environment, NoiseModel, LSB, FLAG_LINK_TIMEOUT, FLAG_LINK_FAULT, FLAG_SAT, FLAG_REF, FLAG_ISENSE


def test_fixed_point_accumulate_matches_float():
    env = Environment(noise=NoiseModel(preset='off'))
    tw = Twin(env=env)
    V = np.array([0.3 + 0.1j, -0.02 + 0.2j, 1e-3 - 4e-3j])
    s = tw.render_samples(V)
    I0, Q0 = tw.nco_accumulate(s, blank=312)
    I1, Q1 = tw.nco_accumulate(s, blank=312, coef_q=14)
    # Q14 coefficients: relative error ~1e-5, absolute a few LSB-sums
    assert np.all(np.abs(I1 - I0) <= 2 + 2e-5 * np.abs(I0))
    assert np.all(np.abs(Q1 - Q0) <= 2 + 2e-5 * np.abs(Q0))
    assert I1.dtype.kind == 'i'


def test_fixed_point_saturates_int32():
    env = Environment(noise=NoiseModel(preset='off'))
    tw = Twin(env=env)
    s = np.full((1, env.dwell_nsamp), 2047, dtype=np.int16)   # DC full scale, huge sum only if coef aligned
    I, Q = tw.nco_accumulate(s, coef_q=14)
    assert -2**31 <= I[0] <= 2**31 - 1


def test_v01_registers_and_flags():
    for name, addr in dict(RF_EN=0x60, BLANK_NSAMP=0x64, LINK_STATUS=0x68, ERR_CNT=0x6C, FW_ID=0x70).items():
        assert P.REG[name] == addr
    assert FLAG_LINK_TIMEOUT == 0x08 and FLAG_LINK_FAULT == 0x10
    assert len({FLAG_SAT, FLAG_REF, FLAG_ISENSE, FLAG_LINK_TIMEOUT, FLAG_LINK_FAULT}) == 5
    pk = P.Command('RF_EN', 1).to_packets(7)
    assert len(pk) == 1 and P.unpack_cmd(pk[0])[3] == 0x60 and P.unpack_cmd(pk[0])[6] == 1


def test_mod_table_command_and_phasors():
    """固件 v0.2 驻留调制表: Command('mod_table') 的包序列可还原表; dwell_phasors 与 step() 的累加值一致."""
    from honeycomb_host.twin import Twin, Environment, NoiseModel, Scenes, mod_table_from_phasors, LSB
    tw = Twin(env=Environment(noise=NoiseModel(preset='off')), scene=Scenes.plate('B2', 0))
    V, Ich, flags, _ = tw.dwell_phasors()
    t = mod_table_from_phasors(V, Ich)
    assert t.shape == (63, 4) and t.dtype == np.int16
    pk = P.Command('mod_table', t).to_packets(0)
    assert len(pk) == 1 + 2 * 63
    op, _, seq, reg, ch, ln, data = P.unpack_cmd(pk[0]); assert reg == P.REG['MOD_ADDR'] and data == 0
    back = np.zeros_like(t)
    for i in range(63):
        _, _, _, reg_v, _, _, dv = P.unpack_cmd(pk[1 + 2 * i]); _, _, _, reg_i, _, _, di = P.unpack_cmd(pk[2 + 2 * i])
        assert reg_v == P.REG['MOD_V'] and reg_i == P.REG['MOD_I']
        back[i] = np.array([dv & 0xFFFF, dv >> 16, di & 0xFFFF, di >> 16], np.uint16).astype(np.int16)
    assert np.array_equal(back, t)
    # 表编码的相量 == 输入相量 (±0.5 LSB), 且 step() 的记录与之一致
    Vt = (t[:, 0] - 1j * t[:, 1]) * LSB / 16
    assert np.abs(Vt - V).max() <= 0.71 * LSB / 16
    fr = tw.step(); n_eff = tw.env.dwell_nsamp - tw.env.link.blank_nsamp
    Vs = (fr.dwells['V_I'] - 1j * fr.dwells['V_Q']) * 2 * LSB / n_eff
    assert np.abs(Vs - V).max() <= 2 * LSB * 2 / n_eff + 1e-9
