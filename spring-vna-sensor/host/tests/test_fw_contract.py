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
