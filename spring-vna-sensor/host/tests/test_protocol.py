import numpy as np, pytest
from honeycomb_host import protocol as P
from honeycomb_host.twin import Twin, Scenes, Environment, NoiseModel

def test_frame_roundtrip_and_crc():
    tw = Twin(env=Environment(noise=NoiseModel(preset='off')), scene=Scenes.point_press())
    fr = tw.step()
    b = P.encode_frame(fr)
    assert len(b) == P.FRAME_HDR_LEN + 63 * 20 + 4
    fr2 = P.decode_frame(b)
    assert fr2.seq == fr.seq and fr2.t_ticks == fr.t_ticks and fr2.nco_word == fr.nco_word
    assert np.array_equal(fr2.dwells, fr.dwells)
    bad = bytearray(b); bad[50] ^= 0x10
    with pytest.raises(P.ProtocolError):
        P.decode_frame(bytes(bad))
    with pytest.raises(P.ProtocolError):
        P.decode_frame(b[:100])

def test_cmd_pack():
    b = P.pack_cmd(P.OP_REG_WRITE, 5, P.REG['DWELL_NSAMP'], 0, 0, 12500)
    assert len(b) == 12 and P.unpack_cmd(b) == (1, 0, 5, 0x20, 0, 0, 12500)
    r = P.pack_rsp(P.OP_IDENTIFY, 0, 5, P.DEVICE_ID)
    assert P.unpack_rsp(r)[0] == 0x84 and P.unpack_rsp(r)[3] == P.DEVICE_ID
    pk = P.Command('dwell_table', np.arange(63, dtype=np.uint16)).to_packets()
    assert len(pk) == 127

def test_sim_device_loop():
    import asyncio, socket, threading, time
    from honeycomb_host.sim_device import SimDevice
    tw = Twin(env=Environment(noise=NoiseModel(preset='off')), scene=Scenes.rest())
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(('127.0.0.1', 0)); rx.settimeout(2)
    dev = SimDevice(tw, '127.0.0.1', 0, '127.0.0.1', rx.getsockname()[1], rate_hz=200)
    dev.sock.bind(('127.0.0.1', 0)); cmd_port = dev.sock.getsockname()[1]
    async def run():
        await asyncio.wait_for(dev.run(), 1.5)
    def runner():
        try:
            asyncio.run(run())
        except asyncio.TimeoutError:
            pass
    dev.sock.close(); dev.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); dev.sock.setblocking(False)
    dev.host = '127.0.0.1'; dev.cmd_port = cmd_port
    th = threading.Thread(target=runner, daemon=True); th.start()
    frames = 0; t0 = time.time()
    while time.time() - t0 < 1.0:
        try:
            b, _ = rx.recvfrom(65535)
        except socket.timeout:
            break
        P.decode_frame(b); frames += 1
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); tx.settimeout(1)
    tx.sendto(P.pack_cmd(P.OP_IDENTIFY, 1), ('127.0.0.1', cmd_port))
    rsp, _ = tx.recvfrom(64)
    assert P.unpack_rsp(rsp)[3] == P.DEVICE_ID
    dev.running = False; th.join(3)
    assert frames > 20
