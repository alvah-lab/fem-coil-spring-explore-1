"""主机↔FPGA UDP 协议 (v0).

沿用 /work/fpga/xilinx/flow-1/adda_demos/adda_project 的 12 字节大端命令与 0x10 批量包 (硬件验证过),
新增 0x11 驻留帧 与 0x12 振动流. 详见 docs/主机协议_UDP帧格式_v0.md.
"""
from __future__ import annotations
import struct
import zlib
from dataclasses import dataclass
import numpy as np
from .twin import Frame, DWELL_DTYPE, N_DWELL

FPGA_IP, FPGA_PORT = '192.168.2.128', 5000
HOST_IP = '192.168.2.1'
DATA_PORT = 5001            # FPGA→主机 驻留帧流 (主机监听)
VIB_PORT = 5002

# ---- 命令 (adda_project) ----
OP_REG_WRITE, OP_REG_READ, OP_BULK_READ, OP_IDENTIFY = 0x01, 0x02, 0x03, 0x04
CMD_FMT = '>BBHBBHI'        # opcode, rsv, seq, reg, ch, len, data  = 12B
RSP_FMT = '>BBHII'          # opcode|0x80, status, seq, payload, fw_version = 12B
DEVICE_ID = 0xE7F10001

# ---- 寄存器 (0x00-0x14 沿用; 0x20+ Track-3 新增) ----
REG = dict(DEVICE_ID=0x00, CTRL=0x04, STATUS=0x08, BULK_ADDR=0x10, BULK_DATA=0x14,
           DWELL_NSAMP=0x20, NCO_FREQ_WORD=0x24, NCO_PHASE=0x28, FRAME_CTRL=0x2C,
           DWELL_TABLE_ADDR=0x30, DWELL_TABLE_DATA=0x34, N_DWELL=0x38, PGA_SEL=0x3C,
           REF_DWELL_EN=0x40, NULL_I=0x44, NULL_Q=0x48, STREAM_MODE=0x4C, DRIVE_AMP=0x50,
           HOST_IP=0x54, HOST_PORT=0x58, FRAME_ID=0x5C)
FRAME_CTRL_STOP, FRAME_CTRL_RUN, FRAME_CTRL_SINGLE = 0, 1, 2

# ---- 数据包 ----
TAG_BULK, TAG_FRAME, TAG_VIB = 0x10, 0x11, 0x12
PROTO_VER = 0
FRAME_HDR = '>BBHIQHIBB'    # tag, ver, seq, frame_id, t_ticks, dwell_nsamp, nco_word, n_dwell, flags = 24B
FRAME_HDR_LEN = struct.calcsize(FRAME_HDR)
DWELL_BE = DWELL_DTYPE.newbyteorder('>')   # 线上大端 20B/驻留

def pack_cmd(opcode: int, seq: int, reg: int = 0, ch: int = 0, length: int = 0, data: int = 0) -> bytes:
    return struct.pack(CMD_FMT, opcode, 0, seq & 0xFFFF, reg & 0xFF, ch & 0xFF, length & 0xFFFF, data & 0xFFFFFFFF)

def unpack_cmd(b: bytes):
    return struct.unpack(CMD_FMT, b[:12])

def pack_rsp(opcode: int, status: int, seq: int, payload: int = 0, fw: int = 0x00010000) -> bytes:
    return struct.pack(RSP_FMT, (opcode | 0x80) & 0xFF, status & 0xFF, seq & 0xFFFF, payload & 0xFFFFFFFF, fw)

def unpack_rsp(b: bytes):
    return struct.unpack(RSP_FMT, b[:12])

def encode_frame(fr: Frame) -> bytes:
    d = np.ascontiguousarray(fr.dwells.astype(DWELL_BE))
    hdr = struct.pack(FRAME_HDR, TAG_FRAME, PROTO_VER, fr.seq & 0xFFFF, fr.frame_id & 0xFFFFFFFF,
                      fr.t_ticks & 0xFFFFFFFFFFFFFFFF, fr.dwell_nsamp & 0xFFFF, fr.nco_word & 0xFFFFFFFF,
                      len(d) & 0xFF, fr.flags & 0xFF)
    body = hdr + d.tobytes()
    return body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)

class ProtocolError(ValueError):
    pass

def decode_frame(b: bytes) -> Frame:
    if len(b) < FRAME_HDR_LEN + 4:
        raise ProtocolError('short packet')
    tag, ver, seq, fid, ticks, nsamp, nco, n, flags = struct.unpack(FRAME_HDR, b[:FRAME_HDR_LEN])
    if tag != TAG_FRAME:
        raise ProtocolError(f'bad tag 0x{tag:02x}')
    if ver != PROTO_VER:
        raise ProtocolError(f'bad version {ver}')
    need = FRAME_HDR_LEN + n * DWELL_BE.itemsize + 4
    if len(b) < need:
        raise ProtocolError('truncated')
    body, crc = b[:need - 4], struct.unpack('>I', b[need - 4:need])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ProtocolError('crc')
    d = np.frombuffer(body[FRAME_HDR_LEN:], dtype=DWELL_BE, count=n).astype(DWELL_DTYPE)
    return Frame(seq, ticks, nsamp, nco, d, fid, flags, None)

@dataclass
class Command:
    """主机→设备 高层命令 (sources/sim_device 共用)."""
    name: str
    value: int | np.ndarray | None = None
    def to_packets(self, seq0: int = 0) -> list[bytes]:
        pk = []
        if self.name == 'start':
            pk.append(pack_cmd(OP_REG_WRITE, seq0, REG['FRAME_CTRL'], 0, 0, FRAME_CTRL_RUN))
        elif self.name == 'stop':
            pk.append(pack_cmd(OP_REG_WRITE, seq0, REG['FRAME_CTRL'], 0, 0, FRAME_CTRL_STOP))
        elif self.name == 'dwell_table':
            tbl = np.asarray(self.value, dtype=np.uint16)
            for i, wv in enumerate(tbl):
                pk.append(pack_cmd(OP_REG_WRITE, seq0 + 2 * i, REG['DWELL_TABLE_ADDR'], 0, 0, i))
                pk.append(pack_cmd(OP_REG_WRITE, seq0 + 2 * i + 1, REG['DWELL_TABLE_DATA'], 0, 0, int(wv)))
            pk.append(pack_cmd(OP_REG_WRITE, seq0 + 2 * len(tbl), REG['N_DWELL'], 0, 0, len(tbl)))
        elif self.name in REG:
            pk.append(pack_cmd(OP_REG_WRITE, seq0, REG[self.name], 0, 0, int(self.value)))
        elif self.name == 'identify':
            pk.append(pack_cmd(OP_IDENTIFY, seq0))
        else:
            raise ValueError(self.name)
        return pk
