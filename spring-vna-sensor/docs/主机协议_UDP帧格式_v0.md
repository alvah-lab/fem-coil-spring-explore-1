# 主机↔FPGA UDP 协议 v0（方案C 19 单元验证板，交固件 session）

主机侧实现：`spring-vna-sensor/host/honeycomb_host/protocol.py`；仿真设备：`sim_device.py`（可直接对着主机 GUI 联调）。

## 0. 沿用与出处

命令通道**原样沿用** `/work/fpga/xilinx/flow-1/adda_demos/adda_project/`（AX7102 + AN9238 + AN9767 + 千兆 UDP，硬件验证过）的 12 字节命令/响应与 `0x10` 批量包，`udp_cmd_bridge_v2.v` 不改；只在 `cmd_interface.v` 寄存器表上加 Track-3 寄存器，并新增一种 FPGA→主机数据包 `0x11`。

地址：FPGA `192.168.2.128:5000`（命令），主机 `192.168.2.1`；数据流发往主机 `HOST_PORT`（默认 5001），振动流 5002。全部大端。

## 1. 命令包（主机→FPGA，12 字节）

```
byte 0    opcode   0x01 REG_WRITE / 0x02 REG_READ / 0x03 BULK_READ / 0x04 IDENTIFY
byte 1    reserved
byte 2-3  seq_id   u16
byte 4    reg_addr u8
byte 5    channel  u8
byte 6-7  length   u16
byte 8-11 data     u32 (REG_WRITE)
```
响应 12 字节：`opcode|0x80, status, seq_id u16, payload u32, fw_version u32`。`IDENTIFY` 的 payload = `DEVICE_ID 0xE7F10001`。

## 2. 寄存器表

| 地址 | 名称 | 说明 |
|---|---|---|
| 0x00 | DEVICE_ID | R |
| 0x04/0x08/0x10/0x14 | CTRL / STATUS / BULK_ADDR / BULK_DATA | 沿用 adda_project |
| **0x20** | DWELL_NSAMP | 每驻留积分样本数。**必须是公共周期的整数倍**：fs=62.5MSps、f0=8MHz 时 125 样本 = 16 周期，默认 **12500 = 200µs = 1600 周期**；若 ADC 改 65MSps 则 13000 |
| 0x24 | NCO_FREQ_WORD | u32，f0 = word/2³²·fs。默认 8MHz@62.5M = 0x20C49BA6；备选 f0 = fs/8 = 7.8125MHz（每周期恰 8 样本，NCO 退化为 8 点表） |
| 0x28 | NCO_PHASE | 起始相位（沿用 `adda_project/top.v` 的 trig_armed 绝对相位参考：驻留首样本对齐 DAC sin(0)） |
| 0x2C | FRAME_CTRL | 0 停 / 1 连续 / 2 单帧 |
| 0x30 | DWELL_TABLE_ADDR | 驻留表写地址 (0..127) |
| 0x34 | DWELL_TABLE_DATA | 低 16 位 = A704 驻留字 `[drv5][sns5][pga2][ref1][vna1][spare2]`（DUAL_FPGA_IO.md）；写入后地址自增 |
| 0x38 | N_DWELL | 每帧驻留数（默认 63 = 61 观测 + 参考 + 电流标定） |
| 0x3C | PGA_SEL | 全局 PGA 覆盖（0 = 按驻留字） |
| 0x40 | REF_DWELL_EN | 参考驻留使能 |
| 0x44/0x48 | NULL_I / NULL_Q | 调零音幅相（AN9767 CH2），v1 保留 |
| 0x4C | STREAM_MODE | 0 驻留帧 / 1 驻留帧+振动流 |
| 0x50 | DRIVE_AMP | 驱动幅度码 |
| 0x54/0x58 | HOST_IP / HOST_PORT | 数据流目的地 |
| 0x5C | FRAME_ID | R，当前帧号 |

默认驻留表（主机 `twin.default_dwell_table()`）：obs 0..18 自观测 `drv=sns=i, pga=0`；obs 19..60 边观测 `(i,j)` 按 `geometry.EDGES` 字典序，`pga=2`；obs 61 参考驻留 `ref=1, pga=1`；obs 62 电流标定 `vna=1`。

## 3. 驻留帧（FPGA→主机，tag 0x11，一帧一包）

```
头 24 字节:  tag u8=0x11 | ver u8=0 | seq u16 | frame_id u32 | t_ticks u64 (逻辑时钟计数)
           | dwell_nsamp u16 | nco_word u32 | n_dwell u8 | flags u8
每驻留 20 字节 × n_dwell:
           obs_id u8 | dwell_word u16 | flags u8 (bit0 饱和, bit1 参考, bit2 电流标定)
           | V_I i32 | V_Q i32 | I_I i32 | I_Q i32
尾 4 字节:  CRC32 (zlib, 覆盖头+体)
```
63 驻留 = 24 + 1260 + 4 = 1288 字节。

**累加约定**（主机与固件必须一致，已由主机 level-1 样本流测试钉死）：
```
v[n] = ADC 码 (12-bit, ±5V FS, LSB = 10/4096 V)，n = 0..N-1，前 blank 个样本置零（开关稳定 5µs ≈ 312 样本）
I_acc = Σ v[n]·cos(2π·k·n/N),  Q_acc = Σ v[n]·sin(2π·k·n/N),  k = N·f0/fs (整数)
复幅度 V = (I_acc − j·Q_acc)·2·LSB/N_eff   (N_eff = N − blank)
```
两路 ADC：`V_*` = 接收电压通道（经 PGA），`I_*` = 电流检测通道（Rs 10Ω×10 = 100 V/A）。主机算 `Z = V/PGA / (I/100)`，`L_eff = Im(Z)/ω`，实部给温度通道。累加字长 ≥20 位有效，i32 足够（12-bit × 12500 = 24 位）。

## 4. 振动流（tag 0x12，可选）

每驻留 10 点 IQ（50kSps，解调低通 20kHz）：头同上 + `obs_id u8, n_pts u8, {I i32, Q i32}×n_pts`。见《滑移检测振动通道Brief》。

## 5. 时序约定

- 驻留周期严格固定（驻留切换梳齿落在 5kHz 整数倍，振动通道靠固定陷波）。
- 每驻留：写 A704 驻留字 → 等 READY → 等保护时间 → 积分 DWELL_NSAMP 样本。
- 帧率 = 1/(N_DWELL × (DWELL_NSAMP/fs + 开关时间)) ≈ 77Hz。

## 6. 主机侧联调

```
cd spring-vna-sensor/host
python -m honeycomb_host.sim_device --scene point_press        # 仿真 FPGA (127.0.0.1:5000 → :5001)
python -m honeycomb_host.gui.main --source udp --device 127.0.0.1:5000
```
固件实现后把 `--device 192.168.2.128:5000` 即可，主机不改。
