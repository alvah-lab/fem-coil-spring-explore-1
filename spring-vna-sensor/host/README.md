# honeycomb_host — 方案C 主机软件栈

硬件回来前用数字孪生跑通 **仿真器 → UDP → 主机解析 → 标定 → 反演 → GUI** 全流程；硬件到手后只换数据源。

```
source ../../venv/bin/activate        # 需 numpy scipy PyQt6 pyqtgraph pytest
python scripts/run_sim.py             # 进程内孪生 + GUI
python scripts/run_sim.py --udp       # sim_device 子进程 + GUI 走 UDP 回环
python -m pytest -q                   # 全部测试 (~3 分钟, 含 Neumann 对照与闭环 MC)
```

| 模块 | 作用 |
|---|---|
| `geometry.py` | lattice / EDGES / 61 观测顺序 / 柔性先验 T（复制自 pcb_verdict.py、series_scheme.py，不 import 它们） |
| `fastmodel.py` | 轴对称矢势表 + 线积分正演：全正演 ~5ms，Jacobian ~0.11s，对直接 Neumann 61 观测误差 <1e-3 |
| `twin.py` | 数字孪生：q(t) → 复阻抗 → C_off 逐驻留折叠 → 驻留级 I/Q 累加值（+ level-1 样本流） |
| `invert.py` | Tracker：双速截断 SVD GN，缓存 J，后台重线性化，只钳位不断言 |
| `pipeline.py` | 帧 → 参考驻留/电流通道标定 → Z=V/I → L_eff → Tracker → 环温 (Re/Im) |
| `protocol.py` / `sim_device.py` | UDP 协议 v0（沿用 adda_project 命令协议 + 0x11 驻留帧）与仿真 FPGA |
| `sources.py` / `recorder.py` | TwinSource / UdpSource / ReplaySource（QThread）、PipelineWorker、npz 录制回放 |
| `gui/` | PyQt6 + pyqtgraph：蜂窝热图、61 观测视图、诊断、孪生场景面板 |

关键约定见 `docs/主机协议_UDP帧格式_v0.md`。坑：旧 FastHenry 脚本导入即跑求解器；`pcb_valid_base.npz` 无效；驻留 12500 样本@62.5MSps；边观测 ~100% 依赖环-环杂化，R–R 精度不可省。
