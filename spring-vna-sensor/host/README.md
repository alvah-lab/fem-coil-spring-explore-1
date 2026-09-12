# honeycomb_host — 方案C 主机软件栈

硬件回来前用数字孪生跑通 **仿真器 → UDP → 主机解析 → 标定 → 反演 → GUI** 全流程；硬件到手后只换数据源。

```
./run_gui.sh                          # 进程内孪生 + GUI (SCENE=sweep NOISE=off 等环境变量可选)
./run_gui.sh udp                      # 仿真 FPGA 子进程 + GUI 走 UDP 回环
./run_gui.sh board                    # 真实板子 192.168.2.128:5000
./run_gui.sh replay rec.npz           # 回放录制
source ../../venv/bin/activate && python -m pytest -q     # 31 项测试 ~40s
```
GUI 操作：左侧"孪生场景"面板拖施力点/压深/剪切/接触温度后立即生效（无需点按钮；"应用预置"用于切换预置场景）；中央蜂窝图点击单元可看其时序；右侧标签页切换"观测"与"3D 场形变"（鼠标拖动旋转、滚轮缩放）；工具栏"录制/保存录制"出 npz。

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

## 回板测试面板（左侧"回板测试"标签，2026-09-12）
状态 / 单驻留 / 基线 / 单元扫描 / 垫片标定 / 噪声-稳定性 六个页，直接消费任意驻留表的原始帧，不经跟踪器。
先在数字孪生上演练：`docs/回板测试手册_数字孪生.md`。核心逻辑在 `honeycomb_host/bringup.py`（`frame_to_Z`、`Baseline`、`CalibLog`、`RunningStats`、`model_shim`），
孪生新增 `Scenes.no_rings()` / `Scenes.shim(unit, gap, tilt, dir, dx, dy)` 与任意驻留表驱动；`sim_device.py --shim 9:1.75:5:0`。
