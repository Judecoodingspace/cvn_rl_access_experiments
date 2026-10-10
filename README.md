# CVN RL Access Experiments

面向认知车联网（CVN）的离散信道与接入模式强化学习实验。仓库包含仿真环境、ESN-DDQN/DDQN/DQN/LSTM-DDQN/Discrete SAC/PPO 等策略、DQI 与占用预测器、实验配置、统计分析和绘图入口。

目标是让新机器检出代码后，能够先验证环境，再开展配置驱动的实验。原始数据、训练权重、历史结果、审稿材料和本机配置不随代码上传；这些内容在本地保留。

## 安装与首次验证

推荐 Python 3.12。以下命令在仓库根目录执行。

```powershell
git clone https://github.com/Judecoodingspace/cvn_rl_access_experiments.git
cd cvn_rl_access_experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
python -m pip check
python scripts/reproduce.py check --device cpu
```

Linux/macOS 将激活命令替换为 `source .venv/bin/activate`。CPU 安装文件适用于 PyTorch 支持的 CPU 平台；Apple Silicon 可先执行 `python -m pip install torch==2.11.0` 再安装 `requirements.txt`。

NVIDIA GPU 机器使用 `requirements-cuda.txt` 替换 CPU 安装文件，安装后执行 `python scripts/reproduce.py doctor --device cuda`。该文件对应参考环境的 CUDA 12.8 wheel；驱动须兼容。需要其他 CUDA 版本时，先安装适配的 PyTorch 2.11.0，再安装通用依赖。

`requirements.txt` 安装项目、测试及可选 backbone/Parquet 依赖。只运行核心实验可用 `python -m pip install -e ".[dev]"`。核心策略使用仓库内的 PyTorch 实现，无须另装 Stable-Baselines3、TorchRL、CuPy、torchvision 或 torchaudio。

## 可复现 pipeline

```text
检出代码 → 安装依赖 → 环境检查 → 完整测试
        → 生成仿真轨迹 → 短时训练 → 独立轨迹评估
        → 指标/统计表/图表 → manifest 与产物校验
```

`python scripts/reproduce.py check` 执行完整测试和 smoke 实验。也可以分步执行：

```powershell
python scripts/reproduce.py doctor --device cpu
python scripts/reproduce.py test
python scripts/reproduce.py preflight --config configs/repro_smoke.json
python scripts/reproduce.py smoke --device cpu
```

Smoke 配置为 `configs/repro_smoke.json`：3 个信道、2 辆车、1 个种子、128 个训练时隙，对比 ESN-DDQN、DDQN 和固定 overlay。它生成因果 surrogate 描述符，用于验证端到端执行，不作为论文正式实验结果。

产物写入 `results/reproducibility/<run_id>/`，包括冻结配置、CSV 指标/摘要、PDF 图、manifest 和记录依赖版本及产物 SHA-256 的 `environment.json`。每次运行创建独立目录。Windows/Linux 的 GitHub Actions 会安装 CPU 环境、检查依赖、执行测试和 smoke，并上传产物。

## 开展实验

先检查配置的外部输入，再运行：

```powershell
python scripts/reproduce.py preflight --config configs/e1_fused_multi_vehicle_pilot.json
python -m cvn_rl.experiments.run_experiment --config configs/e1_fused_multi_vehicle_pilot.json --dry-run
python -m cvn_rl.experiments.run_experiment --config configs/e1_fused_multi_vehicle_pilot.json --quick
```

安装后的 `cvn-run` 与 `python -m cvn_rl.experiments.run_experiment` 等价。运行器默认自动选择 CUDA/CPU；PowerShell 用 `$env:CVN_RL_DEVICE="cpu"`，Linux 用 `export CVN_RL_DEVICE=cpu` 指定设备。

| 入口 | 用途 | 新检出的前置条件 |
| --- | --- | --- |
| `configs/repro_smoke.json` | 首次验证 | 无外部数据或权重 |
| `configs/e1_fused_multi_vehicle_pilot.json` | 多车辆协调 pilot | 自动生成 surrogate 轨迹 |
| `configs/e2_d0_dqi_label_audit.json` | DQI 标签审计 | 自动生成仿真数据 |
| `configs/e2_r1a_prediction_oracle_gate.json` | 预测 oracle 诊断 | 配置校准与 gate；oracle 不属于正式预测结果 |
| `configs/r5c2_reproduce.json` | 分阶段 DQI 敏感性实验 | 须先准备 guidance 预测器；见复现说明 |
| `configs/*formal*.json` / frozen 配置 | 正式训练、消融与鲁棒性 | 按配置准备真实冻结模型/数据包及所需 gate |

正式配置保留原模型 SHA-256 和 gate 约束。重新训练得到不同权重时，应建立新配置并记录新哈希，不应移除校验或把 surrogate 当成正式模型。`preflight` 是输入清单检查，实际 loader 继续执行因果性、模型哈希和数据划分校验。

详细步骤见 [复现说明](docs/reproducibility.md)，托管范围及全部配置输入清单见 [仓库审计](docs/repository_audit.md)。

## 仓库结构

```text
src/cvn_rl/    agents / predictors / simulation / experiments / reporting
configs/       继承式 JSON 配置，包括历史诊断配置
scripts/       训练、验证、分析、绘图及 reproduce.py
tests/         自动测试，临时数据与权重由 fixture 生成
docs/          复现说明、实验设计和托管范围
.github/       Windows/Linux CPU 验证 workflow
```

源码和配置之间复用较多，因此保留完整实验代码。历史配置是否有外部输入，应依据审计表判断，文件名中的 `formal` 不代表已经完成论文级验证。忽略目录由程序按需创建，不需要拷贝本机的 `results/`、`models/` 或虚拟环境。

## 复现边界

该仓库提供可重建的仿真实验和运行流程；未附带 Berlin 等真实数据、历史冻结权重或论文结果包。正式论文结果的逐项复现仍需要匹配的数据版本、权重哈希、配置和种子。直接依赖已固定版本，传递依赖尚未提供跨平台完整 lockfile；跨硬件训练不保证逐字节一致。

当前未指定开源许可证。代码托管不等于已授予开源使用许可；正式公开发布前需由作者确定许可及数据/模型发布范围。
