# 实验复现说明

## 1. 从新检出开始

按照根目录 README 建立 Python 3.12 环境，先选择 CPU/CUDA 的 PyTorch wheel，再安装 `requirements.txt`。editable 安装让训练脚本和 `cvn-run` 使用当前源码。

```powershell
python -m pip check
python scripts/reproduce.py check --device cpu
```

这一步不需要真实数据或历史模型。测试使用临时生成的权重和数据；smoke 生成仿真轨迹，完成训练、独立评估、统计和绘图。结果默认保存在 `results/reproducibility/`。

GitHub Actions 配置了 Python 3.12 的 Windows/Linux CPU 验证。CI 状态须以远端实际运行结果为准。

首次托管前，已在只含 Git 候选文件的代码副本和全新 Windows Python 3.12 CPU 环境中验证：依赖安装成功，`pip check` 无冲突，159 项测试通过，1 项作者工作区快照审计跳过，三策略 smoke 的训练、独立评估及产物校验通过。该验证不代表所有正式论文实验已经复现。

## 2. 选择实验

配置使用相对路径，命令从仓库根目录运行。不要仅复制子配置，它的 `inherits` 父配置也必须保留。

```powershell
python scripts/reproduce.py preflight --config configs/e1_fused_multi_vehicle_pilot.json
python -m cvn_rl.experiments.run_experiment --config configs/e1_fused_multi_vehicle_pilot.json --dry-run
python -m cvn_rl.experiments.run_experiment --config configs/e1_fused_multi_vehicle_pilot.json --quick
```

`dry-run` 仍会短时运行并写出产物；`quick` 使用较小训练预算。两者均属诊断，不能据此宣称正式论文结论。去掉两者即可使用完整预算，正式配置的模型/gate 要求仍适用。

全部配置见 [config_inventory.csv](config_inventory.csv)。保留历史诊断配置便于定位建模变化，但不是每个历史入口都适合作为新实验默认值。

## 3. DQI 标签与 oracle 诊断

```powershell
python scripts/audit_dqi_labels.py --config configs/e2_d0_dqi_label_audit.json --output-dir results/d0_dqi_label_audit
python -m cvn_rl.experiments.run_experiment --config configs/e2_r1a_prediction_oracle_gate.json --results-root results/e2_r1_phase_a
```

标签审计检查 quantile 顺序、未来窗口、train-only 归一化、划分隔离和数值退化，失败返回非零退出码。R1-B 需要 R1-A 输出的通过报告：

```powershell
python -m cvn_rl.experiments.run_experiment --config configs/e2_r1b_dqi_oracle_gate.json --gate-report <R1-A-run>/analysis/e2_r1_gate.json
```

Oracle 使用特权信息，只能作为诊断；frozen/formal loader 不允许将其替代真实预测器。

## 4. 准备冻结预测器

仓库提供训练代码，不附带本机权重。占用预测与 DQI 模型入口：

```powershell
python scripts/train_bs_occupancy_predictor.py --output models/attention_lstm_bs_occupancy_h1.pt
python scripts/train_dqi_attention_predictor.py --config configs/e2_d3_attention_dqi_frozen.json --output-dir models/dqi_retrained
```

训练输出 `.pt` 和 `.meta.json`，记录结构、输入语义、划分、验证指标和 SHA-256。模型只能用于输入语义与物理配置匹配的实验。训练预算可能较大，应先检查 `--help` 并使用新输出目录。

历史配置的路径和 SHA-256 是原实验要求。重新训练不保证产生相同字节，尤其跨 CPU/GPU 时。新实验应复制 guidance/frozen 配置，在副本中填写新路径和对应哈希，保留 loader 校验，记录新配置及 seed。

## 5. R5-C2 分阶段 pipeline

`configs/r5c2_reproduce.json` 将原来从历史 `source_meta` 读取的结构和训练参数显式写入配置，避免首次检出依赖旧 DQI 结果。它继承原协议，仍需要 guidance 中声明的冻结占用预测器。

先训练占用预测器，复制 `configs/e2_d4a_fast_core_joint_prediction_dqi_factorial.json` 为新 guidance 文件，在 `prediction_plus_dqi` 条件中更新占用模型路径及哈希；再复制 `configs/r5c2_reproduce.json`，将 `guidance_source` 指向新文件。保留历史配置，使用新的 `output_root`。

设新协议为 `configs/r5c2_local.json`，执行以下阶段。全量流程涉及 11 个 DQI 目标和多个 RL 条件，不属于快速 smoke。

```powershell
python scripts/r5c2_dqi_sensitivity.py resolve --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py prepare-data --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py train-predictors --stage pilot --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py pilot-freshness --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py pilot-runtime --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py freeze --phase protocol --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py train-predictors --stage formal --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py freeze --phase runs --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py run-rl --stage formal --resume --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py summarize --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py plot --config configs/r5c2_local.json
python scripts/r5c2_dqi_sensitivity.py verify --config configs/r5c2_local.json
```

该 pipeline 保存协议哈希、划分与归一化信息、模型索引和任务索引。`--resume` 只复用兼容产物。正式协议须完成 runtime probe，不能用 `--skip-full-probe` 跳过后冻结。

## 6. 外部数据包与正式模式

新增 R5-C2 v2 代码亦保留在仓库中，但 `resolve` 依赖作者工作区的 `r5c2_v2_protected_sources.json`，以及其中列出的旧模型/结果快照。它不是首次检出即可完整启动的公共入口。公开检出的测试仍执行 v2 物理、因果性和模型测试，只跳过“作者旧工作区文件未改变”这一项本地审计；正式运行器的保护检查没有关闭。

artifact descriptor 配置需要按 loader schema 提供：

```text
manifest.json
physical_trace.npz
dqi_predictions.npz
theta_predictions.npz
```

物理轨迹、DQI 和 theta 预测须来自匹配划分和冻结模型。manifest 记录文件/模型哈希、时序截止点及验证指标。

```powershell
python scripts/assemble_frozen_descriptor_bundle.py --help
python scripts/validate_frozen_descriptor_bundles.py <bundle-root> --output results/bundle_validation.json
```

正式模式拒绝 surrogate、未来泄漏、改动后的文件和错误模型哈希。Berlin 外部验证还需要另外取得符合许可的真实数据。仓库未提供下载器或公开结果包，仅检出代码不能直接重建全部论文数字。

## 7. 保存实验与环境

保存 Git commit、冻结配置、种子、数据/模型 SHA-256、依赖版本、设备信息和输出 manifest。直接依赖固定在 `pyproject.toml`，wheel 来源在 CPU/CUDA requirements 文件中；传递依赖尚未完全锁定。

通用运行器的 `manifest.json` 自动记录 Python/依赖版本、Git commit、工作区修改状态及源码 SHA-256。没有 Git 元数据时 commit 字段为 null；smoke 另存 `environment.json` 和产物校验和。

数据/权重可通过有版本及校验和的独立发布包分发。不要用 `git add -f` 将虚拟环境、数据集和训练产物混入代码仓库。
