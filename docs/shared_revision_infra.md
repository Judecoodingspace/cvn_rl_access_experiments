# 可复用的物理 trace 与评估场景接口

## 层次与边界

公共层位于 `simulation/revision_physics.py`、`revision_physics_spec.py`、`revision_scenarios.py`。
它负责物理量、估计信息、随机流、场景和manifest，不定义某一实验的训练预算、参数扫描矩阵或科学结论。
`experiments/r5c2_v2/physics.py`保留原导入路径，转发到公共实现；R5-C2的固定参数仍由其独立validator约束。

| 层次 | 当前能力 | 仍由具体实验决定 |
|---|---|---|
| 真实物理 | VT–BS、VT–PU、PU→VT gain；rate/SNR/RSS同源；UMi LOS/NLOS；阴影与小尺度衰落；位置/速度/Doppler | 场景、参数范围、采样预算；未实现完整3GPP簇级或NR-V2X协议栈 |
| 基站获知信息 | 位置报告误差、估计增益偏差/随机误差、CAM/额外延迟、cap时钟、保护裕量、估计路损模型 | 误差分布/代表值；估计模型失配的具体假设 |
| 新评价分布 | PU过程、感知误差、到包负载、速度/速度变化、方向、链路传播条件 | 持出场景矩阵、冻结策略加载、重复评估与统计；复杂转弯/真实道路轨迹尚需扩展 |
| 控制/模型 | 场景对象记录并检查冻结policy/predictor/scaler/Rref引用 | 实际模型加载、执行前后权重校验和端到端评估由runner实现 |

现有R5-C2 runner仍专用于其N10/M10、11目标、19Full条件。其他实验调用公共物理层，并编写各自runner/contract，不能把它们塞进R5-C2矩阵或将该runner宣称为通用OOD评估器。

## 真实世界与估计信息分开

`generate(entry, seed, slots, vehicles, split, future_slots=0)`产生真实位置、gain、SNR/RSS/rate、PU占用、感知、到包、争用，以及基站收到的信息。
独立随机流保证改变估计误差不会顺带改变真实传播；改变某条链路的传播参数也不会改其他链路随机流。

```python
from cvn_rl.simulation.revision_physics import generate
from cvn_rl.simulation.revision_scenarios import observation_variant, component_hashes

base = generate(entry, seed=1, slots=1000, vehicles=10, split="eval")
variant, variant_entry = observation_variant(base, entry, {
    "position_error_std_m": 5.0,
    "estimated_gain_bias_db": -3.0,
    "protection_margin": 3.0,
})
assert component_hashes(base)["truth_hash"] == component_hashes(variant)["truth_hash"]
```

估计增益的dB偏差乘在线性增益上，正偏差使估计更大；负偏差可能放宽功率上限。随机增益误差用独立Gaussian dB过程，按CAM时刻生成并随报告/cap持有。
位置误差默认水平方向Gaussian；可显式选`position_error_distribution="uniform"`，每个水平坐标边界为±sqrt(3)×σ，使标准差仍为σ。
若估计位置超出UMi适用域，明确失败，不偷偷夹断距离；实验contract须选合法几何/误差分布。
初始位置在slot0已登记，额外延迟作用于后续CAM；该初始化假设写入manifest。

`observation_variant`只接受估计/报告/cap字段，复制数组并检查真实与外生数组未改变，不修改传入base。

## 只改变PU链路的不确定性

```python
from cvn_rl.simulation.revision_scenarios import scenario_entry

pu_uncertain = scenario_entry(entry, {
    "physics": {"link_overrides": {"vt_pu": {"shadow_sigma_db": 6.0}}}
})
trace = generate(pu_uncertain, seed=1, slots=1000, vehicles=10, split="eval")
```

支持`vt_bs`、`vt_pu`、`pu_vt`三条链路独立配置路损、shadow σ/相关距离、Rician K及`rician/rayleigh/none`小尺度族。
VT–PU真实shadow变化时，VT–BS服务源、DQI输入源和估计cap保持原设定；实际干扰可能改变。
改变真实传播条件时，实际gain不能要求与参考场景逐字节相同；应记录同seed/潜在随机流和唯一变化项。

`component_hashes`分别记录truth、observation、exogenous和full hash。信息误差比较匹配truth/exogenous；传播/流量分布变化比较匹配seed与未变化组件。
不能直接沿用R5-C2分析器“所有完整physical hash相同”的判据来分析这些场景。

## 冻结模型的持出分布场景

`build_frozen_evaluation_case`接收训练entry、评价override、冻结引用、seed及case_id。
引用须包含policy、DQI模型、占用模型、预处理SHA256和训练Rref；场景对象不训练、不拟合scaler，`assert_frozen_references`拒绝替换这些引用。
这层检查的是引用契约，实际文件hash与运行后权重仍须由后续OOD runner核验。

支持改变PU转移/已有PU过程族、Pd/Pfa、到包概率、传播参数、固定速度、方向和速度schedule。
`speed_schedule=[{"start_slot": 1000, "speed_kmh": 90.0}]`让同一条轨迹在该slot改变速度；位置连续、速度/Doppler及时更新，参考vmax保持配置值。
复杂道路转弯、外部真实轨迹输入、在线适应训练需另建接口；不能将当前简化直线路径称为已覆盖所有mobility patterns。

区分两种科学问题：冻结原策略在新条件下评价，测部署鲁棒性；每条件重新训练，测该条件下的学习结果。冻结OOD证据采用前者，允许内部时序状态随观测变化，但不更新权重、不重拟合归一化。

## 标准来源和独立假设

路径损耗引用[3GPP TR38.901 v18.1.0 Release18，Table7.4.1-1](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.01.00_60/tr_138901v180100p.pdf)。
LOS采用双斜率与breakpoint，NLOS采用max(LOS,NLOS)，固定有效环境高度1m并检查距离、频率和天线高度域。
默认参考选择LOS，未采样标准LOS概率；NLOS由具体场景显式选择。

Rician K、一步J0相关AR1、shadow相关距离、速度、噪声、功率、PU容限、流量、感知与时钟是单独声明的假设。
所选shadow σ恰好等于某标准表数值，也不代表完整传播、波形、接入或仿真框架符合3GPP/NR-V2X。
`validate_physics`给出可导出的standard/independent_assumptions/domain清单；R5-C2的manifest额外列论文固定值。

## 验证和版本

```powershell
python -m pytest tests/test_revision_physics.py tests/test_r5c2_v2_physics.py tests/test_r5c2_v2_pipeline.py
```

新测试覆盖信息误差不改变真实世界、PU链路变化不改变BS服务源、延迟/裕量/偏差方向、LOS/NLOS公式与breakpoint、速度变化位置连续，以及冻结引用保护。
公共层提取时另与提交85c7218的N10/N15默认物理输出逐数组比较，27个数组和完整physical hash相同。
默认参数没有增加新性能sweep或改变R5-C2合同；历史冻结/旧结果原样保留。

缓存与协议仍按源码/配置/hash检查。公共源码更新后不能把旧freeze重新签名冒充一致；新实验使用新的输出根目录或经分项兼容检查复用。
公开检出没有作者私有保护快照时，v2使用repository-only模式并记录依赖hash，不宣称已审计作者历史工作区；存在的本地保护快照仍强制检查。
