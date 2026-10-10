# RL 实验设计

## E1. RL 基线对比

目的：
证明在修订后的质量感知与预测感知离散接入问题中，ESN 辅助能够改善收敛表现。

对比策略：

- 提出的 ESN-DDQN；
- 不带 ESN reservoir 的 DDQN；
- DQN；
- 采用离散信道-模式动作的 PPO；
- 表格型 Q-learning；
- 固定 overlay；
- 固定 underlay。

指标：
PLR、VT 吞吐量、PU 吞吐量、PU 干扰违规率、平均选中 DQI、underlay 选择比例、收敛时隙数、训练时间和推理时延。

## E2. 状态与奖励消融

目的：
区分 sensing、DQI、prediction 和 PU 保护层各自的贡献。

状态变体：
`delta`、`fresh_dqi`、`theta`、两两组合、`full_main_tex_state` 和 `full_plus_reviewer_protection`。

奖励/保护变体：

- 仅使用 reward gate；
- reward gate 加 occupancy/power-cap 状态；
- 使用动作掩码的三层保护。

预期论文产物：
一张表，展示 DQI 有助于信道质量选择，prediction 有助于避开 PU 回归，而保护层能够降低 PU 违规。

## E3. Prediction 与 Sensing 鲁棒性

目的：
回答当感知不完美时，接入策略是否仍然安全。

扫描项：

- prediction false alarm 和 missed detection；
- sensing detection probability 和 false alarm；
- path-loss 估计误差；
- fading margin 权衡。

预期论文产物：
一张热力图或成对曲线，展示 missed detection 比 false alarm 更危险，并说明完整策略比消融策略更能保持 PU 安全性。

## E4. 规模与 PU 流量敏感性

目的：
超越 `M=3,N=5` 和 `M=5,N=3` 的小规模设置。

场景：
paper-scarce、paper-abundant、medium-dense 和 large-dense。

流量族：
包含多个 `p01/p10` 的固定 Markov、时变 Markov、突发 ON/OFF、带抖动的周期流量，以及 mixed-by-channel。

预期论文产物：
一张规模/流量敏感性表，包含不同随机种子下的均值和标准差。

## E5. 开销与陈旧性敏感性

目的：
量化集中式广播成本和描述符陈旧性。

扫描项：
`T_upd`、DQI 有效时间、每信道 payload bit 数，以及可选的推理时延预算。

预期论文产物：
一张部署可行性表，将性能退化与广播开销关联起来。

## 为什么不把连续控制基线作为主要基线

论文中的动作是 `(channel, mode)`：信道选择是有限集合，接入模式是二元选择。DDPG、TD3 和 SAC 主要面向连续控制，并且需要额外的离散动作映射。因此，DQN/DDQN 和 PPO-discrete 是主要 DRL 基线。如果后续把功率纳入动作空间，可以将连续控制变体作为未来工作讨论。
