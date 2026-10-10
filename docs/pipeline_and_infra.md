# 流程与基础设施

> 本文件描述实验运行器的基础机制。安装、检出验证及冻结模型准备步骤见 [reproducibility.md](reproducibility.md)。

## 流程

1. 加载一个 JSON 配置，并将其冻结到运行目录中。
2. 为仿真、学习和报告设置随机种子。
3. 生成或加载轨迹：PU occupancy、sensing result、DQI、DQI age、预测空闲概率、信道质量和 underlay power cap。
4. 训练或加载感知模块：DQI 和 prediction 输出必须保持 `main.tex` 中定义的语义。
5. 在相同轨迹和划分下训练/评估接入策略。
6. 针对 prediction、sensing、path-loss proxy、fading margin、PU traffic、scale 和 descriptor staleness 运行鲁棒性扫描。
7. 导出指标、摘要、图、manifest 和分析笔记。

## 运行目录约定

```text
results/<run_id>/
  manifest.json
  config.frozen.json
  metrics/
    access_metrics.csv
  tables/
    summary_by_agent.csv
  figures/
  analysis/
    experiment_notes.md
```

## 报告原则

- 配置优先：不要在临时脚本中隐藏论文参数。
- 同一轨迹，多种 agent：基线使用完全相同的信道、DQI 和 prediction 输出。
- 安全指标是强制项：PU violation rate 应与 PLR 和吞吐量并列报告。
- 统计报告：所有论文表格都应包含跨随机种子的均值和标准差。
- IEEE 输出：最终图使用 `paper_figure_style.py`，并导出 EPS/PDF/PNG。
- 分析连续性：每次运行都写入一份简短笔记，说明该运行回应了哪条审稿意见。
