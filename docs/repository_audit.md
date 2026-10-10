# 托管范围审计

审计以 `.gitignore` 过滤后的 Git 文件清单为准，不发布被忽略的历史数据。

## 保留与排除

| 内容 | 处理 | 原因 |
| --- | --- | --- |
| `src/`、`configs/`、实验分析脚本、`tests/` | 保留 | 模型、仿真、配置继承和验证相互复用 |
| README、依赖声明、复现文档、CI | 保留 | 新机器安装、验证和启动实验 |
| `.venv*`、助手/编辑器配置、缓存 | 忽略 | 本机环境，可重新建立 |
| `results/`、`models/`、数据、日志、图表 | 忽略 | 运行产物或外部输入，不适合普通 Git |
| `revision_round2/`、`archive/`、内部讨论文档 | 忽略 | 审稿材料、论文副本、本机路径和历史笔记 |
| 论文整理、Windows EMF 导出脚本 | 忽略 | 与实验复现无关且依赖本机论文环境 |

排除均为 Git 忽略规则，未删除本地数据。没有选择许可证，也没有改变远端可见性。

## 文件与配置清单

候选清单：298 个文件，1.60 MiB（工作区文本大小）。

| 目录/文件 | 文件数 |
| --- | ---: |
| `.gitattributes` | 1 |
| `.github` | 1 |
| `.gitignore` | 1 |
| `AGENTS.md` | 1 |
| `README.md` | 1 |
| `configs` | 115 |
| `docs` | 6 |
| `pyproject.toml` | 1 |
| `requirements-cpu.txt` | 1 |
| `requirements-cuda.txt` | 1 |
| `requirements.txt` | 1 |
| `scripts` | 67 |
| `src` | 70 |
| `tests` | 31 |

配置与外部输入清单见 [config_inventory.csv](config_inventory.csv)。 共 115 个入口，其中 58 个声明了未随 Git 发布的输入。

`no-declared-external-inputs` 只表示未发现显式文件依赖，不表示实验已通过科学 gate。历史配置的 frozen 哈希和 reuse 约束保留。

## 重新审计

```powershell
python scripts/audit_release.py
git status --short
git diff --cached --stat
```

脚本解析 Python 语法、拒绝大于 10 MiB 的候选文件、检查常见凭据特征，追踪配置继承及 guidance/scenario 输入。这不替代人工审查。
