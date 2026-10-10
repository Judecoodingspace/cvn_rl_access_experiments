"""Audit Git-selected files and list external inputs of experiment configs."""
from __future__ import annotations

import ast
from collections import Counter
import csv
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cvn_rl.experiments.config import load_config


def selected_files() -> list[str]:
    result = subprocess.run(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'ls-files',
         '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted(set(result.stdout.decode('utf-8').strip('\0').split('\0')) - {''})


def config_inputs(path: Path, seen=None) -> set[str]:
    seen = set() if seen is None else seen
    if path in seen:
        return set()
    seen.add(path)
    config = load_config(path)
    inputs = set()
    def visit(value, key=''):
        if isinstance(value, dict):
            for name, item in value.items():
                visit(item, name)
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str):
            if key in {'scenario_source', 'guidance_source'}:
                dependency = ROOT / value
                if dependency.is_file():
                    if key == 'guidance_source' and config.get('protocol_version') == 'r5c2_v1':
                        guidance = load_config(dependency)
                        full = next(c for c in guidance['module_conditions'] if c['name'] == 'prediction_plus_dqi')
                        # R5-C2 retrains DQI and replaces those model fields in resolve_protocol.
                        for name, item in full['descriptors'].items():
                            if name.endswith('model_path') and not name.startswith('dqi_'):
                                inputs.add(item)
                    else:
                        inputs.update(config_inputs(dependency, seen))
                else:
                    inputs.add(value)
            elif key.endswith('model_path') or key in {
                'source_meta', 'bundle_path_template', 'trace_path', 'dataset_path', 'data_path'
            }:
                inputs.add(value)
    visit(config)
    return inputs


def main() -> int:
    files = selected_files()
    problems = []
    credential = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for name in files:
        path = ROOT / name
        if path.stat().st_size > 10 * 1024 * 1024:
            problems.append(f'Large file: {name}')
        content = path.read_text(encoding='utf-8-sig')
        if path.suffix == '.py':
            ast.parse(content, filename=name)
        if credential.search(content):
            problems.append(f'Credential-like content: {name}')
    if problems:
        raise RuntimeError('\n'.join(problems))
    rows = []
    file_set = set(files)
    for path in sorted((ROOT / 'configs').glob('*.json')):
        config = load_config(path)
        external = [name for name in sorted(config_inputs(path)) if name not in file_set]
        constraints = []
        if config.get('requires_pilot_gate') or config.get('requires_e2_r1_gate'):
            constraints.append('gate')
        if config.get('reuse_gate_requirement'):
            constraints.append('historical-reuse')
        if config.get('descriptors', {}).get('require_frozen_models'):
            constraints.append('frozen-models')
        rows.append({'config': path.name, 'experiment': config.get('experiment', ''),
                     'external_inputs': ' | '.join(external), 'constraints': ' | '.join(constraints),
                     'checkout_status': 'external-inputs-required' if external else 'no-declared-external-inputs'})
    with (ROOT / 'docs/config_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(name.split('/')[0] for name in files)
    total = sum((ROOT / name).stat().st_size for name in files)
    body = '# 托管范围审计\n\n'
    body += '审计以 `.gitignore` 过滤后的 Git 文件清单为准，不发布被忽略的历史数据。\n\n'
    body += '## 保留与排除\n\n'
    body += '| 内容 | 处理 | 原因 |\n| --- | --- | --- |\n'
    body += '| `src/`、`configs/`、实验分析脚本、`tests/` | 保留 | 模型、仿真、配置继承和验证相互复用 |\n'
    body += '| README、依赖声明、复现文档、CI | 保留 | 新机器安装、验证和启动实验 |\n'
    body += '| `.venv*`、助手/编辑器配置、缓存 | 忽略 | 本机环境，可重新建立 |\n'
    body += '| `results/`、`models/`、数据、日志、图表 | 忽略 | 运行产物或外部输入，不适合普通 Git |\n'
    body += '| `revision_round2/`、`archive/`、内部讨论文档 | 忽略 | 审稿材料、论文副本、本机路径和历史笔记 |\n'
    body += '| 论文整理、Windows EMF 导出脚本 | 忽略 | 与实验复现无关且依赖本机论文环境 |\n\n'
    body += '排除均为 Git 忽略规则，未删除本地数据。没有选择许可证，也没有改变远端可见性。\n\n'
    body += '## 文件与配置清单\n\n'
    body += f'候选清单：{len(files)} 个文件，{total / 1024 / 1024:.2f} MiB（工作区文本大小）。\n\n'
    body += '| 目录/文件 | 文件数 |\n| --- | ---: |\n'
    body += ''.join(f'| `{name}` | {count} |\n' for name, count in sorted(counts.items()))
    body += '\n配置与外部输入清单见 [config_inventory.csv](config_inventory.csv)。'
    body += f' 共 {len(rows)} 个入口，其中 {sum(bool(row["external_inputs"]) for row in rows)} 个声明了未随 Git 发布的输入。\n\n'
    body += '`no-declared-external-inputs` 只表示未发现显式文件依赖，不表示实验已通过科学 gate。历史配置的 frozen 哈希和 reuse 约束保留。\n\n'
    body += '## 重新审计\n\n```powershell\npython scripts/audit_release.py\ngit status --short\ngit diff --cached --stat\n```\n\n'
    body += '脚本解析 Python 语法、拒绝大于 10 MiB 的候选文件、检查常见凭据特征，追踪配置继承及 guidance/scenario 输入。这不替代人工审查。\n'
    (ROOT / 'docs/repository_audit.md').write_text(body, encoding='utf-8')
    print(f'Audited {len(files)} files, {total / 1024 / 1024:.2f} MiB; {len(rows)} configs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
