#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python analysis.py
python verify_project.py
printf '\n完成：请打开 report/电商大促运营分析与退货风险预测.html\n'
