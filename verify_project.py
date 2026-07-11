"""验证 Online Retail II 项目输出及旧模拟口径是否已清除。"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
required = ["cleaning_audit.csv", "monthly_trend.csv", "country_analysis.csv", "product_analysis.csv",
            "cohort_retention.csv", "rfm_k_selection.csv", "rfm_cluster_summary.csv",
            "rfm_user_segments.csv", "dashboard_data.json"]
missing = [p for p in required if not (ROOT / "outputs/tables" / p).exists()]
if missing:
    raise SystemExit(f"缺少输出：{missing}")
d = json.loads((ROOT / "outputs/tables/dashboard_data.json").read_text(encoding="utf-8"))
assert d["source"]["platform"] == "和鲸社区"
assert d["kpis"]["orders"] > 30000 and d["kpis"]["customers"] > 5000
assert len(d["k_selection"]) == 6 and d["kpis"]["best_k"] == 4
assert len(d["segments"]) == 4
html = (ROOT / "report/在线零售用户与经营洞察.html").read_text(encoding="utf-8")
for forbidden in ["AI 扩展数据", "渠道 ROI", "A/B 营销方案", "退货风险模型"]:
    assert forbidden not in html, f"仍包含旧口径：{forbidden}"
assert "outputs/figures" not in html and "<img" not in html
audit = pd.read_csv(ROOT / "outputs/tables/cleaning_audit.csv")
assert int(audit.loc[audit["检查项"] == "原始行数", "数量"].iloc[0]) == 1067371
print("验证通过：数据源、输出、四类 RFM 分群和无图片前端均符合要求。")
