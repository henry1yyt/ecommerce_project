"""项目交付前的快速完整性检查。"""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
errors = []
checks = []

required_files = [
    ROOT / "data/raw_orders_demo.csv",
    ROOT / "data/ecommerce_orders_ai_extended_dirty.csv",
    ROOT / "data/clean_orders.csv",
    ROOT / "outputs/tables/model_evaluation.csv",
    ROOT / "outputs/tables/risk_profile_test.csv",
    ROOT / "outputs/tables/rfm_k_selection.csv",
    ROOT / "outputs/tables/dashboard_data.json",
    ROOT / "outputs/models/return_risk_model.joblib",
    ROOT / "outputs/models/rfm_kmeans.joblib",
    ROOT / "report/电商大促运营分析与退货风险预测.html",
]
required_files += [ROOT / f"outputs/figures/{i:02d}_{name}.png" for i, name in [
    (1, "monthly_gmv_trend"),
    (2, "category_gmv_return"),
    (3, "channel_roi"),
    (4, "cohort_retention"),
    (5, "kmeans_selection"),
    (6, "rfm_segments"),
    (7, "model_roc"),
    (8, "risk_profile_test"),
]]

for path in required_files:
    ok = path.exists() and path.stat().st_size > 0
    checks.append((str(path.relative_to(ROOT)), ok))
    if not ok:
        errors.append(f"缺少或为空：{path.relative_to(ROOT)}")

try:
    clean = pd.read_csv(ROOT / "data/clean_orders.csv")
    if clean["order_id"].duplicated().any():
        errors.append("clean_orders.csv 仍存在重复订单号")
    key_cols = ["order_id", "user_id", "order_date", "price", "quantity", "is_returned"]
    if clean[key_cols].isna().any().any():
        errors.append("clean_orders.csv 核心字段仍有缺失值")
    if len(clean) < 5000:
        errors.append(f"清洗后订单数过少：{len(clean)}")
except Exception as exc:
    errors.append(f"读取 clean_orders.csv 失败：{exc}")

try:
    evaluation = pd.read_csv(ROOT / "outputs/tables/model_evaluation.csv")
    best_auc = float(evaluation["测试集AUC"].max())
    if best_auc <= 0.60:
        errors.append(f"测试集 AUC 偏低：{best_auc:.3f}")
except Exception as exc:
    errors.append(f"读取模型评估失败：{exc}")

try:
    risk = pd.read_csv(ROOT / "outputs/tables/risk_profile_test.csv")
    rates = dict(zip(risk["risk_level"].astype(str), risk["实际退货率"]))
    if rates.get("高风险", 0) <= rates.get("低风险", 1):
        errors.append("测试集高风险层退货率未高于低风险层")
except Exception as exc:
    errors.append(f"读取风险分层失败：{exc}")

try:
    payload = json.loads((ROOT / "outputs/tables/dashboard_data.json").read_text(encoding="utf-8"))
    if not payload.get("conclusions"):
        errors.append("dashboard_data.json 缺少结论")
except Exception as exc:
    errors.append(f"读取 dashboard_data.json 失败：{exc}")

print("项目文件检查：")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

if errors:
    print("\n检查未通过：")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("\n全部检查通过，可以提交。")
