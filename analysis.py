"""电商大促运营分析与退货风险预测：一键复现脚本。

执行流程：
1. 使用课程 demo 作为种子，按业务规则扩展模拟订单；
2. 注入并清洗脏数据，生成审计表；
3. 完成经营分析、A/B 测试、同期群、RFM + KMeans；
4. 训练并评估退货风险模型，严格使用留出测试集验证；
5. 输出 CSV、模型、PNG 图表与离线 HTML 报告。

说明：扩展数据均为教学模拟数据，不代表真实企业经营情况。
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
N_EXPANDED = 6000
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "report"
NOTEBOOK_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"

for directory in [DATA_DIR, TABLE_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR, NOTEBOOK_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def records(frame: pd.DataFrame, digits: int = 4) -> list[dict[str, Any]]:
    result = frame.copy()
    for col in result.select_dtypes(include="number").columns:
        result[col] = result[col].round(digits)
    return result.replace({np.nan: None}).to_dict("records")


def save_csv(frame: pd.DataFrame, filename: str, index: bool = False) -> None:
    frame.to_csv(TABLE_DIR / filename, index=index, encoding="utf-8-sig")


def build_extended_data(seed: pd.DataFrame, n: int = N_EXPANDED) -> pd.DataFrame:
    """按业务逻辑扩展 demo 数据，而不是简单复制标签。"""
    base = seed.sample(n=n, replace=True, random_state=SEED).reset_index(drop=True).copy()
    base["order_id"] = [f"AI{i:08d}" for i in range(1, n + 1)]

    user_pool = np.array([f"U{i:05d}" for i in range(1, 1801)])
    user_weights = 1 / np.power(np.arange(1, len(user_pool) + 1), 0.42)
    user_weights /= user_weights.sum()
    base["user_id"] = RNG.choice(user_pool, n, p=user_weights)

    # 订单日期覆盖 18 个月，618、双十一和周末提高订单概率。
    dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    date_weights = np.ones(len(dates), dtype=float)
    month_day = dates.strftime("%m-%d")
    date_weights[(month_day >= "06-01") & (month_day <= "06-20")] *= 3.0
    date_weights[(month_day >= "11-01") & (month_day <= "11-12")] *= 2.4
    date_weights[dates.weekday >= 5] *= 1.15
    date_weights /= date_weights.sum()
    base["order_date"] = pd.to_datetime(RNG.choice(dates, n, p=date_weights))
    md = base["order_date"].dt.strftime("%m-%d")
    base["campaign_name"] = np.select(
        [(md >= "06-01") & (md <= "06-20"), (md >= "11-01") & (md <= "11-12")],
        ["618大促", "双十一"],
        default="日常",
    )

    # 同一用户属性保持一致。
    profiles = pd.DataFrame({"user_id": user_pool})
    profiles["user_gender"] = RNG.choice(["男", "女"], len(profiles), p=[0.48, 0.52])
    profiles["user_age"] = np.clip(np.rint(RNG.normal(34, 10, len(profiles))), 18, 65).astype(int)
    profiles["user_level"] = RNG.choice(
        ["新客", "普通会员", "学生会员", "银卡会员", "金卡会员"],
        len(profiles),
        p=[0.12, 0.39, 0.15, 0.23, 0.11],
    )
    for column in ["user_gender", "user_age", "user_level"]:
        base = base.drop(columns=[column], errors="ignore").merge(
            profiles[["user_id", column]], on="user_id", how="left"
        )
    base["age_group"] = pd.cut(
        base["user_age"],
        [17, 22, 30, 40, 50, 65],
        labels=["18-22", "23-30", "31-40", "41-50", "51-65"],
    ).astype(str)

    channel_map = {"线上商城": "网页商城", "手机App": "手机APP"}
    base["sales_channel"] = base["sales_channel"].replace(channel_map)

    level_multiplier = base["user_level"].map(
        {"新客": 0.92, "普通会员": 0.96, "学生会员": 0.88, "银卡会员": 1.06, "金卡会员": 1.18}
    ).astype(float)
    base["price"] = np.maximum(
        5, base["price"].astype(float) * RNG.lognormal(0, 0.16, n) * level_multiplier
    ).round(2)
    base["cost"] = (base["price"] * RNG.uniform(0.43, 0.71, n)).round(2)

    campaign_discount = base["campaign_name"].map({"日常": 0.96, "618大促": 0.82, "双十一": 0.80}).astype(float)
    base["discount"] = np.clip(campaign_discount + RNG.normal(0, 0.05, n), 0.65, 1.0).round(2)

    # 用户级稳定分组，避免同一用户同时进入 A/B 两组。
    base["ab_group"] = np.where(base["user_id"].str[-1].astype(int) % 2 == 0, "A", "B")
    quantity_lambda = (
        0.75
        + (base["campaign_name"] != "日常").astype(float) * 0.45
        + (base["ab_group"] == "B").astype(float) * 0.20
        + (1 - base["discount"]) * 1.2
    )
    base["quantity"] = np.clip(1 + RNG.poisson(quantity_lambda), 1, 8).astype(int)

    coupon_raw_a = RNG.choice([0, 5, 10, 20], n, p=[0.62, 0.16, 0.16, 0.06])
    coupon_raw_b = RNG.choice([0, 10, 20, 30], n, p=[0.34, 0.29, 0.24, 0.13])
    coupon_raw = np.where(base["ab_group"].eq("B"), coupon_raw_b, coupon_raw_a)
    gross_before_coupon = base["price"] * base["quantity"] * base["discount"]
    base["coupon_amount"] = np.minimum(coupon_raw, gross_before_coupon * 0.40).round(2)

    # 渠道成本结构：直播间和第三方平台获客成本更高。
    channel = base["sales_channel"]
    base["shipping_fee"] = np.where(
        channel.eq("线下门店"),
        0,
        RNG.choice([0, 6, 8, 10, 12], n, p=[0.10, 0.22, 0.31, 0.25, 0.12]),
    ).astype(float)
    base["marketing_cost"] = np.select(
        [
            channel.eq("直播间"),
            channel.eq("第三方平台"),
            channel.eq("手机APP"),
            channel.eq("网页商城"),
        ],
        [
            RNG.uniform(45, 95, n),
            RNG.uniform(25, 55, n),
            RNG.uniform(20, 45, n),
            RNG.uniform(15, 35, n),
        ],
        default=RNG.uniform(8, 20, n),
    ).round(2)
    base["delivery_days"] = np.where(
        channel.eq("线下门店"), 0, np.clip(RNG.poisson(2.2, n) + 1, 1, 8)
    ).astype(int)

    # 退货概率由品类、折扣、渠道、件数、活动和会员等级共同决定。
    category_effect = base["category"].map(
        {
            "服饰运动": 1.35,
            "美妆个护": 0.88,
            "数码配件": 0.45,
            "电脑办公": 0.32,
            "生活用品": 0.02,
            "图书文具": -0.15,
            "食品饮料": -0.40,
        }
    ).fillna(0.05)
    level_protection = base["user_level"].map(
        {"新客": 0.0, "普通会员": 0.08, "学生会员": 0.04, "银卡会员": 0.22, "金卡会员": 0.40}
    ).astype(float)
    logit = (
        -3.55
        + category_effect
        + (1 - base["discount"]) * 5.0
        + channel.eq("直播间").astype(float) * 1.00
        + channel.eq("第三方平台").astype(float) * 0.42
        + (base["quantity"] >= 5).astype(float) * 0.70
        + (base["campaign_name"] != "日常").astype(float) * 0.65
        - level_protection
    )
    return_probability = 1 / (1 + np.exp(-logit))
    base["is_returned"] = RNG.binomial(1, return_probability).astype(int)
    return_reasons = RNG.choice(
        ["尺码不合适", "商品与描述不符", "质量问题", "不想要了", "物流破损"],
        n,
        p=[0.28, 0.24, 0.20, 0.18, 0.10],
    )
    base["return_reason"] = np.where(base["is_returned"].eq(1), return_reasons, "未退货")

    rating_raw = (
        4.58
        - base["is_returned"] * 1.62
        - (base["delivery_days"] >= 5).astype(float) * 0.42
        + RNG.normal(0, 0.52, n)
    )
    base["customer_rating"] = np.clip(np.rint(rating_raw), 1, 5).astype(int)

    hour_weights = np.array([2, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 10, 7, 2], dtype=float)
    hour_weights /= hour_weights.sum()
    base["order_hour"] = RNG.choice(np.arange(8, 24), n, p=hour_weights)

    # 注入少量脏数据，用于展示课程中的数据清洗流程。
    dirty = pd.concat([base, base.iloc[:18].copy()], ignore_index=True)
    dirty.loc[RNG.choice(len(dirty), 25, replace=False), "customer_rating"] = np.nan
    dirty.loc[RNG.choice(len(dirty), 15, replace=False), "city"] = np.nan
    dirty.loc[RNG.choice(len(dirty), 8, replace=False), "price"] = -1
    dirty.loc[RNG.choice(len(dirty), 5, replace=False), "discount"] = 1.25
    dirty.loc[RNG.choice(len(dirty), 3, replace=False), "order_date"] = None
    return dirty


def clean_orders(dirty: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """清洗数据并返回清洗结果与审计表。"""
    audit_rows = [
        {"检查项": "扩展后总行数（含脏数据）", "数量": len(dirty)},
        {"检查项": "重复订单号", "数量": int(dirty.duplicated("order_id").sum())},
        {"检查项": "城市缺失", "数量": int(dirty["city"].isna().sum())},
        {"检查项": "评分缺失", "数量": int(dirty["customer_rating"].isna().sum())},
        {"检查项": "价格异常（<=0）", "数量": int((pd.to_numeric(dirty["price"], errors="coerce") <= 0).sum())},
        {"检查项": "折扣异常（不在0至1）", "数量": int((~pd.to_numeric(dirty["discount"], errors="coerce").between(0, 1)).sum())},
        {"检查项": "日期缺失或不可解析", "数量": int(pd.to_datetime(dirty["order_date"], errors="coerce").isna().sum())},
    ]

    df = dirty.drop_duplicates("order_id").copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    numeric_columns = [
        "price",
        "cost",
        "quantity",
        "discount",
        "coupon_amount",
        "shipping_fee",
        "marketing_cost",
        "customer_rating",
        "is_returned",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["city"] = df["city"].fillna("未知城市")
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
    df["return_reason"] = df["return_reason"].fillna("未退货")
    required = ["order_date", "price", "cost", "quantity", "discount", "is_returned"]
    df = df.dropna(subset=required)
    valid_mask = (
        (df["price"] > 0)
        & (df["cost"] >= 0)
        & (df["quantity"] > 0)
        & df["discount"].between(0, 1)
        & df["customer_rating"].between(1, 5)
        & df["is_returned"].isin([0, 1])
    )
    df = df.loc[valid_mask].copy()
    df["quantity"] = df["quantity"].astype(int)
    df["is_returned"] = df["is_returned"].astype(int)
    df["customer_rating"] = df["customer_rating"].round(1)

    # 派生业务指标。
    df["sales_amount"] = (
        df["price"] * df["quantity"] * df["discount"] - df["coupon_amount"]
    ).clip(lower=0).round(2)
    df["refund_amount"] = (df["sales_amount"] * df["is_returned"]).round(2)
    df["net_sales_amount"] = (df["sales_amount"] - df["refund_amount"]).round(2)
    df["product_cost"] = (df["cost"] * df["quantity"]).round(2)
    df["gross_profit"] = (df["net_sales_amount"] - df["product_cost"]).round(2)
    df["reverse_logistics_cost"] = (df["is_returned"] * 8.0).round(2)
    df["net_profit"] = (
        df["gross_profit"]
        - df["shipping_fee"]
        - df["marketing_cost"]
        - df["reverse_logistics_cost"]
    ).round(2)
    df["gross_margin"] = np.where(
        df["net_sales_amount"] > 0, df["gross_profit"] / df["net_sales_amount"], np.nan
    )
    df["net_margin"] = np.where(
        df["net_sales_amount"] > 0, df["net_profit"] / df["net_sales_amount"], np.nan
    )
    df["is_weekend"] = (df["order_date"].dt.weekday >= 5).astype(int)
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    audit_rows.extend(
        [
            {"检查项": "去重后的订单数", "数量": int(dirty["order_id"].nunique())},
            {"检查项": "最终清洗后订单数", "数量": len(df)},
            {"检查项": "清洗过程中剔除的唯一订单", "数量": int(dirty["order_id"].nunique() - len(df))},
        ]
    )
    return df.reset_index(drop=True), pd.DataFrame(audit_rows)


def grouped_analysis(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    grouped = (
        df.groupby(dimension)
        .agg(
            订单量=("order_id", "nunique"),
            用户数=("user_id", "nunique"),
            GMV=("sales_amount", "sum"),
            净销售额=("net_sales_amount", "sum"),
            净利润=("net_profit", "sum"),
            退货率=("is_returned", "mean"),
            平均评分=("customer_rating", "mean"),
            营销成本=("marketing_cost", "sum"),
        )
        .reset_index()
    )
    grouped["客单价"] = grouped["GMV"] / grouped["订单量"].replace(0, np.nan)
    grouped["利润率"] = grouped["净利润"] / grouped["净销售额"].replace(0, np.nan)
    grouped["ROI"] = grouped["净利润"] / grouped["营销成本"].replace(0, np.nan)
    return grouped.sort_values("GMV", ascending=False).reset_index(drop=True)


def ab_test(df: pd.DataFrame) -> pd.DataFrame:
    """以用户为实验单位，对用户总消费进行 99% Winsorize 后做 Welch t 检验。"""
    user_metric = (
        df.groupby(["user_id", "ab_group"])
        .agg(总消费=("sales_amount", "sum"), 订单数=("order_id", "nunique"))
        .reset_index()
    )
    upper = user_metric["总消费"].quantile(0.99)
    user_metric["检验消费"] = user_metric["总消费"].clip(upper=upper)
    a = user_metric.loc[user_metric["ab_group"].eq("A"), "检验消费"]
    b = user_metric.loc[user_metric["ab_group"].eq("B"), "检验消费"]
    t_stat, p_value = ttest_ind(a, b, equal_var=False)

    difference = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    numerator = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
    denominator = (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    degrees_freedom = numerator / denominator
    critical = student_t.ppf(0.975, degrees_freedom)
    ci_low, ci_high = difference - critical * se, difference + critical * se
    pooled_sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    cohen_d = difference / pooled_sd if pooled_sd else np.nan

    return pd.DataFrame(
        [
            {
                "检验指标": "用户总消费（99% Winsorize）",
                "A样本量": len(a),
                "B样本量": len(b),
                "A人均消费": a.mean(),
                "B人均消费": b.mean(),
                "绝对差异": difference,
                "提升率": b.mean() / a.mean() - 1,
                "差异95%CI下限": ci_low,
                "差异95%CI上限": ci_high,
                "Cohen_d": cohen_d,
                "t统计量": t_stat,
                "p值": p_value,
                "显著": bool(p_value < 0.05),
            }
        ]
    )


def cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    cohort = df[["user_id", "order_date"]].copy()
    cohort["order_month"] = cohort["order_date"].dt.to_period("M")
    cohort["cohort_month"] = cohort.groupby("user_id")["order_month"].transform("min")
    cohort["cohort_index"] = (
        (cohort["order_month"].dt.year - cohort["cohort_month"].dt.year) * 12
        + cohort["order_month"].dt.month
        - cohort["cohort_month"].dt.month
    )
    count_table = cohort.groupby(["cohort_month", "cohort_index"])["user_id"].nunique().unstack()
    retention = count_table.div(count_table[0], axis=0)
    retention.index = retention.index.astype(str)
    retention.columns = retention.columns.astype(int)
    return retention


def rfm_kmeans(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("user_id")
        .agg(
            R=("order_date", lambda values: (snapshot - values.max()).days),
            F=("order_id", "nunique"),
            M=("sales_amount", "sum"),
        )
        .reset_index()
    )
    transformed = np.log1p(rfm[["R", "F", "M"]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)

    selection_rows: list[dict[str, Any]] = []
    for k in range(2, 8):
        candidate = KMeans(n_clusters=k, random_state=SEED, n_init=12)
        labels = candidate.fit_predict(scaled)
        selection_rows.append(
            {"K": k, "惯性": candidate.inertia_, "轮廓系数": silhouette_score(scaled, labels, sample_size=min(500, len(scaled)), random_state=SEED)}
        )
    selection = pd.DataFrame(selection_rows)

    # 选择 4 群：兼顾轮廓系数和业务可解释性，便于形成四类运营策略。
    selected_k = 4
    model = KMeans(n_clusters=selected_k, random_state=SEED, n_init=15)
    rfm["cluster"] = model.fit_predict(scaled)
    summary = (
        rfm.groupby("cluster")
        .agg(用户数=("user_id", "count"), R=("R", "mean"), F=("F", "mean"), M=("M", "mean"))
        .reset_index()
    )

    standardized_summary = summary.copy()
    for column in ["R", "F", "M"]:
        std = standardized_summary[column].std(ddof=0)
        standardized_summary[f"z_{column}"] = (
            standardized_summary[column] - standardized_summary[column].mean()
        ) / (std if std else 1)
    standardized_summary["价值分"] = -standardized_summary["z_R"] + standardized_summary["z_F"] + standardized_summary["z_M"]

    dormant_cluster = int(summary.sort_values("R", ascending=False).iloc[0]["cluster"])
    remaining = standardized_summary.loc[standardized_summary["cluster"] != dormant_cluster].sort_values("价值分")
    name_map = {
        dormant_cluster: "沉睡用户",
        int(remaining.iloc[0]["cluster"]): "低价值用户",
        int(remaining.iloc[1]["cluster"]): "潜力用户",
        int(remaining.iloc[2]["cluster"]): "高价值用户",
    }
    rfm["用户分群"] = rfm["cluster"].map(name_map)
    summary["用户分群"] = summary["cluster"].map(name_map)
    summary = summary.merge(
        standardized_summary[["cluster", "价值分"]], on="cluster", how="left"
    ).sort_values("价值分", ascending=False)

    bundle = {
        "model": model,
        "scaler": scaler,
        "labels": name_map,
        "selected_k": selected_k,
        "feature_transform": "log1p + StandardScaler",
    }
    return rfm, summary.reset_index(drop=True), selection, bundle


def build_model_pipeline(model: Any, categorical: list[str], numerical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numerical),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.02):
        prediction = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "阈值": threshold,
                "Precision": precision_score(y_true, prediction, zero_division=0),
                "Recall": recall_score(y_true, prediction, zero_division=0),
                "F1": f1_score(y_true, prediction, zero_division=0),
                "F2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    # 退货预警更重视减少漏判，因此采用 F2 最大的阈值。
    best_threshold = float(table.sort_values(["F2", "F1"], ascending=False).iloc[0]["阈值"])
    return best_threshold, table


def train_return_model(df: pd.DataFrame) -> dict[str, Any]:
    features = [
        "category",
        "region",
        "sales_channel",
        "campaign_name",
        "user_level",
        "traffic_source",
        "device_type",
        "price",
        "quantity",
        "discount",
        "coupon_amount",
        "marketing_cost",
        "shipping_fee",
        "order_hour",
        "is_weekend",
    ]
    categorical = [column for column in features if df[column].dtype == "object"]
    numerical = [column for column in features if column not in categorical]
    X = df[features].copy()
    y = df["is_returned"].astype(int)

    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=y_train_val,
    )

    candidates = {
        "逻辑回归": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        "随机森林": RandomForestClassifier(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    evaluation_rows = []
    fitted_models: dict[str, Pipeline] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        pipeline = build_model_pipeline(estimator, categorical, numerical)
        pipeline.fit(X_train, y_train)
        val_probability = pipeline.predict_proba(X_val)[:, 1]
        threshold, threshold_table = choose_threshold(y_val, val_probability)
        test_probability = pipeline.predict_proba(X_test)[:, 1]
        test_prediction = (test_probability >= threshold).astype(int)
        matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        evaluation_rows.append(
            {
                "模型": name,
                "验证集AUC": roc_auc_score(y_val, val_probability),
                "测试集AUC": roc_auc_score(y_test, test_probability),
                "测试集PR_AUC": average_precision_score(y_test, test_probability),
                "阈值": threshold,
                "Precision": precision_score(y_test, test_prediction, zero_division=0),
                "Recall": recall_score(y_test, test_prediction, zero_division=0),
                "F1": f1_score(y_test, test_prediction, zero_division=0),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )
        fitted_models[name] = pipeline
        threshold_tables[name] = threshold_table
        validation_probabilities[name] = val_probability
        test_probabilities[name] = test_probability

    evaluation = pd.DataFrame(evaluation_rows).sort_values(
        ["验证集AUC", "测试集PR_AUC"], ascending=False
    ).reset_index(drop=True)
    best_name = str(evaluation.iloc[0]["模型"])
    best_pipeline = fitted_models[best_name]
    best_threshold = float(evaluation.iloc[0]["阈值"])
    best_test_probability = test_probabilities[best_name]
    best_val_probability = validation_probabilities[best_name]

    # 风险等级阈值仅由验证集概率确定，再用于独立测试集。
    risk_q1, risk_q2 = np.quantile(best_val_probability, [1 / 3, 2 / 3])
    if math.isclose(risk_q1, risk_q2):
        risk_q2 = min(1.0, risk_q1 + 1e-6)

    def risk_level(probability: np.ndarray) -> pd.Categorical:
        return pd.cut(
            probability,
            bins=[-np.inf, risk_q1, risk_q2, np.inf],
            labels=["低风险", "中风险", "高风险"],
            ordered=True,
        )

    test_scored = df.loc[idx_test, ["order_id", "is_returned"]].copy()
    test_scored["return_risk_score"] = best_test_probability
    test_scored["risk_level"] = risk_level(best_test_probability)
    risk_profile = (
        test_scored.groupby("risk_level", observed=True)
        .agg(
            订单数=("order_id", "count"),
            退货数=("is_returned", "sum"),
            实际退货率=("is_returned", "mean"),
            平均风险分=("return_risk_score", "mean"),
        )
        .reset_index()
    )

    # 留出集评估完成后，用训练集+验证集重新拟合最终模型，用于全量订单评分与交付。
    final_model = build_model_pipeline(candidates[best_name], categorical, numerical)
    final_model.fit(X_train_val, y_train_val)
    full_probability = final_model.predict_proba(X)[:, 1]
    scored_orders = df.copy()
    scored_orders["return_risk_score"] = full_probability
    scored_orders["risk_level"] = risk_level(full_probability)

    # 提取特征重要性，增强模型解释性。
    feature_names = final_model.named_steps["prep"].get_feature_names_out()
    fitted_estimator = final_model.named_steps["model"]
    if hasattr(fitted_estimator, "coef_"):
        importance_values = np.abs(fitted_estimator.coef_[0])
        signed_values = fitted_estimator.coef_[0]
    else:
        importance_values = fitted_estimator.feature_importances_
        signed_values = fitted_estimator.feature_importances_
    feature_importance = pd.DataFrame(
        {"特征": feature_names, "重要性": importance_values, "方向值": signed_values}
    ).sort_values("重要性", ascending=False).head(25)

    fpr, tpr, _ = roc_curve(y_test, best_test_probability)
    return {
        "features": features,
        "evaluation": evaluation,
        "best_name": best_name,
        "best_threshold": best_threshold,
        "threshold_table": threshold_tables[best_name],
        "risk_profile": risk_profile,
        "scored_orders": scored_orders,
        "feature_importance": feature_importance,
        "model_bundle": {
            "model": final_model,
            "classification_threshold": best_threshold,
            "risk_quantiles": [float(risk_q1), float(risk_q2)],
            "features": features,
            "best_model_name": best_name,
            "random_seed": SEED,
        },
        "roc": {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, best_test_probability)},
        "test_base_rate": float(y_test.mean()),
    }


def create_figures(
    monthly: pd.DataFrame,
    category: pd.DataFrame,
    channel: pd.DataFrame,
    retention: pd.DataFrame,
    rfm: pd.DataFrame,
    rfm_summary: pd.DataFrame,
    k_selection: pd.DataFrame,
    model_result: dict[str, Any],
) -> None:
    plt.figure(figsize=(11, 5.5))
    plt.plot(monthly["month"], monthly["GMV"] / 10000, marker="o")
    plt.title("月度 GMV 趋势")
    plt.xlabel("月份")
    plt.ylabel("GMV（万元）")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_monthly_gmv_trend.png", dpi=180)
    plt.close()

    ordered_category = category.sort_values("GMV", ascending=False)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(ordered_category["category"], ordered_category["GMV"] / 10000)
    ax1.set_ylabel("GMV（万元）")
    ax1.set_xlabel("品类")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(ordered_category["category"], ordered_category["退货率"] * 100, marker="o")
    ax2.set_ylabel("退货率（%）")
    plt.title("品类 GMV 与退货率")
    fig.tight_layout()
    plt.savefig(FIGURE_DIR / "02_category_gmv_return.png", dpi=180)
    plt.close(fig)

    ordered_channel = channel.sort_values("ROI", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(ordered_channel["sales_channel"], ordered_channel["ROI"])
    plt.title("渠道 ROI 对比（净利润 / 营销成本）")
    plt.xlabel("渠道")
    plt.ylabel("ROI")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_channel_roi.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 7))
    values = retention.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = plt.imshow(masked, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(image, label="留存率")
    plt.title("同期群留存热力图")
    plt.xlabel("距首次消费月数")
    plt.ylabel("首次消费月份")
    plt.xticks(range(len(retention.columns)), retention.columns)
    plt.yticks(range(len(retention.index)), retention.index)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "04_cohort_retention.png", dpi=180)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(k_selection["K"], k_selection["轮廓系数"], marker="o")
    ax1.set_xlabel("聚类数量 K")
    ax1.set_ylabel("轮廓系数")
    ax2 = ax1.twinx()
    ax2.plot(k_selection["K"], k_selection["惯性"], marker="s")
    ax2.set_ylabel("惯性")
    plt.title("KMeans 聚类数量选择")
    fig.tight_layout()
    plt.savefig(FIGURE_DIR / "05_kmeans_selection.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(10, 6))
    for segment in rfm_summary["用户分群"]:
        subset = rfm.loc[rfm["用户分群"].eq(segment)]
        plt.scatter(
            subset["R"],
            subset["M"],
            s=np.clip(subset["F"] * 12, 20, 180),
            alpha=0.55,
            label=segment,
        )
    plt.yscale("log")
    plt.title("RFM 用户分群（点大小代表购买频次）")
    plt.xlabel("R：距最近一次消费天数")
    plt.ylabel("M：累计消费金额（对数坐标）")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06_rfm_segments.png", dpi=180)
    plt.close()

    roc_data = model_result["roc"]
    plt.figure(figsize=(7, 6))
    plt.plot(roc_data["fpr"], roc_data["tpr"], label=f"AUC = {roc_data['auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"退货风险模型 ROC：{model_result['best_name']}")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_model_roc.png", dpi=180)
    plt.close()

    risk = model_result["risk_profile"].copy()
    plt.figure(figsize=(8, 5))
    plt.bar(risk["risk_level"].astype(str), risk["实际退货率"] * 100)
    plt.title("独立测试集风险分层校验")
    plt.xlabel("风险等级")
    plt.ylabel("实际退货率（%）")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "08_risk_profile_test.png", dpi=180)
    plt.close()


def create_html_report(payload: dict[str, Any]) -> None:
    data_json = json.dumps(payload, ensure_ascii=False, default=_json_default)
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>电商大促运营分析与退货风险预测</title>
<style>
:root{--ink:#14212a;--muted:#63707b;--paper:#f4f1e9;--card:#fff;--accent:#136f68;--warm:#e7664c;--gold:#d39b32;--line:#ded9cf}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",sans-serif}.hero{padding:58px max(5vw,24px) 48px;background:#102f31;color:#fff;position:relative;overflow:hidden}.hero:after{content:"";position:absolute;right:-130px;top:-230px;width:520px;height:520px;border:92px solid #1a5050;border-radius:50%}.hero>*{position:relative;z-index:1}.eyebrow{font-size:12px;letter-spacing:3px;color:#a7d2ca}.hero h1{font-size:clamp(34px,5vw,68px);max-width:980px;line-height:1.08;margin:14px 0}.hero p{max-width:820px;color:#d2e0de;line-height:1.8}.wrap{max-width:1280px;margin:auto;padding:24px 22px 70px}.nav{position:sticky;top:0;z-index:8;display:flex;gap:8px;flex-wrap:wrap;padding:12px 0;background:rgba(244,241,233,.94);backdrop-filter:blur(9px)}.nav button{border:1px solid var(--line);background:#fff;padding:9px 14px;border-radius:99px;cursor:pointer}.nav button.active{background:var(--ink);color:#fff}.panel{display:none}.panel.active{display:block}.head{margin:34px 0 17px}.head span{color:var(--warm);font-weight:700;letter-spacing:1px}.head h2{font-size:29px;margin:5px 0}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 28px #14212a0a}.kpi{padding:18px;border-top:3px solid var(--accent)}.kpi .label{font-size:13px;color:var(--muted)}.kpi .value{font-size:26px;font-weight:800;margin-top:8px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:15px}.card{grid-column:span 6;padding:20px}.card.full{grid-column:1/-1}.card.third{grid-column:span 4}.card img{width:100%;border-radius:9px;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted)}.story{background:#fff;border-left:4px solid var(--warm);padding:15px 19px;margin:10px 0;line-height:1.7}.pill{display:inline-block;border-radius:99px;padding:4px 9px;background:#e3f0ed;color:var(--accent);font-weight:700}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.step{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px;line-height:1.55}.step b{display:block;color:var(--warm);margin-bottom:6px}.note{color:var(--muted);font-size:13px;line-height:1.7}.heat{display:grid;gap:3px;overflow:auto}.heat div{min-width:54px;padding:8px 5px;text-align:center;border-radius:3px;font-size:11px}.footer{margin-top:40px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px;line-height:1.7}@media(max-width:880px){.kpis{grid-template-columns:repeat(2,1fr)}.card,.card.third{grid-column:1/-1}.flow{grid-template-columns:repeat(2,1fr)}}

/* 2026 visual system: cinematic analytics */
:root{--night:#071416;--night2:#0b2427;--mint:#59e0c5;--coral:#ff7b61;--ice:#dffcf5}
body{background:
radial-gradient(circle at 15% 12%,rgba(89,224,197,.10),transparent 28rem),
linear-gradient(180deg,#eef2ed 0%,#f7f4ed 45%,#edf2ee 100%);overflow-x:hidden}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.035;z-index:20;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.7'/%3E%3C/svg%3E")}
.hero{min-height:88vh;display:grid;align-content:center;padding-top:110px;padding-bottom:90px;background:
radial-gradient(circle at 78% 28%,rgba(89,224,197,.15),transparent 26%),
radial-gradient(circle at 18% 88%,rgba(255,123,97,.12),transparent 26%),
linear-gradient(135deg,var(--night),var(--night2));isolation:isolate}
.hero:after{width:50vw;height:50vw;right:-16vw;top:-30vw;border:1px solid rgba(143,255,231,.17);box-shadow:0 0 0 70px rgba(89,224,197,.025),0 0 0 140px rgba(89,224,197,.018);animation:orbit 16s linear infinite}
.hero-copy{max-width:1050px}.eyebrow{display:inline-flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid rgba(167,210,202,.24);border-radius:999px;background:rgba(255,255,255,.04);backdrop-filter:blur(12px)}
.eyebrow:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 18px var(--mint);animation:pulse 2s infinite}
.hero h1{font-size:clamp(42px,6.6vw,94px);letter-spacing:-.055em;line-height:.98;margin:24px 0;max-width:1050px}
.hero h1 em{font-style:normal;color:transparent;background:linear-gradient(100deg,#fff 5%,var(--mint) 52%,#f5d9ad);background-clip:text;-webkit-background-clip:text}
.hero p{font-size:clamp(15px,1.5vw,20px);max-width:760px}
.hero-metrics{display:flex;gap:12px;flex-wrap:wrap;margin-top:34px}.hero-chip{min-width:156px;padding:14px 17px;border:1px solid rgba(255,255,255,.12);border-radius:16px;background:rgba(255,255,255,.055);backdrop-filter:blur(14px);transform:translateY(20px);opacity:0;animation:rise .7s forwards}.hero-chip:nth-child(2){animation-delay:.12s}.hero-chip:nth-child(3){animation-delay:.24s}
.hero-chip b{display:block;font-size:21px;color:#fff}.hero-chip span{font-size:11px;color:#9fb7b5;letter-spacing:.08em}
.scroll-cue{position:absolute;right:max(5vw,24px);bottom:38px;writing-mode:vertical-rl;color:#8ca6a4;font-size:10px;letter-spacing:.25em}.scroll-cue:after{content:"";display:inline-block;width:1px;height:62px;margin-top:12px;background:linear-gradient(var(--mint),transparent);animation:scan 2s infinite}
.wrap{max-width:1380px}.nav{top:12px;margin-top:-28px;padding:8px;border:1px solid rgba(255,255,255,.7);border-radius:999px;width:max-content;max-width:100%;box-shadow:0 18px 50px rgba(20,33,42,.12);background:rgba(247,244,237,.78)}
.nav button{position:relative;border:0;background:transparent;transition:.35s cubic-bezier(.2,.8,.2,1)}.nav button:hover{transform:translateY(-2px)}.nav button.active{background:var(--night);box-shadow:0 7px 18px rgba(7,20,22,.24)}
.head{margin-top:64px}.head span{font-size:11px;letter-spacing:.18em}.head h2{font-size:clamp(28px,3vw,46px);letter-spacing:-.035em}
.kpis{gap:14px}.kpi,.card,.story,.step{position:relative;overflow:hidden;border-color:rgba(20,33,42,.09);box-shadow:0 18px 60px rgba(12,33,34,.07);transition:transform .5s cubic-bezier(.2,.8,.2,1),box-shadow .5s}
.kpi:before,.card:before{content:"";position:absolute;inset:0;background:radial-gradient(420px circle at var(--mx,50%) var(--my,0%),rgba(89,224,197,.14),transparent 42%);opacity:0;transition:.4s;pointer-events:none}.kpi:hover:before,.card:hover:before{opacity:1}.kpi:hover,.card:hover{transform:translateY(-6px);box-shadow:0 28px 75px rgba(12,33,34,.12)}
.kpi{padding:22px;border-top:0}.kpi:after{content:"";position:absolute;left:0;bottom:0;height:3px;width:100%;background:linear-gradient(90deg,var(--mint),var(--coral));transform:scaleX(0);transform-origin:left;transition:.7s}.kpi:hover:after{transform:scaleX(1)}.kpi .value{font-size:clamp(24px,2.5vw,38px);letter-spacing:-.04em}
.card{border-radius:22px;padding:25px;background:rgba(255,255,255,.76);backdrop-filter:blur(18px)}.card h3{font-size:19px}.card img{filter:saturate(.75) contrast(1.02);transition:.55s}.card:hover img{filter:saturate(1) contrast(1.04);transform:scale(1.012)}
table{font-variant-numeric:tabular-nums}tr{transition:.2s}tbody tr:hover{background:rgba(89,224,197,.08)}.pill{box-shadow:inset 0 0 0 1px rgba(19,111,104,.12)}
.story{border-left:0;border-radius:18px;padding:20px 24px}.story b{display:inline-grid;place-items:center;width:30px;height:30px;margin-right:8px;border-radius:50%;background:var(--night);color:var(--mint)}
.step{border-radius:18px;min-height:130px}.step b{font-size:12px;letter-spacing:.08em}.step:hover{transform:translateY(-5px) rotate(-.3deg)}
.panel.active{animation:panelIn .55s cubic-bezier(.2,.8,.2,1)}.reveal{opacity:0;transform:translateY(34px);transition:opacity .75s,transform .75s cubic-bezier(.2,.8,.2,1)}.reveal.visible{opacity:1;transform:none}
.cursor-glow{position:fixed;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(89,224,197,.10),transparent 68%);pointer-events:none;z-index:0;transform:translate(-50%,-50%);transition:opacity .2s}
@keyframes rise{to{opacity:1;transform:none}}@keyframes panelIn{from{opacity:0;transform:translateY(12px)}}@keyframes orbit{to{transform:rotate(360deg)}}@keyframes pulse{50%{opacity:.4;box-shadow:0 0 5px var(--mint)}}@keyframes scan{0%{transform:scaleY(0);transform-origin:top}50%{transform:scaleY(1);transform-origin:top}51%{transform-origin:bottom}100%{transform:scaleY(0);transform-origin:bottom}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}.reveal{opacity:1;transform:none}}@media(max-width:880px){.hero{min-height:78vh}.hero h1{letter-spacing:-.035em}.nav{width:100%;border-radius:22px}.scroll-cue{display:none}}


/* Native front-end charts + continuous story */
.panel{display:block!important;scroll-margin-top:112px;padding-top:1px}.panel+.panel{margin-top:86px;padding-top:36px;border-top:1px solid rgba(20,33,42,.09)}
.viz{position:relative;min-height:340px;width:100%;padding:8px 4px 0}.viz svg{display:block;width:100%;height:330px;overflow:visible}.viz .grid-line{stroke:rgba(20,33,42,.09);stroke-width:1}.viz .axis-label{fill:#6c777c;font-size:11px}.viz .value-label{fill:#17282a;font-size:11px;font-weight:700}.viz .series-line{fill:none;stroke:var(--accent);stroke-width:3;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 6px 8px rgba(19,111,104,.18));stroke-dasharray:1200;stroke-dashoffset:1200;animation:drawLine 1.8s .2s cubic-bezier(.2,.8,.2,1) forwards}.viz .series-area{fill:url(#areaGradient);opacity:0;animation:fadeArea 1s .8s forwards}.viz .dot{fill:#fff;stroke:var(--accent);stroke-width:3;transition:.25s;cursor:pointer}.viz .dot:hover{r:7;fill:var(--mint)}.viz .bar-shape{transform-box:fill-box;transform-origin:center bottom;animation:growBar .9s cubic-bezier(.2,.8,.2,1) both;transition:opacity .2s,filter .2s;cursor:pointer}.viz .bar-shape:hover{filter:brightness(1.08);opacity:.86}.viz .bubble{stroke:#fff;stroke-width:2;opacity:.78;transform-box:fill-box;transform-origin:center;animation:popBubble .7s cubic-bezier(.2,.8,.2,1) both;transition:.25s;cursor:pointer}.viz .bubble:hover{opacity:1;stroke-width:4;transform:scale(1.08)}
.chart-tooltip{position:fixed;z-index:40;padding:9px 11px;border-radius:10px;background:#071416;color:#fff;font-size:12px;line-height:1.5;pointer-events:none;opacity:0;transform:translate(-50%,-115%);transition:opacity .15s;box-shadow:0 12px 35px rgba(0,0,0,.2)}.chart-legend{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 8px;color:var(--muted);font-size:12px}.chart-legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.nav button.active{background:var(--night);color:#fff}.nav{overflow-x:auto;scrollbar-width:none}.nav::-webkit-scrollbar{display:none}
@keyframes drawLine{to{stroke-dashoffset:0}}@keyframes fadeArea{to{opacity:.18}}@keyframes growBar{from{transform:scaleY(0)}}@keyframes popBubble{from{transform:scale(0);opacity:0}}

</style>
</head>
<body>
<header class="hero"><div class="hero-copy"><div class="eyebrow">AI-EXTENDED DATA · INTERNSHIP CAPSTONE</div><h1>增长之后，<br><em>真正留下了什么？</em></h1><p>从课程 demo 的 AI 规则扩展，到数据清洗、经营复盘、同期群、RFM 聚类和退货风险预警。本报告所有指标均由同一条可复现分析链路生成。</p><div class="hero-metrics"><div class="hero-chip"><b id="heroGmv">—</b><span>GMV / 经营规模</span></div><div class="hero-chip"><b id="heroAuc">—</b><span>TEST AUC / 模型能力</span></div><div class="hero-chip"><b id="heroLift">—</b><span>RISK LIFT / 风险区分</span></div></div></div><div class="scroll-cue">SCROLL TO EXPLORE</div></header>
<main class="wrap"><nav class="nav" id="nav"></nav><div id="app"></div><div class="footer">数据声明：本项目以课程 demo 为种子，经 Python 按业务规则扩展为教学模拟数据，不代表真实企业经营情况。随机种子：42；模型风险分层使用独立测试集验证。</div></main>
<script>
const D=__DATA__;
const tabs=[['overview','经营总览'],['operation','运营复盘'],['customer','用户洞察'],['risk','风险模型'],['method','方法与结论']];
const nav=document.querySelector('#nav'),app=document.querySelector('#app');
const money=x=>'¥'+Number(x).toLocaleString('zh-CN',{maximumFractionDigits:0});
const pct=x=>(100*Number(x)).toFixed(1)+'%';
const num=x=>Number(x).toLocaleString('zh-CN',{maximumFractionDigits:3});
function fmt(c,v){if(v===null||v===undefined)return '-';if(c.includes('率')||c.includes('AUC')||['Precision','Recall','F1'].includes(c))return pct(v);if(['GMV','净利润','净销售额','营销成本','M','A人均消费','B人均消费'].includes(c))return money(v);return num(v)}
function table(rows,cols){return `<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map((c,i)=>`<td>${i?fmt(c,r[c]):r[c]}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
function panel(id,html){app.insertAdjacentHTML('beforeend',`<section class="panel" id="${id}">${html}</section>`)}
tabs.forEach(([id,label],i)=>nav.insertAdjacentHTML('beforeend',`<button data-id="${id}" class="${i?'':'active'}">${label}</button>`));
panel('overview',`<div class="head"><span>01 / BUSINESS PULSE</span><h2>核心经营指标</h2></div><div class="kpis">${[['GMV',money(D.kpi.GMV)],['净销售额',money(D.kpi.净销售额)],['净利润',money(D.kpi.净利润)],['退货率',pct(D.kpi.退货率)],['用户数',num(D.kpi.用户数)]].map(x=>`<div class="kpi"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('')}</div><div class="head"><h2>趋势与结构</h2></div><div class="grid"><div class="card"><h3>月度 GMV 趋势</h3><div class="chart-legend"><span><i style="background:#136f68"></i>GMV</span></div><div class="viz" id="monthlyChart"></div><p class="note">大促月份带来明显增长，但应同时观察净利润与退货率。</p></div><div class="card"><h3>品类 GMV 与退货率</h3><div class="chart-legend"><span><i style="background:#136f68"></i>GMV</span><span><i style="background:#ff7b61"></i>退货率</span></div><div class="viz" id="categoryChart"></div><p class="note">高 GMV 品类不一定拥有更低售后风险。</p></div></div>`);
panel('operation',`<div class="head"><span>02 / OPERATION</span><h2>品类、渠道与活动</h2></div><div class="grid"><div class="card"><h3>品类表现</h3>${table(D.category,['category','GMV','净利润','退货率','利润率'])}</div><div class="card"><h3>渠道表现</h3>${table(D.channel,['sales_channel','ROI','净利润','退货率'])}</div><div class="card"><h3>渠道 ROI 对比</h3><div class="viz" id="channelChart"></div><p class="note">绿色代表正向回报，珊瑚色代表投入尚未收回。</p></div><div class="card"><h3>活动表现</h3>${table(D.campaign,['campaign_name','订单量','GMV','净利润','退货率','ROI'])}</div><div class="card full"><h3>A/B 测试</h3><p>B 组相对 A 组人均消费变化 <span class="pill">${pct(D.ab.提升率)}</span>，p=${num(D.ab.p值)}。${D.ab.显著?'在 5% 显著性水平下差异显著。':'在 5% 显著性水平下，尚无充分证据证明差异显著。'} 检验以用户为单位，并对极端用户消费做 99% Winsorize。</p></div></div>`);
function heat(){let R=D.retention,cols=R.columns,rows=R.index,out=`<div class="heat" style="grid-template-columns:100px repeat(${cols.length},58px)"><div></div>${cols.map(c=>`<div>第${c}月</div>`).join('')}`;rows.forEach((r,i)=>{out+=`<div>${r}</div>`;R.data[i].forEach(v=>out+=v===null?`<div></div>`:`<div style="background:rgba(19,111,104,${.12+.82*v});color:${v>.55?'white':'#14212a'}">${pct(v)}</div>`)});return out+'</div>'}
panel('customer',`<div class="head"><span>03 / CUSTOMER</span><h2>同期群与 RFM 用户分群</h2></div><div class="grid"><div class="card full"><h3>同期群留存率</h3>${heat()}</div><div class="card"><h3>KMeans 聚类数量选择</h3><div class="viz" id="kChart"></div><p class="note">选择 K=4，兼顾轮廓系数与四类运营策略的可解释性。</p></div><div class="card"><h3>RFM 用户分群</h3><div class="viz" id="rfmChart"></div><p class="note">横轴为近期活跃度 R，纵轴为消费金额 M，气泡大小代表用户规模。</p></div><div class="card full"><h3>用户分群摘要</h3>${table(D.rfm,['用户分群','用户数','R','F','M','价值分'])}</div></div>`);
panel('risk',`<div class="head"><span>04 / RISK MODEL</span><h2>退货风险预警</h2></div><div class="grid"><div class="card"><h3>模型区分能力</h3><div class="viz" id="modelChart"></div><p class="note">模型仅使用下单时已知字段，避免评分、退货原因和实际配送天数等事后变量造成数据泄漏。</p></div><div class="card"><h3>独立测试集风险分层</h3><div class="viz" id="riskChart"></div><p class="note">高、中、低风险层由验证集概率分位点确定，并在独立测试集上校验。</p></div><div class="card full"><h3>模型评估</h3>${table(D.model,['模型','测试集AUC','测试集PR_AUC','阈值','Precision','Recall','F1'])}<p class="note">阈值选择采用 F2 最大原则，更重视召回真实退货订单，适合“预警后人工复核”的业务场景。</p></div><div class="card full"><h3>测试集风险分层</h3>${table(D.risk,['risk_level','订单数','退货数','实际退货率','平均风险分'])}</div><div class="card full"><h3>业务动作</h3><div class="flow"><div class="step"><b>高风险</b>发货前确认尺码、型号、地址和购买意愿</div><div class="step"><b>中风险</b>强化详情提示与售后提醒</div><div class="step"><b>低风险</b>正常履约，减少人工打扰</div><div class="step"><b>指标监控</b>持续跟踪 AUC、Recall 与层间退货率</div><div class="step"><b>结果回流</b>定期用真实退货结果更新模型</div><div class="step"><b>合规边界</b>不使用敏感属性实施差别待遇</div></div></div></div>`);
panel('method',`<div class="head"><span>05 / STORY</span><h2>完整链路与业务结论</h2></div><div class="flow"><div class="step"><b>1 问题提出</b>增长、利润、用户和退货四类痛点</div><div class="step"><b>2 数据获取</b>课程 demo 100 条作为种子</div><div class="step"><b>3 AI 规则扩展</b>生成约 6000 条业务模拟订单</div><div class="step"><b>4 数据清洗</b>去重、补缺、类型校验和异常过滤</div><div class="step"><b>5 分析建模</b>统计、检验、聚类和分类</div><div class="step"><b>6 展示表达</b>CSV、图表、模型与 HTML 报告</div></div><div class="head"><h2>数据驱动的建议</h2></div>${D.conclusions.map((x,i)=>`<div class="story"><b>${i+1}.</b> ${x}</div>`).join('')}<div class="grid"><div class="card full"><h3>清洗审计</h3>${table(D.cleaning,['检查项','数量'])}</div><div class="card full"><h3>项目边界</h3><p class="note">本项目使用教学模拟数据，模型结果适合展示完整分析流程，不应直接替代真实企业的业务实验、财务核算或生产决策。实际落地需使用真实曝光、转化、库存、客服和退货原因数据进行再训练与监控。</p></div></div>`);
const NS='http://www.w3.org/2000/svg',tip=document.createElement('div');tip.className='chart-tooltip';document.body.appendChild(tip);
function tooltip(e,html){tip.innerHTML=html;tip.style.left=e.clientX+'px';tip.style.top=e.clientY+'px';tip.style.opacity=1}function hideTip(){tip.style.opacity=0}
function svgRoot(id){const el=document.getElementById(id);if(!el)return null;const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 720 330');svg.setAttribute('role','img');el.appendChild(svg);return svg}
function node(svg,tag,attrs={},content=''){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(content)n.textContent=content;svg.appendChild(n);return n}
function axes(svg,yTicks=4){for(let i=0;i<=yTicks;i++){let y=274-i*(224/yTicks);node(svg,'line',{x1:56,y1:y,x2:690,y2:y,class:'grid-line'})}}
function lineChart(id,rows,xKey,yKey,format){const svg=svgRoot(id);if(!svg)return;axes(svg);const vals=rows.map(r=>+r[yKey]),max=Math.max(...vals)*1.1,min=Math.min(0,...vals);const pts=rows.map((r,i)=>[56+i*634/(rows.length-1),274-(+r[yKey]-min)/(max-min)*224]);const defs=node(svg,'defs'),grad=document.createElementNS(NS,'linearGradient');grad.id='areaGradient';grad.setAttribute('x1','0');grad.setAttribute('y1','0');grad.setAttribute('x2','0');grad.setAttribute('y2','1');defs.appendChild(grad);[['0%','#59e0c5'],['100%','#59e0c5']].forEach(([o,c],i)=>{const s=document.createElementNS(NS,'stop');s.setAttribute('offset',o);s.setAttribute('stop-opacity',i?0:.9);s.setAttribute('stop-color',c);grad.appendChild(s)});const d=pts.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ');node(svg,'path',{d:d+' L '+pts.at(-1)[0]+',274 L 56,274 Z',class:'series-area'});node(svg,'path',{d,class:'series-line'});pts.forEach((p,i)=>{const c=node(svg,'circle',{cx:p[0],cy:p[1],r:4,class:'dot'});c.onmousemove=e=>tooltip(e,'<b>'+rows[i][xKey]+'</b><br>'+yKey+'：'+format(rows[i][yKey]));c.onmouseleave=hideTip;if(i%2===0)node(svg,'text',{x:p[0],y:304,'text-anchor':'middle',class:'axis-label'},String(rows[i][xKey]).slice(2))})}
function barChart(id,rows,labelKey,valueKey,format,colors){const svg=svgRoot(id);if(!svg)return;axes(svg);const vals=rows.map(r=>+r[valueKey]),min=Math.min(0,...vals),max=Math.max(...vals),span=max-min||1,zero=274-(0-min)/span*224,w=Math.min(70,540/rows.length);node(svg,'line',{x1:46,y1:zero,x2:694,y2:zero,stroke:'#63707b','stroke-width':1});rows.forEach((r,i)=>{const x=70+i*620/rows.length,y=274-(+r[valueKey]-min)/span*224,h=Math.abs(zero-y);const rect=node(svg,'rect',{x,y:Math.min(y,zero),width:w,height:h||2,rx:7,fill:typeof colors==='function'?colors(r,i):colors||'#136f68',class:'bar-shape',style:'animation-delay:'+(i*70)+'ms'});rect.onmousemove=e=>tooltip(e,'<b>'+r[labelKey]+'</b><br>'+valueKey+'：'+format(r[valueKey]));rect.onmouseleave=hideTip;node(svg,'text',{x:x+w/2,y:304,'text-anchor':'middle',class:'axis-label'},r[labelKey])})}
function comboCategory(){const svg=svgRoot('categoryChart');if(!svg)return;axes(svg);const rows=D.category,max=Math.max(...rows.map(r=>r.GMV));rows.forEach((r,i)=>{const x=62+i*625/rows.length,w=45,h=r.GMV/max*205;const b=node(svg,'rect',{x,y:274-h,width:w,height:h,rx:7,fill:'#136f68',class:'bar-shape',style:'animation-delay:'+(i*60)+'ms'});b.onmousemove=e=>tooltip(e,'<b>'+r.category+'</b><br>GMV：'+money(r.GMV)+'<br>退货率：'+pct(r.退货率));b.onmouseleave=hideTip;const cy=274-r.退货率/.22*205;node(svg,'circle',{cx:x+w/2,cy,r:5,fill:'#ff7b61',stroke:'#fff','stroke-width':2,class:'dot'});node(svg,'text',{x:x+w/2,y:304,'text-anchor':'middle',class:'axis-label'},r.category)})}
function rfmBubble(){const svg=svgRoot('rfmChart');if(!svg)return;axes(svg);const rows=D.rfm,maxR=Math.max(...rows.map(r=>r.R)),maxM=Math.max(...rows.map(r=>r.M)),maxU=Math.max(...rows.map(r=>r.用户数)),colors=['#59e0c5','#e6b85c','#ff7b61','#526d72'];rows.forEach((r,i)=>{const cx=75+r.R/maxR*570,cy=270-Math.log1p(r.M)/Math.log1p(maxM)*210,rad=15+Math.sqrt(r.用户数/maxU)*25;const c=node(svg,'circle',{cx,cy,r:rad,fill:colors[i],class:'bubble',style:'animation-delay:'+(i*130)+'ms'});c.onmousemove=e=>tooltip(e,'<b>'+r.用户分群+'</b><br>用户：'+num(r.用户数)+'<br>R：'+num(r.R)+' · F：'+num(r.F)+'<br>M：'+money(r.M));c.onmouseleave=hideTip;node(svg,'text',{x:cx,y:cy+4,'text-anchor':'middle',class:'value-label'},r.用户分群)})}
lineChart('monthlyChart',D.monthly,'month','GMV',money);comboCategory();barChart('channelChart',D.channel,'sales_channel','ROI',num,r=>r.ROI>=0?'#136f68':'#ff7b61');lineChart('kChart',D.k_selection,'K','轮廓系数',num);rfmBubble();barChart('modelChart',D.model,'模型','测试集AUC',pct,'#136f68');barChart('riskChart',D.risk,'risk_level','实际退货率',pct,(r,i)=>['#59e0c5','#e6b85c','#ff7b61'][i]);

nav.onclick=e=>{const id=e.target.dataset.id;if(!id)return;document.getElementById(id).scrollIntoView({behavior:'smooth',block:'start'})};
const sections=[...document.querySelectorAll('.panel')];
const spy=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.id===entry.target.id))}),{rootMargin:'-22% 0px -62% 0px',threshold:0});
sections.forEach(s=>spy.observe(s));

const glow=document.createElement('div');glow.className='cursor-glow';document.body.appendChild(glow);
document.addEventListener('pointermove',e=>{glow.style.left=e.clientX+'px';glow.style.top=e.clientY+'px';document.querySelectorAll('.card:hover,.kpi:hover').forEach(el=>{const r=el.getBoundingClientRect();el.style.setProperty('--mx',(e.clientX-r.left)+'px');el.style.setProperty('--my',(e.clientY-r.top)+'px')})});
const best=D.model.slice().sort((a,b)=>b['测试集AUC']-a['测试集AUC'])[0], hi=D.risk.find(x=>x.risk_level==='高风险'), lo=D.risk.find(x=>x.risk_level==='低风险');
document.querySelector('#heroGmv').textContent=money(D.kpi.GMV);
document.querySelector('#heroAuc').textContent=(best['测试集AUC']*100).toFixed(1)+'%';
document.querySelector('#heroLift').textContent=(hi['实际退货率']/lo['实际退货率']).toFixed(1)+'×';
function animateNumber(el){if(el.dataset.done)return;el.dataset.done=1;const raw=el.textContent,match=raw.replace(/,/g,'').match(/-?\d+(\.\d+)?/);if(!match)return;const target=+match[0],start=performance.now(),duration=900;function tick(now){const p=Math.min(1,(now-start)/duration),e=1-Math.pow(1-p,4),v=target*e;el.textContent=raw.replace(match[0],target%1?v.toFixed(1):Math.round(v).toLocaleString());if(p<1)requestAnimationFrame(tick)}requestAnimationFrame(tick)}
const observer=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');e.target.querySelectorAll('.value').forEach(animateNumber);observer.unobserve(e.target)}}),{threshold:.12});
function armMotion(){document.querySelectorAll('.panel .card,.panel .kpi,.panel .story,.panel .step').forEach((el,i)=>{el.classList.add('reveal');el.style.transitionDelay=Math.min(i*45,260)+'ms';observer.observe(el)})}
armMotion();

</script>
</body></html>'''
    html = template.replace("__DATA__", data_json)
    (REPORT_DIR / "电商大促运营分析与退货风险预测.html").write_text(html, encoding="utf-8")


def main() -> None:
    seed_path = DATA_DIR / "raw_orders_demo.csv"
    if not seed_path.exists():
        raise FileNotFoundError(
            f"未找到种子数据：{seed_path}\n请确认 data/raw_orders_demo.csv 位于项目目录中。"
        )

    print("[1/8] 读取种子数据并生成 AI 规则扩展数据……", flush=True)
    seed = pd.read_csv(seed_path)
    dirty = build_extended_data(seed)
    dirty.to_csv(DATA_DIR / "ecommerce_orders_ai_extended_dirty.csv", index=False, encoding="utf-8-sig")
    print("[2/8] 清洗数据并构建业务指标……", flush=True)
    clean, cleaning_audit = clean_orders(dirty)
    clean.to_csv(DATA_DIR / "clean_orders.csv", index=False, encoding="utf-8-sig")
    save_csv(cleaning_audit, "cleaning_audit.csv")

    print("[3/8] 生成经营分析与 A/B 测试结果……", flush=True)
    kpi = {
        "订单量": int(clean["order_id"].nunique()),
        "用户数": int(clean["user_id"].nunique()),
        "GMV": float(clean["sales_amount"].sum()),
        "净销售额": float(clean["net_sales_amount"].sum()),
        "客单价": float(clean["sales_amount"].mean()),
        "净利润": float(clean["net_profit"].sum()),
        "毛利率": float(clean["gross_profit"].sum() / clean["net_sales_amount"].sum()),
        "净利率": float(clean["net_profit"].sum() / clean["net_sales_amount"].sum()),
        "退货率": float(clean["is_returned"].mean()),
        "平均评分": float(clean["customer_rating"].mean()),
    }

    category = grouped_analysis(clean, "category")
    region = grouped_analysis(clean, "region")
    channel = grouped_analysis(clean, "sales_channel")
    campaign = grouped_analysis(clean, "campaign_name")
    monthly = (
        clean.groupby("month")
        .agg(
            订单量=("order_id", "nunique"),
            GMV=("sales_amount", "sum"),
            净利润=("net_profit", "sum"),
            退货率=("is_returned", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    for filename, frame in [
        ("category_analysis.csv", category),
        ("region_analysis.csv", region),
        ("channel_analysis.csv", channel),
        ("campaign_analysis.csv", campaign),
        ("monthly_trend.csv", monthly),
    ]:
        save_csv(frame, filename)

    ab = ab_test(clean)
    save_csv(ab, "ab_test.csv")

    print("[4/8] 计算同期群留存……", flush=True)
    retention = cohort_analysis(clean)
    retention.to_csv(TABLE_DIR / "cohort_retention.csv", encoding="utf-8-sig")

    print("[5/8] 执行 RFM 与 KMeans 用户分群……", flush=True)
    rfm, rfm_summary, k_selection, rfm_bundle = rfm_kmeans(clean)
    save_csv(rfm, "rfm_user_segments.csv")
    save_csv(rfm_summary, "rfm_cluster_summary.csv")
    save_csv(k_selection, "rfm_k_selection.csv")
    joblib.dump(rfm_bundle, MODEL_DIR / "rfm_kmeans.joblib")

    print("[6/8] 训练和评估退货风险模型……", flush=True)
    model_result = train_return_model(clean)
    save_csv(model_result["evaluation"], "model_evaluation.csv")
    save_csv(model_result["threshold_table"], "model_threshold_selection.csv")
    save_csv(model_result["risk_profile"], "risk_profile_test.csv")
    save_csv(model_result["feature_importance"], "model_feature_importance.csv")
    high_risk_columns = [
        "order_id",
        "user_id",
        "order_date",
        "category",
        "sales_channel",
        "campaign_name",
        "sales_amount",
        "is_returned",
        "return_risk_score",
        "risk_level",
    ]
    high_risk = model_result["scored_orders"].sort_values("return_risk_score", ascending=False).head(200)
    save_csv(high_risk[high_risk_columns], "high_risk_orders.csv")
    joblib.dump(model_result["model_bundle"], MODEL_DIR / "return_risk_model.joblib")

    print("[7/8] 生成可视化图表……", flush=True)
    create_figures(monthly, category, channel, retention, rfm, rfm_summary, k_selection, model_result)

    top_category = category.iloc[0]
    highest_return_category = category.sort_values("退货率", ascending=False).iloc[0]
    best_channel = channel.sort_values("ROI", ascending=False).iloc[0]
    weakest_channel = channel.sort_values("ROI", ascending=True).iloc[0]
    promo = campaign.loc[campaign["campaign_name"].eq("618大促")].iloc[0]
    daily = campaign.loc[campaign["campaign_name"].eq("日常")].iloc[0]
    high_risk_rate = float(
        model_result["risk_profile"].loc[
            model_result["risk_profile"]["risk_level"].astype(str).eq("高风险"), "实际退货率"
        ].iloc[0]
    )
    low_risk_rate = float(
        model_result["risk_profile"].loc[
            model_result["risk_profile"]["risk_level"].astype(str).eq("低风险"), "实际退货率"
        ].iloc[0]
    )
    ab_row = ab.iloc[0]
    best_model_row = model_result["evaluation"].iloc[0]
    conclusions = [
        f"618 大促贡献 GMV ¥{promo['GMV']:,.0f}，退货率 {promo['退货率']:.1%}；日常订单退货率为 {daily['退货率']:.1%}。活动带来增长的同时，也明显增加售后压力。",
        f"{top_category['category']}是 GMV 最高品类（¥{top_category['GMV']:,.0f}）；{highest_return_category['category']}退货率最高（{highest_return_category['退货率']:.1%}），应优先优化商品描述、质检或尺码指引。",
        f"{best_channel['sales_channel']}渠道 ROI 最高（{best_channel['ROI']:.2f}），{weakest_channel['sales_channel']}渠道 ROI 最低（{weakest_channel['ROI']:.2f}），建议按利润效率而非只按订单量分配预算。",
        f"A/B 测试中 B 组人均消费相对 A 组变化 {ab_row['提升率']:.1%}，p={ab_row['p值']:.4f}，{'差异显著' if ab_row['显著'] else '尚无充分证据证明差异显著'}。",
        f"退货风险模型以{model_result['best_name']}表现最佳，独立测试集 AUC={best_model_row['测试集AUC']:.3f}、Recall={best_model_row['Recall']:.1%}；高风险层实际退货率为 {high_risk_rate:.1%}，约为低风险层的 {high_risk_rate / max(low_risk_rate, 1e-9):.1f} 倍。",
    ]

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "kpi": kpi,
        "category": records(category),
        "region": records(region),
        "channel": records(channel.sort_values("ROI", ascending=False)),
        "campaign": records(campaign),
        "monthly": records(monthly),
        "ab": records(ab)[0],
        "retention": {
            "index": list(retention.index),
            "columns": [int(column) for column in retention.columns],
            "data": retention.where(retention.notna(), None).values.tolist(),
        },
        "rfm": records(rfm_summary),
        "k_selection": records(k_selection),
        "model": records(model_result["evaluation"]),
        "risk": records(model_result["risk_profile"]),
        "feature_importance": records(model_result["feature_importance"].head(12)),
        "cleaning": records(cleaning_audit),
        "conclusions": conclusions,
        "model_note": {
            "best_model": model_result["best_name"],
            "threshold": model_result["best_threshold"],
            "test_base_rate": model_result["test_base_rate"],
        },
    }
    (TABLE_DIR / "dashboard_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    print("[8/8] 生成 HTML 数据故事报告……", flush=True)
    create_html_report(payload)

    summary = {
        "项目目录": str(ROOT),
        "清洗后订单数": kpi["订单量"],
        "用户数": kpi["用户数"],
        "GMV": round(kpi["GMV"], 2),
        "退货率": round(kpi["退货率"], 4),
        "最佳模型": model_result["best_name"],
        "测试集AUC": round(float(best_model_row["测试集AUC"]), 4),
        "报告": str(REPORT_DIR / "电商大促运营分析与退货风险预测.html"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
