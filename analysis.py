"""Online Retail II 全量分析。

数据源：和鲸数据集 online-retail-ii（英国礼品零售商数据）
https://www.heywhale.com/mw/dataset/6a124df17e367d3a68e4c96b

将下载后的 online_retail_II.xlsx 放入 data/raw/ 后运行本脚本。脚本不会生成、扩展
或补造订单，只对公开数据进行清洗、聚合、同期群和 RFM/KMeans 分析。
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/online_retail_II.xlsx"
TABLES = ROOT / "outputs/tables"
REPORT = ROOT / "report/在线零售用户与经营洞察.html"
TABLES.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

SOURCE = {
    "name": "online-retail-ii（英国礼品零售商数据）",
    "platform": "和鲸社区",
    "url": "https://www.heywhale.com/mw/dataset/6a124df17e367d3a68e4c96b",
    "original": "Online Retail II / UCI ML Repository",
    "license": "CC BY 4.0",
}

def load_data(path: Path = RAW) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"请从和鲸下载数据并放到：{path}")
    book = pd.ExcelFile(path)
    frames = [pd.read_excel(path, sheet_name=s) for s in book.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})
    needed = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"原始文件缺少字段：{sorted(missing)}")
    return df[needed]

def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    before = len(df)
    dup = int(df.duplicated().sum())
    customer_missing = int(df.CustomerID.isna().sum())
    desc_missing = int(df.Description.isna().sum())
    df = df.drop_duplicates()
    df["InvoiceNo"] = df.InvoiceNo.astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate, errors="coerce")
    df["Quantity"] = pd.to_numeric(df.Quantity, errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df.UnitPrice, errors="coerce")
    df["is_cancelled"] = df.InvoiceNo.str.upper().str.startswith("C") | (df.Quantity < 0)
    df["line_value"] = df.Quantity.abs() * df.UnitPrice
    valid = df[
        df.InvoiceDate.notna() & df.CustomerID.notna() & df.Description.notna()
        & (df.UnitPrice > 0) & (df.Quantity != 0)
    ].copy()
    valid["CustomerID"] = valid.CustomerID.astype(int).astype(str)
    valid["Description"] = valid.Description.astype(str).str.strip()
    valid["Country"] = valid.Country.fillna("Unknown").astype(str).str.strip()
    valid["year_month"] = valid.InvoiceDate.dt.to_period("M").astype(str)
    audit = pd.DataFrame({"检查项": ["原始行数", "完全重复行", "客户ID缺失", "商品描述缺失", "清洗后有效行数"],
                          "数量": [before, dup, customer_missing, desc_missing, len(valid)]})
    return valid, audit

def analyse(df: pd.DataFrame) -> dict:
    sales = df[~df.is_cancelled].copy()
    cancelled = df[df.is_cancelled].copy()
    invoice = sales.groupby("InvoiceNo").agg(CustomerID=("CustomerID", "first"), InvoiceDate=("InvoiceDate", "min"),
        Country=("Country", "first"), revenue=("line_value", "sum"), items=("Quantity", "sum")).reset_index()
    cancelled_invoices = cancelled.InvoiceNo.nunique()
    total_invoice_events = invoice.InvoiceNo.nunique() + cancelled_invoices
    monthly = invoice.assign(month=invoice.InvoiceDate.dt.to_period("M").astype(str)).groupby("month").agg(
        revenue=("revenue", "sum"), orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique")).reset_index()
    country = invoice.groupby("Country").agg(revenue=("revenue", "sum"), orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique")).sort_values("revenue", ascending=False).reset_index().head(12)
    products = sales.groupby(["StockCode", "Description"]).agg(quantity=("Quantity", "sum"),
        revenue=("line_value", "sum"), orders=("InvoiceNo", "nunique")).sort_values("revenue", ascending=False).reset_index().head(15)

    # 商品 ABC：按销售额降序计算累计贡献率。进入 70% 的为 A，70%～90% 为 B，其余为 C。
    product_all = sales.groupby(["StockCode", "Description"]).agg(quantity=("Quantity", "sum"),
        revenue=("line_value", "sum"), orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique")).reset_index()
    product_all = product_all.sort_values("revenue", ascending=False).reset_index(drop=True)
    product_all["revenue_share"] = product_all.revenue / product_all.revenue.sum()
    product_all["cumulative_share"] = product_all.revenue_share.cumsum()
    product_all["abc_class"] = np.select(
        [product_all.cumulative_share <= .70, product_all.cumulative_share <= .90], ["A", "B"], default="C")
    # 保证跨越阈值的商品仍被归入前一档，使累计贡献完整覆盖阈值。
    for threshold, cls in [(.70, "A"), (.90, "B")]:
        idx = product_all.cumulative_share.ge(threshold).idxmax()
        product_all.loc[idx, "abc_class"] = cls
    abc_summary = product_all.groupby("abc_class", sort=False).agg(products=("StockCode", "size"),
        revenue=("revenue", "sum"), quantity=("quantity", "sum"), orders=("orders", "sum")).reset_index()
    abc_summary["revenue_share"] = abc_summary.revenue / product_all.revenue.sum()

    # 时间规律：一个订单只在其首条明细时间计数，避免明细行数放大订单量。
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    time_invoice = invoice.assign(weekday_num=invoice.InvoiceDate.dt.weekday, hour=invoice.InvoiceDate.dt.hour)
    weekday = time_invoice.groupby("weekday_num").agg(revenue=("revenue", "sum"), orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique")).reindex(range(7), fill_value=0).reset_index()
    weekday["weekday"] = weekday.weekday_num.map(dict(enumerate(weekday_names)))
    hourly = time_invoice.groupby("hour").agg(revenue=("revenue", "sum"), orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique")).reindex(range(24), fill_value=0).reset_index()

    # 取消分析：取消数量占销售与取消绝对数量合计的比例，用于识别需优先复核的商品。
    sold_qty = sales.groupby(["StockCode", "Description"]).Quantity.sum().rename("sold_quantity")
    cancel_qty = cancelled.groupby(["StockCode", "Description"]).Quantity.apply(lambda x: x.abs().sum()).rename("cancel_quantity")
    cancel_product = pd.concat([sold_qty, cancel_qty], axis=1).fillna(0).reset_index()
    cancel_product["cancel_quantity_rate"] = cancel_product.cancel_quantity / (cancel_product.sold_quantity + cancel_product.cancel_quantity)
    cancel_product["cancelled_invoices"] = cancel_product.set_index(["StockCode", "Description"]).index.map(
        cancelled.groupby(["StockCode", "Description"]).InvoiceNo.nunique()).fillna(0).astype(int)
    cancel_product = cancel_product[cancel_product.cancel_quantity >= 20].sort_values(
        ["cancel_quantity", "cancel_quantity_rate"], ascending=False).head(15)

    # 购物篮关联：以有效发票为购物篮，统计同一发票内的无序商品对。
    # 为避免极端批发发票产生组合爆炸，仅排除商品种类超过 100 的异常超大购物篮，并披露口径。
    baskets = sales.groupby("InvoiceNo").StockCode.apply(lambda x: sorted(set(map(str, x))))
    eligible = baskets[(baskets.str.len() >= 2) & (baskets.str.len() <= 100)]
    pair_counts, item_counts = Counter(), Counter()
    for items in eligible:
        item_counts.update(items)
        pair_counts.update(combinations(items, 2))
    product_names = sales[["StockCode", "Description"]].drop_duplicates("StockCode").copy()
    product_names["StockCode"] = product_names.StockCode.astype(str)
    desc_map = product_names.set_index("StockCode").Description.to_dict()
    basket_n = len(eligible)
    association_rows = []
    for (a, b), n in pair_counts.most_common():
        if n < 20:
            break
        support = n / basket_n
        conf_ab, conf_ba = n / item_counts[a], n / item_counts[b]
        lift = support / ((item_counts[a] / basket_n) * (item_counts[b] / basket_n))
        # 同一无序商品对保留更高置信度的方向，便于业务解释。
        antecedent, consequent, confidence = (a, b, conf_ab) if conf_ab >= conf_ba else (b, a, conf_ba)
        association_rows.append({"antecedent_code": antecedent, "antecedent": desc_map.get(antecedent, antecedent),
            "consequent_code": consequent, "consequent": desc_map.get(consequent, consequent), "pair_orders": n,
            "support": support, "confidence": confidence, "lift": lift})
    associations = pd.DataFrame(association_rows)
    associations = associations[(associations.pair_orders >= 50) & (associations.lift > 1.2)].sort_values(
        ["pair_orders", "confidence", "lift"], ascending=False).head(20)

    max_date = invoice.InvoiceDate.max() + pd.Timedelta(days=1)
    rfm = invoice.groupby("CustomerID").agg(last_purchase=("InvoiceDate", "max"), frequency=("InvoiceNo", "nunique"),
        monetary=("revenue", "sum")).reset_index()
    rfm["recency"] = (max_date - rfm.last_purchase).dt.days
    features = np.log1p(rfm[["recency", "frequency", "monetary"]])
    scaled = StandardScaler().fit_transform(features)
    selection = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=20).fit(scaled)
        selection.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(scaled, model.labels_)})
    # K=2 的轮廓系数最高，但只形成“高价值/其余”二分，无法支撑精细运营。
    # K=4 在惯性继续明显下降的同时轮廓系数较 K=3 回升，因此采用 K=4，
    # 并在报告中同时披露全部指标，避免把业务选择伪装成单指标最优。
    best_k = 4
    model = KMeans(n_clusters=best_k, random_state=42, n_init=30).fit(scaled)
    rfm["cluster"] = model.labels_
    summary = rfm.groupby("cluster").agg(customers=("CustomerID", "size"), recency=("recency", "mean"),
        frequency=("frequency", "mean"), monetary=("monetary", "mean")).reset_index()
    rank = summary.sort_values(["monetary", "frequency"], ascending=False).cluster.tolist()
    names = ["核心高价值", "稳定贡献", "成长潜力", "待唤醒", "低频长尾", "流失风险", "其他"]
    mapping = {c: names[i] for i, c in enumerate(rank)}
    rfm["segment"] = rfm.cluster.map(mapping)
    summary["segment"] = summary.cluster.map(mapping)
    summary = summary.sort_values("monetary", ascending=False)

    first = invoice.groupby("CustomerID").InvoiceDate.min().dt.to_period("M")
    cohort_base = invoice.assign(cohort=invoice.CustomerID.map(first), order_month=invoice.InvoiceDate.dt.to_period("M"))
    cohort_base["period"] = (cohort_base.order_month.dt.year - cohort_base.cohort.dt.year) * 12 + cohort_base.order_month.dt.month - cohort_base.cohort.dt.month
    counts = cohort_base.groupby(["cohort", "period"]).CustomerID.nunique().unstack(fill_value=0)
    retention = counts.div(counts[0], axis=0)

    repeat = (rfm.frequency > 1).mean()
    kpis = {"raw_rows": int(len(df)), "orders": int(invoice.InvoiceNo.nunique()), "customers": int(invoice.CustomerID.nunique()),
        "revenue": float(invoice.revenue.sum()), "aov": float(invoice.revenue.mean()), "repeat_rate": float(repeat),
        "cancel_rate": float(cancelled_invoices / total_invoice_events), "date_start": str(df.InvoiceDate.min().date()),
        "date_end": str(df.InvoiceDate.max().date()), "best_k": int(best_k)}
    audit_path = TABLES / "cleaning_audit.csv"
    monthly.to_csv(TABLES / "monthly_trend.csv", index=False)
    country.to_csv(TABLES / "country_analysis.csv", index=False)
    products.to_csv(TABLES / "product_analysis.csv", index=False)
    product_all.to_csv(TABLES / "product_abc_detail.csv", index=False)
    abc_summary.to_csv(TABLES / "product_abc_summary.csv", index=False)
    weekday[["weekday", "revenue", "orders", "customers"]].to_csv(TABLES / "weekday_analysis.csv", index=False)
    hourly.to_csv(TABLES / "hourly_analysis.csv", index=False)
    cancel_product.to_csv(TABLES / "cancel_product_analysis.csv", index=False)
    associations.to_csv(TABLES / "basket_associations.csv", index=False)
    rfm.drop(columns=["last_purchase"]).to_csv(TABLES / "rfm_user_segments.csv", index=False)
    summary.to_csv(TABLES / "rfm_cluster_summary.csv", index=False)
    pd.DataFrame(selection).to_csv(TABLES / "rfm_k_selection.csv", index=False)
    retention.to_csv(TABLES / "cohort_retention.csv")
    top_weekday = weekday.loc[weekday.orders.idxmax()]
    top_hour = hourly.loc[hourly.orders.idxmax()]
    a_row = abc_summary.set_index("abc_class").loc["A"]
    top_rule = associations.iloc[0]
    top_cancel = cancel_product.iloc[0]
    insights = [
        f"ABC 分析显示，仅 {int(a_row['products'])} 个 A 类商品贡献 {a_row['revenue_share']:.1%} 销售额；该结论由商品销售额降序及累计贡献率 70% 阈值得出。",
        f"订单在{top_weekday['weekday']}最多（{int(top_weekday['orders']):,} 单），单日时段在 {int(top_hour['hour']):02d}:00 最集中（{int(top_hour['orders']):,} 单）；结论来自订单级星期与小时聚合。",
        f"高关联规则“{top_rule['antecedent']} → {top_rule['consequent']}”的置信度为 {top_rule['confidence']:.1%}、提升度为 {top_rule['lift']:.2f}；提升度大于 1 表示共同出现高于随机独立预期。",
        f"取消绝对数量最高的商品是“{top_cancel['Description']}”，取消数量 {int(top_cancel['cancel_quantity']):,}；该结论按商品汇总取消/冲销记录的绝对数量得出。",
    ]
    payload = {"source": SOURCE, "kpis": kpis, "monthly": monthly.round(2).to_dict("records"),
        "country": country.round(2).to_dict("records"), "products": products.round(2).to_dict("records"),
        "k_selection": selection, "segments": summary.round(2).to_dict("records"),
        "abc_summary": abc_summary.round(4).to_dict("records"),
        "weekday": weekday[["weekday", "revenue", "orders", "customers"]].round(2).to_dict("records"),
        "hourly": hourly.round(2).to_dict("records"),
        "cancel_products": cancel_product.round(4).to_dict("records"),
        "associations": associations.round(4).to_dict("records"), "insights": insights,
        "retention": {str(i): {str(k): round(float(v), 4) for k, v in row.dropna().items()} for i, row in retention.iterrows()}}
    (TABLES / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

def build_report(data: dict) -> None:
    j = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = (ROOT / "report/template.html").read_text(encoding="utf-8")
    REPORT.write_text(template.replace("__DASHBOARD_DATA__", j), encoding="utf-8")

def main() -> None:
    raw = load_data()
    clean, audit = clean_data(raw)
    audit.to_csv(TABLES / "cleaning_audit.csv", index=False)
    data = analyse(clean)
    build_report(data)
    print(json.dumps(data["kpis"], ensure_ascii=False, indent=2))
    print(f"报告已生成：{REPORT}")

if __name__ == "__main__":
    main()
