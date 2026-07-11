"""Online Retail II 全量分析。

数据源：和鲸数据集 online-retail-ii（英国礼品零售商数据）
https://www.heywhale.com/mw/dataset/6a124df17e367d3a68e4c96b

将下载后的 online_retail_II.xlsx 放入 data/raw/ 后运行本脚本。脚本不会生成、扩展
或补造订单，只对公开数据进行清洗、聚合、同期群和 RFM/KMeans 分析。
"""
from __future__ import annotations

import json
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
    rfm.drop(columns=["last_purchase"]).to_csv(TABLES / "rfm_user_segments.csv", index=False)
    summary.to_csv(TABLES / "rfm_cluster_summary.csv", index=False)
    pd.DataFrame(selection).to_csv(TABLES / "rfm_k_selection.csv", index=False)
    retention.to_csv(TABLES / "cohort_retention.csv")
    payload = {"source": SOURCE, "kpis": kpis, "monthly": monthly.round(2).to_dict("records"),
        "country": country.round(2).to_dict("records"), "products": products.round(2).to_dict("records"),
        "k_selection": selection, "segments": summary.round(2).to_dict("records"),
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
