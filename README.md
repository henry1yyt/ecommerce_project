# 基于 AI 扩展数据的电商大促运营分析与退货风险预测

本项目以课程提供的 `data/raw_orders_demo.csv`（100 条订单）作为种子数据，使用 Python 按电商业务规律扩展为约 6000 条教学模拟订单，并完成一条可复现的数据分析与机器学习链路：

> 问题提出 → 数据获取 → AI 规则扩展 → 数据清洗 → 经营分析 → A/B 测试 → 同期群分析 → RFM + KMeans → 退货风险预测 → 结论解释 → HTML 展示

## 一、项目解决的业务问题

1. 大促带来了多少 GMV，利润是否同步增长？
2. 哪些品类、地区和渠道值得增加投入？
3. 哪些用户是高价值、潜力、沉睡或低价值用户？
4. 能否在发货前识别退货高风险订单，降低物流与售后成本？
5. A/B 营销方案的消费差异是否具有统计显著性？

## 二、优化后的项目亮点

- **一键复现**：所有路径均相对于项目目录解析，Windows、Jupyter、和鲸和 Kaggle 均可运行。
- **AI 扩展可说明**：扩展规则、字段约束和数据边界均写入 `docs/AI数据扩展说明.md`。
- **清洗过程可审计**：输出 `cleaning_audit.csv`，记录重复、缺失和异常数据数量。
- **K 值有依据**：比较 K=2～7 的惯性和轮廓系数，同时结合业务可解释性选择 K=4。
- **模型评估更严谨**：训练集、验证集、测试集分离；分类阈值仅在验证集选择；风险分层在独立测试集校验。
- **避免数据泄漏**：退货模型只使用下单时已知字段，不使用评分、实际配送天数和退货原因。
- **模型可解释**：输出特征重要性、阈值选择表、混淆矩阵指标和高风险订单清单。
- **展示完整**：同时生成 CSV、PNG 图表、模型文件和单文件 HTML 数据故事报告。

## 三、运行方法

### 方法 1：命令行一键运行

```bash
pip install -r requirements.txt
python analysis.py
python verify_project.py
```

Windows 也可以双击：

```text
run_project.bat
```

Linux / macOS：

```bash
chmod +x run_project.sh
./run_project.sh
```

### 方法 2：Jupyter / 和鲸 / Kaggle

打开：

```text
notebooks/ecommerce_analysis.ipynb
```

依次运行单元格。脚本执行完成后，打开：

```text
report/电商大促运营分析与退货风险预测.html
```

## 四、项目结构

```text
ecommerce_project/
├── analysis.py                     # 一键生成全部数据、模型、图表和报告
├── verify_project.py               # 完整性与质量检查
├── requirements.txt
├── README.md
├── .gitignore
├── run_project.bat
├── run_project.sh
├── data/
│   ├── raw_orders_demo.csv         # 课程原始 demo（100 条）
│   ├── ecommerce_orders_ai_extended_dirty.csv
│   └── clean_orders.csv
├── docs/
│   ├── AI数据扩展说明.md
│   ├── 数据字典.md
│   ├── 建模与评估说明.md
│   ├── 项目答辩提纲.md
│   └── 优化说明.md
├── notebooks/
│   └── ecommerce_analysis.ipynb
├── outputs/
│   ├── figures/                    # 8 张核心图表
│   ├── models/                     # RFM 与退货风险模型
│   └── tables/                     # 分析结果和模型评估表
└── report/
    └── 电商大促运营分析与退货风险预测.html
```

## 五、关键业务口径

- **GMV** = 单价 × 数量 × 折扣 − 优惠券金额
- **退款金额** = GMV × 是否退货
- **净销售额** = GMV − 退款金额
- **毛利润** = 净销售额 − 商品成本
- **净利润** = 毛利润 − 物流成本 − 营销成本 − 逆向物流成本
- **ROI** = 净利润 ÷ 营销成本
- **退货率** = 退货订单数 ÷ 总订单数

## 六、主要输出

### 经营分析

- `category_analysis.csv`
- `region_analysis.csv`
- `channel_analysis.csv`
- `campaign_analysis.csv`
- `monthly_trend.csv`

### 用户分析

- `cohort_retention.csv`
- `rfm_k_selection.csv`
- `rfm_cluster_summary.csv`
- `rfm_user_segments.csv`

### 风险模型

- `model_evaluation.csv`
- `model_threshold_selection.csv`
- `model_feature_importance.csv`
- `risk_profile_test.csv`
- `high_risk_orders.csv`
- `return_risk_model.joblib`

## 七、项目边界

本项目的数据为教学模拟数据，主要用于展示数据分析和机器学习的完整流程。模型结果不应直接替代真实企业的业务实验、财务核算或生产决策。真实落地时应使用实际曝光、转化、库存、客服、物流和退货原因数据重新训练，并持续监控数据漂移与模型效果。
