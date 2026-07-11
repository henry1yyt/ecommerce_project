# Online Retail II：用户与经营洞察

本项目已完全移除课程 demo、AI 扩展订单及其衍生结果，统一使用和鲸公开的
`online-retail-ii（英国礼品零售商数据）`重新计算。

数据集：https://www.heywhale.com/mw/dataset/6a124df17e367d3a68e4c96b

数据包含 2009-12-01 至 2011-12-09 的 1,067,371 条真实交易明细。上游来源为
UCI Online Retail II，许可为 CC BY 4.0。

## 分析内容

- 数据质量审计与清洗
- 月度销售额、订单和客户趋势
- 国家/地区与商品贡献
- 复购率与同期群留存
- RFM 用户特征与 KMeans 四类用户分群
- K=2～7 的惯性与轮廓系数比较
- 单页交互式 HTML 数据故事

源数据没有成本、渠道、营销实验、物流时效或退货原因字段，因此项目不再展示旧版的
净利润、渠道 ROI、A/B 测试或发货前退货预测，避免使用无法验证的模拟结论。

## 运行

1. 从和鲸下载 `online_retail_II.xlsx`，放到 `data/raw/online_retail_II.xlsx`。
2. 执行：

```bash
pip install -r requirements.txt
python analysis.py
python verify_project.py
```

最终报告：`report/在线零售用户与经营洞察.html`

## 关键口径

- 有效销售：非 C 前缀且数量为正的发票明细
- 取消交易：发票号以 C 开头或数量为负
- 销售额：`Quantity × UnitPrice`
- 复购客户：有效订单数至少为 2 的客户
- RFM：距期末天数、有效发票数、累计有效销售额

`outputs/tables/` 包含全部分析表。HTML 图表由浏览器实时绘制 SVG，不引用图片。
