# p10_mixed —— 无模式基线

无模式基线：胜负各半（对照组）。

**模拟数据**：代码真实存在，名称/价格/成交均为构造值。

- 记录 126 条，标的 18 只
- 格式：同花顺风格 + 另一家风格拼接
- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/p10_mixed --price-history examples/synthetic_accounts/p10_mixed_price_history.csv -o report.html`
