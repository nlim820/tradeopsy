# p06_etf_dca —— ETF 定投型

ETF 定投型：月定投 + 偶尔止盈。

**模拟数据**：代码真实存在，名称/价格/成交均为构造值。

- 记录 50 条，标的 4 只
- 格式：同花顺风格 + 另一家风格拼接
- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/p06_etf_dca --price-history examples/synthetic_accounts/p06_etf_dca_price_history.csv -o report.html`
