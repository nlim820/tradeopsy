# p05_split_orders —— 频繁拆单型

频繁拆单型：分批建仓出货（压测 FIFO）。

**模拟数据**：代码真实存在，名称/价格/成交均为构造值。

- 记录 94 条，标的 10 只
- 格式：华泰风格(GB18030)
- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/p05_split_orders --price-history examples/synthetic_accounts/p05_split_orders_price_history.csv -o report.html`
