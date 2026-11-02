# p07_bond_t0 —— 可转债 T+0 型

可转债 T+0 型：当日买卖（压测 T+0 与沪深前缀）。

**模拟数据**：代码真实存在，名称/价格/成交均为构造值。

- 记录 324 条，标的 6 只
- 格式：同花顺风格(UTF-8 BOM)
- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/p07_bond_t0 --price-history examples/synthetic_accounts/p07_bond_t0_price_history.csv -o report.html`
