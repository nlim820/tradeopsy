# p04_student —— 学生小账户

学生小账户：几千块一笔（小样本）。

**模拟数据**：代码真实存在，名称/价格/成交均为构造值。

- 记录 15 条，标的 5 只
- 格式：同花顺风格(UTF-8 BOM)
- 运行：`PYTHONPATH=src python -m tradeopsy report examples/synthetic_accounts/p04_student --price-history examples/synthetic_accounts/p04_student_price_history.csv -o report.html`
