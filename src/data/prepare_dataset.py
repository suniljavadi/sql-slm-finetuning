import json
from pathlib import Path


def build_seed_dataset(path: str | Path = "./data/raw/sql_seed.jsonl") -> list[dict]:
    rows = [
        {
            "instruction": "Generate SQL to find customers from Hyderabad.",
            "input": "Show all customers in Hyderabad.",
            "schema": "customers(id, name, city, revenue)",
            "output": "SELECT * FROM customers WHERE city = 'Hyderabad';",
        },
        {
            "instruction": "Generate SQL to count customers by city.",
            "input": "How many customers are in each city?",
            "schema": "customers(id, city)",
            "output": "SELECT city, COUNT(*) AS customer_count FROM customers GROUP BY city ORDER BY customer_count DESC;",
        },
        {
            "instruction": "Generate SQL to identify top revenue customers.",
            "input": "List the top 5 customers by revenue.",
            "schema": "customers(id, name, revenue)",
            "output": "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 5;",
        },
        {
            "instruction": "Explain this SQL query.",
            "input": "SELECT country, COUNT(*) AS total FROM customers GROUP BY country ORDER BY total DESC;",
            "schema": "customers(id, country)",
            "output": "This query groups customers by country and counts how many customers are in each country. It then orders the results by the number of customers in descending order.",
        },
        {
            "instruction": "Correct the SQL query.",
            "input": "SELECT * FORM customers WHERE city = 'Boston';",
            "schema": "customers(id, city)",
            "output": "SELECT * FROM customers WHERE city = 'Boston';",
        },
        {
            "instruction": "Optimize the query.",
            "input": "SELECT * FROM orders WHERE order_date >= '2024-01-01' AND customer_id IN (SELECT customer_id FROM customers WHERE active = 1);",
            "schema": "orders(order_id, order_date, customer_id); customers(customer_id, active)",
            "output": "Use an INNER JOIN between orders and customers to avoid a subquery when the relationship is straightforward. Example: SELECT o.* FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date >= '2024-01-01' AND c.active = 1;",
        },
        {
            "instruction": "Write SQL for a join between orders and customers.",
            "input": "Show order IDs and customer names for orders after 2024-01-01.",
            "schema": "orders(order_id, customer_id, order_date); customers(customer_id, name)",
            "output": "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date >= '2024-01-01';",
        },
        {
            "instruction": "Generate a SQL query using a window function.",
            "input": "Rank customers by revenue within each city.",
            "schema": "customers(id, name, city, revenue)",
            "output": "SELECT city, name, revenue, RANK() OVER (PARTITION BY city ORDER BY revenue DESC) AS revenue_rank FROM customers;",
        },
        {
            "instruction": "Generate SQL to calculate running sales totals.",
            "input": "Show cumulative sales by order date.",
            "schema": "orders(order_id, order_date, amount)",
            "output": "SELECT order_date, amount, SUM(amount) OVER (ORDER BY order_date) AS running_total FROM orders;",
        },
        {
            "instruction": "Generate SQL with a CASE expression.",
            "input": "Label customers as high or low value based on revenue.",
            "schema": "customers(id, revenue)",
            "output": "SELECT id, CASE WHEN revenue >= 10000 THEN 'high_value' ELSE 'low_value' END AS value_bucket FROM customers;",
        },
        {
            "instruction": "Generate SQL using a CTE.",
            "input": "Find the average revenue per city and then display cities above the average.",
            "schema": "customers(id, city, revenue)",
            "output": "WITH city_revenue AS (SELECT city, AVG(revenue) AS avg_revenue FROM customers GROUP BY city) SELECT city FROM city_revenue WHERE avg_revenue > (SELECT AVG(avg_revenue) FROM city_revenue);",
        },
        {
            "instruction": "Generate a SQL query with JOIN and aggregation.",
            "input": "What is the total sales per customer?",
            "schema": "customers(id, name); orders(order_id, customer_id, amount)",
            "output": "SELECT c.name, SUM(o.amount) AS total_sales FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY total_sales DESC;",
        },
        {
            "instruction": "Generate SQL to filter a date range.",
            "input": "Find orders in Q1 2024.",
            "schema": "orders(order_id, order_date)",
            "output": "SELECT * FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2024-04-01';",
        },
        {
            "instruction": "Generate SQL using a subquery.",
            "input": "Find customers whose revenue is above the city average.",
            "schema": "customers(id, name, city, revenue)",
            "output": "SELECT c1.name, c1.city, c1.revenue FROM customers c1 WHERE c1.revenue > (SELECT AVG(c2.revenue) FROM customers c2 WHERE c2.city = c1.city);",
        },
        {
            "instruction": "Generate SQL to explain a query.",
            "input": "WITH recent_orders AS (SELECT customer_id, SUM(amount) AS total FROM orders WHERE order_date >= '2024-01-01' GROUP BY customer_id) SELECT c.name, r.total FROM recent_orders r JOIN customers c ON r.customer_id = c.id ORDER BY r.total DESC;",
            "schema": "orders(customer_id, amount, order_date); customers(id, name)",
            "output": "This query creates a CTE named recent_orders that filters orders from 2024 onward and sums the amount per customer. It then joins that result to the customers table to display each customer's name and total spend, sorted highest to lowest.",
        },
        {
            "instruction": "Correct SQL with wrong keyword.",
            "input": "SEECT * FROM orders WHERE amount > 100;",
            "schema": "orders(order_id, amount)",
            "output": "SELECT * FROM orders WHERE amount > 100;",
        },
        {
            "instruction": "Generate SQL to calculate monthly order totals.",
            "input": "Find the total order amount for each month.",
            "schema": "orders(order_id, order_date, amount)",
            "output": "SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS total_amount FROM orders GROUP BY DATE_TRUNC('month', order_date) ORDER BY month;",
        },
        {
            "instruction": "Generate SQL that returns filtered and sorted records.",
            "input": "Show active customers in Boston sorted by name.",
            "schema": "customers(id, name, city, active)",
            "output": "SELECT id, name FROM customers WHERE city = 'Boston' AND active = 1 ORDER BY name ASC;",
        },
        {
            "instruction": "Generate SQL to find duplicate records.",
            "input": "List duplicated account IDs.",
            "schema": "accounts(id, account_id)",
            "output": "SELECT account_id, COUNT(*) AS duplicates FROM accounts GROUP BY account_id HAVING COUNT(*) > 1;",
        },
        {
            "instruction": "Generate SQL with a left join.",
            "input": "Show all customers and their order totals if any exist.",
            "schema": "customers(id, name); orders(customer_id, amount)",
            "output": "SELECT c.id, c.name, COALESCE(SUM(o.amount), 0) AS total_orders FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name;",
        },
        {
            "instruction": "Write a SQL query for data engineering context.",
            "input": "Create a fact table summary of revenue by day.",
            "schema": "daily_sales(date, revenue)",
            "output": "SELECT date, SUM(revenue) AS daily_revenue FROM daily_sales GROUP BY date ORDER BY date;",
        },
        {
            "instruction": "Generate SQL to identify missing values.",
            "input": "Find customers missing a city.",
            "schema": "customers(id, city)",
            "output": "SELECT * FROM customers WHERE city IS NULL OR city = '';",
        },
        {
            "instruction": "Generate SQL for an analytic ranking query.",
            "input": "Rank stores by total revenue.",
            "schema": "stores(store_id, name, revenue)",
            "output": "SELECT name, revenue, DENSE_RANK() OVER (ORDER BY revenue DESC) AS revenue_rank FROM stores;",
        },
        {
            "instruction": "Generate SQL using date functions.",
            "input": "Find orders placed in the previous 30 days.",
            "schema": "orders(order_id, order_date)",
            "output": "SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY;",
        },
        {
            "instruction": "Generate SQL with filter and aggregation.",
            "input": "Show the number of orders by status for active customers only.",
            "schema": "customers(id, active); orders(order_id, customer_id, status)",
            "output": "SELECT o.status, COUNT(*) AS order_count FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.active = 1 GROUP BY o.status;",
        },
        {
            "instruction": "Generate SQL for schema-aware explanation.",
            "input": "SELECT c.name, SUM(o.amount) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.name;",
            "schema": "customers(id, name); orders(customer_id, amount)",
            "output": "This query joins customers with orders, groups records by customer name, and calculates the sum of order amounts for each customer. The LEFT JOIN ensures customers without orders still appear with a null or zero-style total depending on the database semantics.",
        },
        {
            "instruction": "Generate SQL to summarize categories.",
            "input": "Return category totals for revenue above 1000.",
            "schema": "sales(category, revenue)",
            "output": "SELECT category, SUM(revenue) AS total_revenue FROM sales WHERE revenue > 1000 GROUP BY category ORDER BY total_revenue DESC;",
        },
        {
            "instruction": "Give a corrected version of this SQL.",
            "input": "SELECT customer_id, SUM(amount) FORM orders GROUP BY customer_id;",
            "schema": "orders(customer_id, amount)",
            "output": "SELECT customer_id, SUM(amount) AS total_amount FROM orders GROUP BY customer_id;",
        },
    ]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows
