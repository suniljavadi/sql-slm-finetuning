import shutil
import sqlite3

import pytest

from src.evaluation.bird_execution import evaluate_bird_examples, read_sqlite_schema
from src.evaluation.sqlite_execution import execute_read_only_query, execution_matches


@pytest.fixture
def customer_db(tmp_path):
    path = tmp_path / "customers.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER, name TEXT, city TEXT)")
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(1, "Ada", "Boston"), (2, "Lin", "Boston"), (3, "Grace", "Seattle")],
        )
    return path


def test_execution_matches_equivalent_sql_and_respects_order(customer_db):
    expected = "SELECT name FROM customers WHERE city = 'Boston' ORDER BY id"
    equivalent = "SELECT name FROM customers WHERE city = 'Boston' ORDER BY id ASC"
    reversed_order = "SELECT name FROM customers WHERE city = 'Boston' ORDER BY id DESC"

    assert execution_matches(customer_db, expected, equivalent)
    assert not execution_matches(customer_db, expected, reversed_order)


def test_execution_compares_unordered_results_as_bags(customer_db):
    expected = "SELECT name FROM customers WHERE city = 'Boston'"
    equivalent = "SELECT name FROM customers WHERE id IN (1, 2)"

    assert execution_matches(customer_db, expected, equivalent)


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM customers", "SELECT name FROM customers; DROP TABLE customers"],
)
def test_execution_rejects_mutations_and_multiple_statements(customer_db, sql):
    with pytest.raises(ValueError):
        execute_read_only_query(customer_db, sql)


def test_execution_enforces_result_row_limit(customer_db):
    with pytest.raises(ValueError, match="row limit"):
        execute_read_only_query(customer_db, "SELECT name FROM customers", max_rows=2)


def test_bird_evaluation_builds_schema_and_measures_execution(customer_db):
    schema = read_sqlite_schema(customer_db)
    assert 'CREATE TABLE "customers"' in schema
    assert '"city" TEXT' in schema

    rows = [
        {
            "db_id": "customers",
            "question_id": 1,
            "question": "List Boston customer names",
            "evidence": "",
            "SQL": "SELECT name FROM customers WHERE city = 'Boston'",
        }
    ]
    database_dir = customer_db.parent / "bird_databases"
    bird_database_dir = database_dir / "customers"
    bird_database_dir.mkdir(parents=True)
    shutil.copyfile(customer_db, bird_database_dir / "customers.sqlite")
    report = evaluate_bird_examples(
        rows,
        database_dir,
        lambda record: "SELECT name FROM customers WHERE city = 'Boston'",
    )

    assert report["summary"]["execution_accuracy"] == 1.0


def test_bird_sampler_is_seeded_and_balanced_across_databases():
    from src.evaluation.bird_execution import select_bird_examples

    rows = [
        {"db_id": database, "question_id": index}
        for database in ("a", "b", "c")
        for index in range(10)
    ]

    first = select_bird_examples(rows, max_examples=6, seed=7)
    second = select_bird_examples(rows, max_examples=6, seed=7)

    assert first == second
    assert {row["db_id"] for row in first} == {"a", "b", "c"}
    assert all(sum(row["db_id"] == database for row in first) == 2 for database in ("a", "b", "c"))