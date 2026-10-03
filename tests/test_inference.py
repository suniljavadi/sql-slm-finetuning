import pytest

from src.inference.generate import generate_sql, validate_sql_output


def test_generate_sql_valid_select():
    sql = generate_sql("Return top 5 customers by revenue.", "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 5;")
    assert "SELECT" in sql.upper()


def test_generate_sql_uses_injected_generator_and_validates_result():
    prompts = []

    def generator(prompt, schema):
        prompts.append((prompt, schema))
        return "SELECT name FROM customers;"

    sql = generate_sql("  list customer names  ", generator=generator, schema=" customers(id, name) ")

    assert prompts == [("list customer names", "customers(id, name)")]
    assert sql == "SELECT name FROM customers;"


def test_generate_sql_trims_next_training_example_from_model_output():
    generated = "SELECT name FROM customers;Human: ### Instruction:\nnext example"

    sql = generate_sql("List customer names", generator=lambda _prompt, _schema: generated)

    assert sql == "SELECT name FROM customers;"


def test_generate_sql_removes_markdown_fence_from_model_output():
    generated = "```sql\nSELECT name FROM customers;\n```"

    sql = generate_sql("List customer names", generator=lambda _prompt, _schema: generated)

    assert sql == "SELECT name FROM customers;"


def test_generate_sql_retries_once_when_model_uses_unknown_schema_column():
    prompts = []
    outputs = iter(("SELECT customer_name FROM customers", "SELECT name FROM customers"))

    def generator(prompt, schema):
        prompts.append((prompt, schema))
        return next(outputs)

    sql = generate_sql(
        "List customer names",
        generator=generator,
        schema="customers(id, name)",
    )

    assert sql == "SELECT name FROM customers"
    assert len(prompts) == 2
    assert "unknown column 'customer_name'" in prompts[1][0]
    assert prompts[1][1] == "customers(id, name)"


def test_generate_sql_rejects_schema_mismatch_after_one_repair_attempt():
    prompts = []

    def generator(prompt, schema):
        prompts.append(prompt)
        return "SELECT display_name FROM customers"

    with pytest.raises(ValueError, match="unknown column"):
        generate_sql("List customer names", generator=generator, schema="customers(id, name)")

    assert len(prompts) == 2


def test_generate_sql_allows_sqlite_double_quoted_string_values():
    sql = generate_sql(
        "Find customers in Boston",
        generator=lambda _prompt, _schema: 'SELECT name FROM customers WHERE city = "Boston"',
        schema="customers(id, name, city)",
    )

    assert sql == 'SELECT name FROM customers WHERE city = "Boston"'


def test_generate_sql_rejects_empty_input():
    try:
        generate_sql("   ")
        raise AssertionError("Expected empty input validation to fail")
    except ValueError:
        pass


def test_generate_sql_rejects_unsafe_sql():
    try:
        validate_sql_output("DELETE FROM customers;")
        raise AssertionError("Expected unsafe SQL validation to fail")
    except ValueError:
        pass
