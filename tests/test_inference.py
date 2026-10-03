from src.inference.generate import generate_sql, validate_sql_output


def test_generate_sql_valid_select():
    sql = generate_sql("Return top 5 customers by revenue.", "SELECT name, revenue FROM customers ORDER BY revenue DESC LIMIT 5;")
    assert "SELECT" in sql.upper()


def test_generate_sql_uses_injected_generator_and_validates_result():
    prompts = []

    def generator(prompt):
        prompts.append(prompt)
        return "SELECT name FROM customers;"

    sql = generate_sql("  list customer names  ", generator=generator)

    assert prompts == ["list customer names"]
    assert sql == "SELECT name FROM customers;"


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
