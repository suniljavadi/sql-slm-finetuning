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
