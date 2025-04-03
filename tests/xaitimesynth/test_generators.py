def test_generator_function():
    assert generator_function() == expected_output

def test_generator_empty():
    assert list(generator_function()) == []