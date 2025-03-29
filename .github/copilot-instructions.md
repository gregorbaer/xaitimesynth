# Coding Conventions

We write clean, efficient Python code with type hints and Google-style docstrings.

We use numpy for efficient vector operations when it improves readability, but prefer clarity over cleverness.

Our functions follow this pattern:
```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description.

    Longer description (if necessary)

    Args:
        param1 (type): Description.
        param2 (type): Description.
    
    Returns:
        type: Description.
    """
```

