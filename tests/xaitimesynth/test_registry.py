def test_registry_functionality():
    registry = Registry()
    item = "test_item"
    
    registry.register(item)
    assert registry.retrieve(item) == item
    assert registry.retrieve("non_existent_item") is None

    registry.unregister(item)
    assert registry.retrieve(item) is None