def test_initialization():
    data_structure = YourDataStructure()
    assert data_structure is not None

def test_manipulation():
    data_structure = YourDataStructure()
    data_structure.add_item('item1')
    assert data_structure.get_items() == ['item1']

def test_expected_outcomes():
    data_structure = YourDataStructure()
    data_structure.add_item('item1')
    data_structure.add_item('item2')
    assert data_structure.get_items() == ['item1', 'item2']