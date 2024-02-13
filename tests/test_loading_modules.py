import igm
import pytest

def test_load_valid_custom_module():
    modules_dict = igm.get_modules_list("valid_custom_module.json")
    __ = igm.load_modules(modules_dict)

def test_load_custom_module_missing_function():
    modules_dict = igm.get_modules_list("missing_function_custom_module.json")
    
    with pytest.raises(AttributeError):
        __ = igm.load_modules(modules_dict)
