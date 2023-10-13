import pytest
from unittest.mock import patch, MagicMock
from spelltest import discover_spelltests


@patch("spelltest.discover_spelltests.pkgutil")
@patch("spelltest.discover_spelltests.importlib")
def test_run_spelltests(mock_importlib, mock_pkgutil):
    # Mock the module and spelltest function
    mock_module = MagicMock()
    mock_module.spelltest_example_test = MagicMock()
    mock_importlib.import_module.return_value = mock_module

    # Mock iter_modules to return an example module
    mock_pkgutil.iter_modules.return_value = [(None, "example_module", None)]

    discover_spelltests.run_spelltests()

    mock_module.spelltest_example_test.assert_called_once()


@patch("spelltest.discover_spelltests.pkgutil")
@patch("spelltest.discover_spelltests.importlib")
def test_run_spelltests_calls_all_spelltest_functions(mock_importlib, mock_pkgutil):
    # Mock the module and spelltest functions
    mock_module = MagicMock()
    mock_module.spelltest_func_1 = MagicMock()
    mock_module.spelltest_func_2 = MagicMock()
    mock_module.non_spelltest_func = MagicMock()
    mock_importlib.import_module.return_value = mock_module

    # Mock iter_modules to return an example module
    mock_pkgutil.iter_modules.return_value = [(None, "example_module", None)]

    discover_spelltests.run_spelltests()

    # Assert only the spelltest_ prefixed functions are called
    mock_module.spelltest_func_1.assert_called_once()
    mock_module.spelltest_func_2.assert_called_once()
    mock_module.non_spelltest_func.assert_not_called()
