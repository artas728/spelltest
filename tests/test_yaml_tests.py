import pytest
from spelltest import yaml_tests
from unittest.mock import patch, mock_open


@patch("builtins.open", new_callable=mock_open, read_data="key: value")
def test_parse_config(mock_file):
    result = yaml_tests.parse_config(".spellforge.yaml")
    assert result == {"key": "value"}


# Mock Data
mock_config = {
    "metrics": {},
    "simulations": {
        "test_simulation_1": {"prompt": "test_prompt_1", "users": ["user_1"], "llm_name": "", "temperature": 0.7,
                              "size": 5, "chat_mode": True},
        "test_simulation_2": {"prompt": "test_prompt_2", "users": ["user_2"], "llm_name": "", "temperature": 0.7,
                              "size": 5, "chat_mode": True}
    },
    "prompts": {
        "test_prompt_1": {"file": "test_prompt_1.txt"},
        "test_prompt_2": {"file": "test_prompt_2.txt"}
    },
    "users": {
        "user_1": {"llm_name": "", "temperature": 0.7, "description": "", "expectation": "",
                   "user_knowledge_about_app": "", "metrics": "__all__"},
        "user_2": {"llm_name": "", "temperature": 0.7, "description": "", "expectation": "",
                   "user_knowledge_about_app": "", "metrics": "__all__"}
    }
}


# @patch("builtins.open", new_callable=mock_open, read_data="prompt text")
# @patch("spelltest.yaml_tests.spelltest_run_simulation")
# @patch("spelltest.yaml_tests.parse_config")
# def test_run_yaml_tests_calls_simulation(mock_parse_config, mock_spelltest_run_simulation, mock_file):
#     mock_parse_config.return_value = mock_config
#     yaml_tests.run_yaml_tests()
#     assert mock_spelltest_run_simulation.call_count == len(mock_config["simulations"])