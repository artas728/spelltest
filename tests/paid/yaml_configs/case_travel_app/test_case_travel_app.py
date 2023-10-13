import pytest
import os
from spelltest.yaml_tests import run_yaml_tests  # assuming yaml_tests contains run_yaml_tests() function
import glob

# Dynamically generate list of available YAML test configurations
yaml_test_files = glob.glob('./*.yaml')

# Parametrized pytest to run for all discovered YAML test files
@pytest.mark.parametrize("yaml_file", yaml_test_files)
def test_yaml_simulations(yaml_file):
    # Assume a function in yaml_tests.py takes filename as input and processes the test
    for file in yaml_test_files:
        result = run_yaml_tests(file)
