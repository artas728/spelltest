import importlib
import pkgutil


def run_spelltests():
    path = 'path_to_your_tests'  # Modify this to your test directory
    for _, name, _ in pkgutil.iter_modules([path]):
        module = importlib.import_module(name)

        # Run functions that start with 'spelltest_'
        for item_name in dir(module):
            if item_name.startswith('spelltest_'):
                item = getattr(module, item_name)
                if callable(item):
                    item()  # Execute the test