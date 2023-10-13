import sys
import argparse
from spelltest.discover_spelltests import run_spelltests
from spelltest.yaml_tests import run_yaml_tests


def main():
    parser = argparse.ArgumentParser(description='Run spelltests')
    parser.add_argument('command', help='Command to run')

    args = parser.parse_args()

    if args.command == 'test':
        run_yaml_tests()
        run_spelltests()

# Include this if you want this script to be runnable directly
if __name__ == "__main__":
    main()
