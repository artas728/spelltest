import os
import subprocess
import webbrowser
import time
import argparse
from spelltest.discover_spelltests import run_spelltests
from spelltest.yaml_tests import run_yaml_tests

def main():
    parser = argparse.ArgumentParser(
        description='SpellTest: AI testing tool that simulates and evaluates LLM responses using synthetic users',
        epilog='Enjoy using SpellTest!'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        help='Path to the configuration file'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze the results of the simulation'
    )

    # Add the new argument for processing all directories.
    parser.add_argument(
        '--all-dirs',
        action='store_true',
        help='Process all files in all subdirectories under the current working directory'
    )

    args = parser.parse_args()

    if args.analyze:
        run_analysis()
    elif args.all_dirs:
        run_simulation()
    else:
        run_simulation(config_file=args.config_file)

def run_simulation(config_file=None):
    import openai
    openai.verify_ssl_certs = False
    if config_file is not None:
        run_yaml_tests(config_file)
    else:
        run_yaml_tests()
        run_spelltests()


def run_analysis():
    # Get the absolute path to the Streamlit app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(script_dir, "analyze_app", "streamlit_app.py")

    try:
        # Check if the streamlit server is already running, if not start it
        subprocess.check_output("pgrep streamlit", shell=True)
        print("Streamlit app is already running.")
    except subprocess.CalledProcessError:
        # Start the streamlit app as a subprocess
        subprocess.Popen(["streamlit", "run", streamlit_app_path], shell=False)
        print("Started Streamlit app.")

        # To make sure streamlit has enough time to start, otherwise the webpage might load with no server available yet
        time.sleep(5)

    # Automatically open the web page in the default web browser
    webbrowser.open("http://localhost:8501", new=2)


# Include this if you want this script to be runnable directly
if __name__ == "__main__":
    main()
