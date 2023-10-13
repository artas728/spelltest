# yaml_tests.py

import os
import yaml
from spelltest.entities.metric import MetricDefinition
from spelltest.spelltest import spelltest_run_simulation, SyntheticUser, SyntheticUserParams  # update module name

def parse_config(filename: str = ".spellforge.yaml"):
    with open(filename, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)
def run_yaml_tests(yaml_config_file=None):
    if yaml_config_file:
        config = parse_config(yaml_config_file)
    else:
        config = parse_config()
    defined_metrics = {
        name: MetricDefinition(name=name, definition=metric["definition"])
        for name, metric in config["metrics"].items()
    }
    for simulation_name, simulation_config in config["simulations"].items():

        prompt_file = config["prompts"][simulation_config["prompt"]]["file"]
        with open(prompt_file, 'r') as file:
            prompt_text = file.read()

        user_names = simulation_config["users"]
        if user_names == "__all__":
            user_names = config["users"].keys()

        users = []
        for user_name in user_names:
            user_config = config["users"][user_name]

            # If metrics is set to __all__, use all defined metrics
            metric_names = user_config.get("metrics", [])
            if metric_names == "__all__":
                metrics = list(defined_metrics.values())
            else:
                metrics = [
                    defined_metrics[name]
                    for name in metric_names
                    if name in defined_metrics
                ]

            user_params = SyntheticUserParams(
                temperature=user_config["temperature"],
                llm_name=user_config["llm_name"],
                description=user_config["description"],
                expectation=user_config["expectation"],
                user_knowledge_about_app=user_config["user_knowledge_about_app"]
            )

            synthetic_user = SyntheticUser(name=user_name, params=user_params, metrics=metrics)
            users.append(synthetic_user)

        # Now run simulation with spelltest_run_simulation
        return spelltest_run_simulation(
            prompt=prompt_text,
            users=users,
            llm_name=simulation_config["llm_name"],
            temperature=simulation_config["temperature"],
            size=simulation_config["size"],
            chat_mode=simulation_config["chat_mode"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),  # or your way of fetching API key
            # ... any other config parameter
        )