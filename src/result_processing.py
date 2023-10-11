import os
from dataclasses import asdict

import requests
import json
import numpy as np
from typing import List

from .entities.metric import Metric
from .entities.simulation import Simulation
from .entities.simulation_job_result import SimulationJobResult

SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

def process_simulation_result(
        simulations: List[Simulation],
        llm_name,
        size: int,
        reason: str,
        reason_value: str
):
    return ProcessSimulationResult(simulations, llm_name, size, reason, reason_value).process()


class ProcessSimulationResult:

    def __init__(self,
                 simulations: List[Simulation],
                 llm_name: str,
                 size: int,
                 reason: str,
                 reason_value: str,
                 status: str = "SUCCESS"
                 ):
        self.simulations = simulations
        self.llm_name = llm_name
        self.size = size
        self.reason = reason
        self.reason_value = reason_value
        self.status = status
        self.prompt_version_id = self.simulations[0].prompt_version_id
        self.app_user_persona_ids = list(set([simulation.app_user_persona_id for simulation in self.simulations]))

    def process(self) -> SimulationJobResult:
        self.aggregated_metrics = self.calculate_aggregated_metrics()
        return self.create_simulation_job()

    def calculate_aggregated_metrics(self):
        # Initialize sums and accuracy lists
        total_accuracy = 0
        total_accuracy_deviation = 0
        accuracies = []
        num_metrics = 0  # count of total metrics

        # Loop through each simulation's metrics and aggregate
        for simulation in self.simulations:
            for evaluation in simulation.evaluations:
                total_accuracy += evaluation.accuracy
                total_accuracy_deviation += evaluation.accuracy_deviation
                accuracies.append(evaluation.accuracy)
                num_metrics += 1

        # Calculate average values
        avg_accuracy = total_accuracy / num_metrics if num_metrics else 0
        avg_accuracy_deviation = total_accuracy_deviation / num_metrics if num_metrics else 0

        # Calculate percentile values
        p999 = np.percentile(accuracies, 99.9) if accuracies else 0
        p99 = np.percentile(accuracies, 99) if accuracies else 0
        p95 = np.percentile(accuracies, 95) if accuracies else 0
        p50 = np.percentile(accuracies, 50) if accuracies else 0

        aggregated_metrics = {
            "accuracy": avg_accuracy,
            "accuracy_deviation": avg_accuracy_deviation,
            "top_999_percentile_accuracy": p999,
            "top_99_percentile_accuracy": p99,
            "top_95_percentile_accuracy": p95,
            "top_50_percentile_accuracy": p50,
        }
        return Metric(**aggregated_metrics)

    def create_simulation_job(self):
        simulation_job_data = self.simulation_job_as_dict()
        headers = {"Authorization": f"Api-Key {SPELLFORGE_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(f"{SPELLFORGE_HOST}api/simulation-jobs/", data=json.dumps(simulation_job_data), headers=headers)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Error creating simulation job: {response.text}")

    def simulation_job_as_dict(self):
        """
        Convert the dataclass instance to a dictionary for HTTP request
        """
        simulation_job_result = SimulationJobResult(
            prompt_version_id=self.prompt_version_id,
            app_user_persona_ids=self.app_user_persona_ids,
            aggregated_metrics=self.aggregated_metrics,
            simulations=self.simulations,
            llm_name=self.llm_name,  # Need to determine how this is set
            size=self.size,
            temperature=0.7,  # Need to determine how this is set
            reason=self.reason,
            reason_value=self.reason_value,
            status=self.status
        )

        # Convert the dataclass instance to a dictionary for HTTP request
        simulation_job_data = asdict(simulation_job_result)
        for simulation in simulation_job_data["simulations"]:
            for evaluation in simulation["evaluations"]:
                evaluation["metric_definition_id"] = evaluation.pop("metric")["db_id"]
        return simulation_job_data

