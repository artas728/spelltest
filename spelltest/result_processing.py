import os
import time
import uuid
from dataclasses import asdict

import requests
import json
import numpy as np
from typing import List

from .entities.metric import Metric
from .entities.simulation import Simulation, ChatSimulationMessageStorage, CompletionSimulationMessageStorage
from .entities.simulation_job_result import SimulationJobResult
from .utils import enum_encoder

SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

def process_simulation_result(
        project_name: str,
        simulations: List[Simulation],
        llm_name,
        size: int,
        chat_mode: bool,
        temperature: float,
        reason: str,
        reason_value: str
):
    return ProcessSimulationResult(project_name, simulations, llm_name, size, chat_mode, temperature, reason, reason_value).process()


class ProcessSimulationResult:
    RESULT_FOLDER_NAME = "spelltest_result"
    ANSI_COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    def __init__(self,
                 project_name: str,
                 simulations: List[Simulation],
                 llm_name: str,
                 size: int,
                 chat_mode: bool,
                 temperature: float,
                 reason: str,
                 reason_value: str,
                 status: str = "SUCCESS"
                 ):
        self.project_name = project_name
        self.simulations = simulations
        self.llm_name = llm_name
        self.size = size
        self.chat_mode = chat_mode
        self.temperature = temperature
        self.reason = reason
        self.reason_value = reason_value
        self.status = status
        self.prompt_version_id = self.simulations[0].prompt_version_id
        self.app_user_persona_ids = list(set([simulation.app_user_persona_id for simulation in self.simulations]))

    def process(self):
        self.aggregated_metrics = self.calculate_aggregated_metrics()
        self.print_simulation_job_result()
        self.save_simulation_job_result()

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

    def print_simulation_job_result(self):
        print(f"{self.ANSI_COLORS['cyan']}📊 {'=' * 23} Simulation Results"
              f"{'=' * 23} 📊{self.ANSI_COLORS['reset']}\n")

        # General Statistics
        print(f"{self.ANSI_COLORS['yellow']}📈 General Statistics:{self.ANSI_COLORS['reset']}")
        stats = [
            ("🎯 Mean Accuracy", self.aggregated_metrics.accuracy),
            ("📏 Deviation", self.aggregated_metrics.accuracy_deviation),
        ]

        max_label_length = max(len(label) for label, _ in stats)
        for label, value in stats:
            print(f"{label.ljust(max_label_length)}: {value * 100:.2f} of 100"
                  if "Deviation" not in label else f"{label.ljust(max_label_length)}: {value * 100:.2f}")

        # Individual Simulations
        print("\n" + "🔚" + "=" * 58 + "🔚")

    def save_simulation_job_result(self):
        for simulation in self.simulations:
            if simulation.message_storage:
                if isinstance(simulation.message_storage, ChatSimulationMessageStorage):
                    chat_history_dicts = []
                    for i, message in enumerate(simulation.message_storage.chat_history):
                        message_dict = {
                            "author": message.author.name,  # Convert MessageType enum to its name
                            "text": message.text,
                            "run_id": message.run_id,
                        }
                        chat_history_dicts.append(message_dict)
                    simulation.message_storage.chat_history = chat_history_dicts
                elif isinstance(simulation.message_storage, CompletionSimulationMessageStorage):
                    simulation.message_storage.prompt = {
                        "author": simulation.message_storage.prompt.author.name,
                        "text": simulation.message_storage.prompt.text,
                        "run_id": simulation.message_storage.prompt.run_id
                    }
                    simulation.message_storage.completion = {
                        "author": simulation.message_storage.completion.author.name,
                        "text": simulation.message_storage.completion.text,
                        "run_id": simulation.message_storage.completion.run_id
                    }
        self.simulation_job_data = SimulationJobResult(
            project_name=self.project_name,
            aggregated_metrics=self.aggregated_metrics,
            simulations=self.simulations,
            llm_name=self.llm_name,
            size=self.size,
            chat_mode=self.chat_mode,
            temperature=self.temperature,
            reason=self.reason,
            reason_value=self.reason_value,
            status=self.status
        )
        # Retrieve the ID from the dictionary
        test_id = str(uuid.uuid4())

        # Get the current time in a human-readable and file-system-friendly format
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Construct the filename
        filename = f"{self.project_name}_{timestamp}_{test_id}.json"

        folder_path = os.path.join(self.RESULT_FOLDER_NAME, self.project_name)

        # Check if the folder exists, if not, create it
        os.makedirs(folder_path, exist_ok=True)

        # Save the dictionary to a file in JSON format
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.simulation_job_data), f, ensure_ascii=False, indent=4)
        print(f"Saved in {file_path} (project '{self.project_name}'), \n"
              f"to get more details open the file within spelltest browser, command `spelltest analyze`")



