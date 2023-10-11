from dataclasses import dataclass
from typing import List
from .metric import Metric
from .simulation import Simulation


@dataclass
class SimulationJobResult:
    prompt_version_id: int
    app_user_persona_ids: List[int]
    aggregated_metrics: Metric
    simulations: List[Simulation]
    llm_name: str
    size: int
    temperature: float
    reason: str
    reason_value: str
    status: str

    def print(self):
        pass