from dataclasses import dataclass
from typing import List

from .managers import EvaluationResult
from ..entities.metric import Metric

@dataclass
class ReasonType:
    GIT_COMMIT = "GIT_COMMIT"
    MANUAL = "MANUAL"
    LLM_UPDATE = "LLM_UPDATE"

@dataclass
class Simulation:
    prompt_version_id: int
    app_user_persona_id: int
    run_ids: List[int]
    length_complexity: float
    chat_id: str or None
    evaluations: List[EvaluationResult]
    granular_evaluation: bool = False

