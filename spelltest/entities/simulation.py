from dataclasses import dataclass
from typing import List, Union
from .managers import EvaluationResult, Message

@dataclass
class ReasonType:
    GIT_COMMIT = "GIT_COMMIT"
    MANUAL = "MANUAL"
    LLM_UPDATE = "LLM_UPDATE"

@dataclass
class ChatSimulationMessageStorage:
    chat_history: List[Message]
    perfect_chat_history: List[Message]

@dataclass
class CompletionSimulationMessageStorage:
    prompt: Message
    completion: Message
    perfect_completion: Message

@dataclass
class Simulation:
    prompt_version_id: int
    app_user_persona_id: int
    run_ids: List[int]
    length_complexity: float
    chat_id: str or None
    evaluations: List[EvaluationResult]
    message_storage: Union[ChatSimulationMessageStorage, CompletionSimulationMessageStorage]
    granular_evaluation: bool = False

