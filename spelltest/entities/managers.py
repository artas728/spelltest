from enum import Enum
from dataclasses import dataclass
from typing import List

from .metric import MetricDefinition



@dataclass
class EvaluationResult:
    metric: MetricDefinition
    accuracy: float
    accuracy_deviation: float
    rationale: str


@dataclass
class ConversationState(Enum):
    CREATED = "CREATED"
    STARTED = "STARTED"
    FINISHED = "FINISHED"


@dataclass
class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    author: MessageType
    text: str
    run_id: str = None
