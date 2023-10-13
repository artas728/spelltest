from dataclasses import dataclass
from enum import Enum


@dataclass
class Mode(Enum):
    RAW_COMPLETION = "RAW_COMPLETION"
    CHAT = "CHAT"