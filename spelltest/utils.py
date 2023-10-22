import math
import string
from enum import Enum
from pathlib import Path

from spelltest.entities.managers import MessageType

THIS_DIRECTORY = Path(__file__).parent.absolute()
TEMPLATE_DIRECTORY = THIS_DIRECTORY / "prompts"


def calculate_accuracy(quality_evaluations):
    return float(sum(quality_evaluations) / len(quality_evaluations) / 100)


def calculate_deviation_factor(quality_evaluations):
    n = len(quality_evaluations)
    mean = sum(quality_evaluations) / n
    squared_differences = [(x - mean) ** 2 for x in quality_evaluations]
    variance = sum(squared_differences) / n
    standard_deviation = math.sqrt(variance)
    return standard_deviation / 100


def load_prompt(template):
    file_destination = TEMPLATE_DIRECTORY / template
    return file_destination.read_text()


def extract_fields(s):
    formatter = string.Formatter()
    return [field_name for _, field_name, _, _ in formatter.parse(s) if field_name]


def prep_history(chat_history):
    history_strings_list = []
    for message in chat_history:
        if type(message.author) is MessageType.ASSISTANT:
            history_strings_list.append(
                f"> AI:\n {message.text}"
            )
        elif type(message.author) is MessageType.USER:
            history_strings_list.append(
                f">> Human:\n {message.text}"
            )
    if len(history_strings_list) > 3:
        history_strings_list = history_strings_list[-3:]
    return "\n".join(history_strings_list)

# Define a custom encoder function to handle enums
def enum_encoder(obj):
    if isinstance(obj, Enum):
        return obj.name  # Serialize enum to its name (string)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")
