from abc import ABC, abstractmethod
from typing import Dict
from ...entities.managers import ConversationState, MessageType, Message


class SyntheticUserRawCompletionManagerBase(ABC):
    def __init__(self, *args, **kwargs):
        self._post_init_check()

    def _post_init_check(self):
        if not hasattr(self, 'metrics'):
            raise TypeError(f"Instances of {self.__class__.__name__} must have a `metrics` attribute.")

    @abstractmethod
    def generate_user_input(self) -> Dict:
        pass

    def setup_simulation_lab_job_id(self, simulation_lab_job_id):
        self.simulation_lab_job_id = simulation_lab_job_id


class AIModelDefaultCompletionManagerBase(ABC):
    def __init__(self, *args, **kwargs):
        if hasattr(self, "target_prompt") and not hasattr(self, "prompt_version_id"):
            self.prompt_version_id = self.target_prompt.promptelligence_params.db_version_id
        self._post_init_check()

    def _post_init_check(self):
        if not hasattr(self, "prompt_version_id"):
            raise TypeError(f"Instances of {self.__class__.__name__} must have a `prompt_version_id` attribute.")

    @abstractmethod
    def generate_completion(self) -> Message:
        pass