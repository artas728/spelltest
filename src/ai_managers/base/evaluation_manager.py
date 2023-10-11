from abc import ABC, abstractmethod
from ...entities.managers import EvaluationResult

class EvaluationManagerBase(ABC):

    def setup_simulation_lab_job_id(self, simulation_lab_job_id):
        self.simulation_lab_job_id = simulation_lab_job_id

    @abstractmethod
    async def evaluate_chat(self, *args, **kwargs) -> EvaluationResult:
        """
        Evaluate a chat history and return the result.

        :param args: Variable arguments
        :param kwargs: Keyword arguments
        :return: Evaluation result
        """
        pass

    @abstractmethod
    async def evaluate_raw_completion(self, *args, **kwargs) -> EvaluationResult:
        """
        Evaluate a raw completion and return the result.

        :param args: Variable arguments
        :param kwargs: Keyword arguments
        :return: Evaluation result
        """
        pass