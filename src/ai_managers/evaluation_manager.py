import asyncio
import httpx
import os
from typing import List
from ..entities.managers import EvaluationResult, MessageType, Message, ConversationState
from .base.evaluation_manager import EvaluationManagerBase


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")


class EvaluationManager(EvaluationManagerBase):
    ACCURACY_EVALUATION_SHOTS = 5
    RATIONALE_SHOTS = 3
    DEFAULT_SLEEP_TIME_IF_ERROR = 10
    def __init__(self,
                 simulation_lab_job_id,
                 openai_api_key,
                 synthetic_user_persona_manager,
                 metric_definitions=None,
                 llm_name_default=None,
                 llm_name_perfect=None,
                 llm_name_rationale=None,
                 llm_name_accuracy=None,
                 ):
        self.simulation_lab_job_id = simulation_lab_job_id
        self.metric_definitions = metric_definitions if metric_definitions else synthetic_user_persona_manager.metrics
        self.llm_name_default = llm_name_default
        self.llm_name_perfect = llm_name_perfect if llm_name_perfect else llm_name_default
        self.llm_name_rationale = llm_name_rationale if llm_name_rationale else llm_name_default
        self.llm_name_accuracy = llm_name_accuracy if llm_name_accuracy else llm_name_default
        self.openai_api_key = openai_api_key
        self.synthetic_user_persona_manager = synthetic_user_persona_manager
        self.perfect_chat_response_key = "response"

        self.base_url = SPELLFORGE_HOST
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {SPELLFORGE_API_KEY}",
        }
        self._session = httpx.AsyncClient()
        self.simulation_lab_job_id = None


    async def evaluate_chat(self, chat_history, user_persona_manager) -> List[EvaluationResult]:
        return await self._request_evaluate(self.prepare_data_basic_data(), chat_mode=True)

    async def evaluate_raw_completion(self, prompt, completion) -> List[EvaluationResult]:
        data = self.prepare_data_basic_data()
        data["prompt"] = prompt
        data["completion"] = completion
        return await self._request_evaluate(data, chat_mode=False)

    def prepare_data_basic_data(self):
        return {
            "simulation_lab_job_id": self.simulation_lab_job_id,
            "evaluation_llm_name": self.llm_name_default,
            "evaluation_llm_name_perfect": self.llm_name_perfect,
            "evaluation_llm_name_rationale": self.llm_name_rationale,
            "evaluation_llm_name_accuracy": self.llm_name_accuracy,
        }

    async def _request_evaluate(self, data, chat_mode):
        try:
            response = await self._session.post(
                self.base_url + "api/evaluate-chat/" if chat_mode else "/api/evaluate-completion/",
                json=data, headers=self.header
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"Error response: {e.response.content}")
            raise e
        except Exception as e:
            print(str(e))
            raise e
        return response.json()