import os
import httpx
from typing import Dict
from langchain import PromptTemplate as DefaultPromptTemplate
from langchain.llms import OpenAI
from .utils.chain import CustomLLMChain
from ..entities.managers import Message, MessageType
from ..tracing.promtelligence_tracing import PromptTemplate as TracedPromptTemplate, PromptelligenceTracer
from .base.raw_completion_manager import SyntheticUserRawCompletionManagerBase, AIModelDefaultCompletionManagerBase
from ..entities.synthetic_user import SyntheticUser
from ..utils import extract_fields, load_prompt


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")


class SyntheticUserCompletionManager(SyntheticUserRawCompletionManagerBase):
    USER_INPUT_ATTEMPTS = 3
    def __init__(self, user: SyntheticUser, target_prompt, openai_api_key, *args, **kwargs):
        self.user = user
        self.metrics = user.metrics
        self.openai_api_key = openai_api_key
        if isinstance(target_prompt, DefaultPromptTemplate):
            self.target_prompt = target_prompt
        elif type(target_prompt) is str:
            self.target_prompt = DefaultPromptTemplate(
                template=target_prompt,
                input_variables=extract_fields(target_prompt)
            )
        else:
            raise Exception(f"Unexpected type of target_prompt: {type(target_prompt)}")
        self.base_url = SPELLFORGE_HOST
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {SPELLFORGE_API_KEY}",
        }
        self._session = httpx.AsyncClient()
        super().__init__(*args, **kwargs)
        self.simulation_lab_job_id = None

    def setup_simulation_lab_job_id(self, simulation_lab_job_id):
        self.simulation_lab_job_id = simulation_lab_job_id

    async def generate_user_input(self) -> Message:
        user_response_message = await self._request_user_input()
        return Message(**user_response_message)

    async def _request_user_input(self):
        try:
            data = {"simulation_lab_job_id": self.simulation_lab_job_id}
            response = await self._session.post(
                self.base_url + "api/generate-user-input/", json=data, headers=self.header
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"Error response: {e.response.content}")
            raise e
        except Exception as e:
            print(str(e))
            raise e
        return response.json()

class AIModelDefaultCompletionManager(AIModelDefaultCompletionManagerBase):
    def __init__(self, target_prompt, llm_name, openai_api_key, *args, **kwargs):
        self.openai_api_key = openai_api_key
        if type(target_prompt) is DefaultPromptTemplate:
            self.target_prompt = target_prompt
        elif type(target_prompt) is str:
            self.target_prompt = TracedPromptTemplate(
                template=target_prompt,
                input_variables=extract_fields(target_prompt),
                alias="Customer prompt",
            )
        self.prompt_version_id = self.target_prompt.promptelligence_params.db_version_id
        self.system_prompt = TracedPromptTemplate(
            template=load_prompt(
                "completion_manager/system.completion_assistant.txt.jinja2"
            ),
            template_format="jinja2",
            input_variables=["SYSTEM_PROMPT"],
            alias="Synthetic ai model default completion system prompt"
        )
        self.tracing_layer = PromptelligenceTracer(prompt=self.system_prompt)

        llm = OpenAI(openai_api_key=openai_api_key, model_name=llm_name)
        self.chain = CustomLLMChain(
            llm=llm,
            prompt=self.system_prompt,
        )
        super().__init__(*args, **kwargs)

    async def generate_completion(self, input_variables: Dict) -> Message:
        response = await self.chain.arun(
            SYSTEM_PROMPT=self.target_prompt.format_prompt(**input_variables).text,
            callbacks=[
                self.tracing_layer
            ],
        )
        return Message(
            author=MessageType.USER,
            text=response["text"],
            run_id=str(response["__run"].run_id)
        )