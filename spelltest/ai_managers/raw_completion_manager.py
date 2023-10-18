import json
import os
from typing import Dict
from langchain import PromptTemplate as DefaultPromptTemplate
from langchain.llms import OpenAI

from .tracing.cost_calculation_tracing import CostCalculationTracer
from .utils.chain import CustomLLMChain
from ..entities.managers import Message, MessageType
from .tracing.promtelligence_tracing import PromptTemplate, PromptelligenceTracer
from .base.raw_completion_manager import SyntheticUserRawCompletionManagerBase, AIModelDefaultCompletionManagerBase
from ..entities.synthetic_user import SyntheticUser
from ..utils import extract_fields, load_prompt


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "https://spellforge.ai/")
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
        self.system_prompt = PromptTemplate(
            template=load_prompt(
                "completion_manager/system.completion_user_agent.txt.jinja2"
            ),
            template_format="jinja2",
            input_variables=["APP_DESCRIPTION", "USER_DESCRIPTION", "input_variables", "target_prompt"],
            alias="Synthetic user default completion system prompt"
        )

        self.tracing_layer = PromptelligenceTracer(prompt=self.system_prompt)

        llm = OpenAI(openai_api_key=openai_api_key, model_name=user.params.llm_name)
        self.chain = CustomLLMChain(
            llm=llm,
            prompt=self.system_prompt,
        )
        super().__init__(*args, **kwargs)

    async def generate_user_input(self) -> Message:
        self.set_cost_tracker_layer()
        for _ in range(self.USER_INPUT_ATTEMPTS):
            try:
                if self.target_prompt.input_variables:
                    input_variables = self.target_prompt.input_variables
                else:
                    input_variables = '["USER_INPUT"]'
                response = await self.chain.arun(
                    APP_DESCRIPTION=self.user.params.user_knowledge_about_app,
                    USER_DESCRIPTION=self.user.params.description,
                    input_variables=input_variables,
                    target_prompt=self.target_prompt.template,
                    callbacks=[
                        self.tracing_layer, self.cost_tracker_layer
                    ],
                )
                json.loads(response["text"])
                return Message(
                    author=MessageType.USER,
                    text=response["text"],
                    run_id=str(response["__run"].run_id)
                )
            except json.decoder.JSONDecodeError as json_error:
                print(str(json_error))
        raise Exception(f"Expected JSON format from LLM but got {response}")

    def set_cost_tracker_layer(self):
        self.cost_tracker_layer = CostCalculationTracer()


class AIModelDefaultCompletionManager(AIModelDefaultCompletionManagerBase):
    def __init__(self, target_prompt, llm_name, openai_api_key, *args, **kwargs):
        self.openai_api_key = openai_api_key
        if type(target_prompt) is DefaultPromptTemplate:
            self.target_prompt = target_prompt
        elif type(target_prompt) is str:
            self.target_prompt = PromptTemplate(
                template=target_prompt,
                input_variables=extract_fields(target_prompt),
                alias="Customer prompt",
            )
        self.prompt_version_id = self.target_prompt.promptelligence_params.db_version_id
        self.system_prompt = PromptTemplate(
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
        self.cost_tracker_layer = CostCalculationTracer()
        if not self.target_prompt.input_variables and "USER_INPUT" in input_variables:
            self.target_prompt.template = f"\n USER_INPUT:\n{input_variables.pop('USER_INPUT')}\nAI:"
        response = await self.chain.arun(
            SYSTEM_PROMPT=self.target_prompt.format_prompt(**input_variables).text,
            callbacks=[
                self.tracing_layer, self.cost_tracker_layer
            ],
        )
        return Message(
            author=MessageType.ASSISTANT,
            text=response["text"],
            run_id=str(response["__run"].run_id)
        )