import os
import asyncio
import openai
from typing import List
from langchain import OpenAI, PromptTemplate as DefaultPromptTemplate

from .tracing.cost_calculation_tracing import CostCalculationTracer
from ..utils import load_prompt, calculate_accuracy, \
    calculate_deviation_factor, prep_history
from .utils.chain import CustomConversationChain, CustomLLMChain
from ..ai_managers.tracing.promtelligence_tracing import PromptTemplate as TracedPromptTemplate, PromptelligenceTracer
from ..entities.managers import EvaluationResult, MessageType, Message, ConversationState
from .base.evaluation_manager import EvaluationManagerBase


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")


class EvaluationManager(EvaluationManagerBase):
    ACCURACY_EVALUATION_SHOTS = 5
    RATIONALE_SHOTS = 3
    DEFAULT_SLEEP_TIME_IF_ERROR = 10
    def __init__(self,
                 openai_api_key,
                 synthetic_user_persona_manager,
                 metric_definitions=None,
                 llm_name_default=None,
                 llm_name_perfect=None,
                 llm_name_rationale=None,
                 llm_name_accuracy=None,
                 ):
        self.metric_definitions = metric_definitions if metric_definitions else synthetic_user_persona_manager.metrics
        self.llm_name_perfect = llm_name_perfect if llm_name_perfect else llm_name_default
        self.llm_name_rationale = llm_name_rationale if llm_name_rationale else llm_name_default
        self.llm_name_accuracy = llm_name_accuracy if llm_name_accuracy else llm_name_default
        self.openai_api_key = openai_api_key
        self.synthetic_user_persona_manager = synthetic_user_persona_manager
        self.perfect_chat_response_key = "response"

    def initialize_evaluation(self):
        self.enable_cost_tracker_layer()
        self._init_perfect_chain()
        self._init_rationale_chain()
        self._init_accuracy_chain()

    def _init_perfect_chain(self):
        perfect_llm = OpenAI(openai_api_key=self.openai_api_key,
                     model_name=self.llm_name_perfect)
        perfect_chat_pre_prompt = TracedPromptTemplate(
            template=load_prompt("evaluation/perfect_chat_message.txt.jinja2"),
            template_format="jinja2",
            input_variables=["USER_DESCRIPTION", "USER_EXPECTATION"],
            alias="Perfect chat message"
        )
        self.perfect_chat_tracing_layer = PromptelligenceTracer(prompt=perfect_chat_pre_prompt)
        self.perfect_chat_prompt = DefaultPromptTemplate(
            template=perfect_chat_pre_prompt.format_prompt(
                USER_DESCRIPTION=self.synthetic_user_persona_manager.user.params.description,
                USER_EXPECTATION=self.synthetic_user_persona_manager.user.params.expectation,
            ).text,
            input_variables=["history", "input"]
        )
        self.perfect_chat_chain = CustomConversationChain(
            llm=perfect_llm,
            prompt=self.perfect_chat_prompt,
        )
        perfect_completion_pre_prompt = TracedPromptTemplate(
            template=load_prompt("evaluation/perfect_completion.txt.jinja2"),
            template_format="jinja2",
            input_variables=["USER_DESCRIPTION", "USER_EXPECTATION"],
            alias="Perfect completion"
        )
        self.perfect_completion_tracing_layer = PromptelligenceTracer(prompt=perfect_completion_pre_prompt)
        self.perfect_completion_prompt = DefaultPromptTemplate(
            template=perfect_completion_pre_prompt.format_prompt(
                USER_DESCRIPTION=self.synthetic_user_persona_manager.user.params.description,
                USER_EXPECTATION=self.synthetic_user_persona_manager.user.params.expectation,
            ).text,
            input_variables=["prompt", "completion"]
        )
        self.perfect_completion_chain = CustomLLMChain(llm=perfect_llm, prompt=self.perfect_completion_prompt)


    def _init_rationale_chain(self):
        rationale_llm = OpenAI(openai_api_key=self.openai_api_key,
                             model_name=self.llm_name_rationale)
        self.rationale_prompt = TracedPromptTemplate(
            template=load_prompt("evaluation/rationale.txt.jinja2"),
            template_format="jinja2",
            input_variables=[
                 "ACCURACY_DEFINITION",
                 "USER_EXPECTATION",
                 "ENVIRONMENT_AWARENESS",
                 "REAL_RESULT",
                 # "PERFECT_RESULT"
                             ],
            alias="Rationale"
        )
        self.rationale_tracing_layer = PromptelligenceTracer(prompt=self.rationale_prompt)
        self.rationale_chain = CustomLLMChain(llm=rationale_llm, prompt=self.rationale_prompt)

    def _init_accuracy_chain(self):
        accuracy_llm = OpenAI(openai_api_key=self.openai_api_key,
                               model_name=self.llm_name_rationale)
        self.accuracy_prompt = TracedPromptTemplate(
            template=load_prompt('evaluation/accuracy.txt.jinja2'),
            template_format="jinja2",
            input_variables=["ALL_SIMULATION_TEXT"],
            alias="Accuracy"
        )
        self.accuracy_tracing_layer = PromptelligenceTracer(prompt=self.rationale_prompt)
        self.accuracy_chain = CustomLLMChain(llm=accuracy_llm, prompt=self.accuracy_prompt)

    async def evaluate_chat(self, chat_history, user_persona_manager) -> List[EvaluationResult]:
        # self.perfect_chat_history = await self._generate_perfect_chat(chat_history)
        return await self._evaluate(chat_history)

    async def evaluate_raw_completion(self, prompt, completion, user_persona_manager) -> List[EvaluationResult]:
        # self.perfect_completion = await self._generate_perfect_completion(prompt, completion)
        return await self._evaluate("USER:\n"+prompt.text+completion.text)

    async def _generate_perfect_chat(self, chat_history: List[Message]) -> List[Message]:
        re_ask_user_manager = False
        perfect_chat_history = []
        for message in chat_history[1:]:
            if message.author is MessageType.USER:
                if re_ask_user_manager:
                    new_user_message = await self.synthetic_user_persona_manager.next_message(new_message, perfect_chat_history)
                    perfect_chat_history.append(new_user_message)
                    re_ask_user_manager = False
                    if self.synthetic_user_persona_manager.conversation_state() is ConversationState.FINISHED:
                        break
                else:
                    perfect_chat_history.append(message)
                continue
            new_message = await self._generate_perfect_chat_message(message, perfect_chat_history)
            perfect_chat_history.append(new_message)
            re_ask_user_manager = True
        return perfect_chat_history

    async def _generate_perfect_completion(self, prompt: Message, completion: Message) -> Message:
        response = await self.perfect_completion_chain.arun(
            prompt=prompt.text,
            completion=completion.text,
            callabacks=[
                self.perfect_completion_tracing_layer, self.cost_tracker_layer
            ]
        )
        return Message(
            author=MessageType.ASSISTANT,
            text=response["text"],
            run_id=str(response["__run"].run_id)
        )

    async def _generate_perfect_chat_message(self, message: Message, perfect_chat_history: List[Message]) -> Message:
        response = await self.perfect_chat_chain.arun(
            history=prep_history(perfect_chat_history),
            input=message.text,
            callbacks=[
                self.perfect_chat_tracing_layer, self.cost_tracker_layer
            ]
        )
        return Message(
            author=MessageType.ASSISTANT,
            text=response[self.perfect_chat_response_key],
            run_id=str(response["__run"].run_id)
        )

    async def _evaluate(self, chat_history):
        evaluations = []
        for metric_definition in self.metric_definitions:
            evaluations.append(await self._evaluate_single(chat_history, metric_definition))
        return evaluations

    async def _evaluate_single(self, chat_history, metric_definition, sleep_time_if_error=DEFAULT_SLEEP_TIME_IF_ERROR):
        try:
            rationale_input, rationale_output = await self._rationale(chat_history,
                                                                      metric_definition)
            accuracy, accuracy_deviation = await self._accuracy(rationale_output)
            return EvaluationResult(
                metric=metric_definition,
                accuracy=accuracy,
                accuracy_deviation=accuracy_deviation,
                rationale=rationale_output
            )
        except openai.error.RateLimitError:
            # TODO: add log
            print(f"openai.error.RateLimitError, {sleep_time_if_error=}")
            if sleep_time_if_error is not None:
                await asyncio.sleep(sleep_time_if_error)
                return await self._evaluate_single(chat_history, metric_definition, sleep_time_if_error*2)
            raise openai.error.RateLimitError

    async def _accuracy(self, all_simulation_text):
        tasks = []
        for _ in range(self.ACCURACY_EVALUATION_SHOTS):
            task = self.accuracy_chain.arun(
                ALL_SIMULATION_TEXT=all_simulation_text,
                callbacks=[
                    self.accuracy_tracing_layer, self.cost_tracker_layer
                ])
            tasks.append(task)

        quality_evaluations = []
        for shot_accuracy in await asyncio.gather(*tasks):
            try:
                quality_evaluations.append(float(shot_accuracy["text"]))
            except ValueError:
                pass
        accuracy = calculate_accuracy(quality_evaluations)
        accuracy_deviation = calculate_deviation_factor(quality_evaluations)
        return accuracy, accuracy_deviation

    async def _rationale(self, chat_history, metric_definition):
        kwargs = dict(
            ACCURACY_DEFINITION=metric_definition.definition,
            USER_EXPECTATION=self.synthetic_user_persona_manager.user.params.expectation,
            ENVIRONMENT_AWARENESS=self.synthetic_user_persona_manager.user.params.user_knowledge_about_app,
            REAL_RESULT=chat_history,
            # PERFECT_RESULT=perfect_chat_history,
        )
        input_text = self.rationale_chain.prompt.format_prompt(**kwargs).text
        rationale = await self.rationale_chain.arun(
            **kwargs,
            callbacks=[
                self.rationale_tracing_layer, self.cost_tracker_layer
            ],
        )
        return input_text, rationale["text"]

    def enable_cost_tracker_layer(self):
        self.cost_tracker_layer = CostCalculationTracer()
        try:
            self.synthetic_user_persona_manager.set_cost_tracker_layer()
        except:
            pass

    def disable_cost_tracker_layer(self):
        self.cost_tracker_layer = None
        try:
            self.synthetic_user_persona_manager.cost_tracker_layer = None
        except:
            pass