import os
from typing import List
from uuid import uuid4
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from .tracing.cost_calculation_tracing import CostCalculationTracer
from .utils.chain import CustomConversationChain
from ..entities.synthetic_user import SyntheticUser
from ..entities.managers import MessageType, ConversationState, Message
from ..utils import load_prompt, extract_fields, prep_history
from .base.chat_manager import ChatManagerBase
from .tracing.promtelligence_tracing import PromptTemplate, PromptelligenceTracer


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

class SyntheticUserChatManager(ChatManagerBase):
    def __init__(self, user: SyntheticUser, openai_api_key):
        self.user = user
        self.metrics = user.metrics
        self.openai_api_key = openai_api_key
        self.chat_id = str(uuid4())
        system_pre_prompt = PromptTemplate(
            template=load_prompt("chat_manager/system.chat_user_agent.txt.jinja2"),
            template_format="jinja2",
            input_variables=[
                "APP_DESCRIPTION",
                "USER_DESCRIPTION",
            ],
            alias="Synthetic user system prompt"
        )
        system_prompt_text = system_pre_prompt.format_prompt(
            APP_DESCRIPTION=self.user.params.description,
            USER_DESCRIPTION=self.user.params.user_knowledge_about_app
        ).text
        self.system_prompt = PromptTemplate(
            template=system_prompt_text,
            input_variables=["history", "input"],
            parent_alias=system_pre_prompt.alias,
        )
        self.tracing_layer = PromptelligenceTracer(prompt=system_pre_prompt)

        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=self.user.params.llm_name)
        self.chain = CustomConversationChain(llm=llm, prompt=self.system_prompt)
        super().__init__(role=MessageType.USER, opposite_role=MessageType.ASSISTANT)

    async def initialize_conversation(self, app_welcome_message: Message) -> Message:
        self.set_cost_tracker_layer()

        if app_welcome_message:
            self.chat_history.append(app_welcome_message)
        user_response = await self.chain.arun(
                history=prep_history(self.chat_history),
                input=app_welcome_message.text,
                callbacks=[self.tracing_layer, self.cost_tracker_layer],
            )
        user_response_message = Message(
            author=MessageType.USER,
            text=user_response["response"].split("> AI:")[0],
            run_id=str(user_response["__run"].run_id)
        )
        self.chat_history.append(user_response_message)
        self.state = ConversationState.STARTED
        return user_response_message

    def set_cost_tracker_layer(self):
        self.cost_tracker_layer = CostCalculationTracer()

    async def next_message(self, app_message: Message = None, chat_history: List[Message] = None) -> Message:
        if app_message and not chat_history:
            self.chat_history.append(app_message)
        user_response = await self.chain.arun(
            history=prep_history(self.chat_history if not chat_history else chat_history),
            input=app_message.text,
            callbacks=[self.tracing_layer, self.cost_tracker_layer],
        )
        user_response_message = Message(
            author=MessageType.USER,
            text=user_response["response"].split("> AI:")[0],
            run_id=str(user_response["__run"].run_id)
        )
        if not chat_history:
            self.chat_history.append(user_response_message)
        if "FINISHED" in user_response_message.text:
            user_response_message.text = user_response_message.text.replace("FINISHED", "")
            self.finish()
        else:
            # make sure ConversationState is not finished
            self.state = ConversationState.STARTED
        return user_response_message

    def finish(self):
        """
        Finish the conversation here
        :return: bool
        """
        self.state = ConversationState.FINISHED


class AIModelDefaultChatManager(ChatManagerBase):
    USER_PSEUDO_REQUEST_FOR_APP_WELCOME_MESSAGE = "Give me your welcome message"
    def __init__(self,
                 target_prompt,
                 llm_name,
                 openai_api_key,
                 target_prompt_params={},
                 temperature=0.5,
                 *args,
                 **kwargs
                 ):
        self.target_prompt = PromptTemplate(
            template=target_prompt,
            input_variables=extract_fields(target_prompt),
            alias="Customer prompt"
        )
        self.prompt_version_id = self.target_prompt.promptelligence_params.db_version_id
        self.llm_name = llm_name
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        system_pre_prompt = PromptTemplate(
            template=load_prompt(
                "chat_manager/system.chat_assistant.txt.jinja2"
            ),
            template_format="jinja2",
            input_variables=["SYSTEM_PROMPT"],
            alias="AI model default chat system prompt"
        )
        system_prompt_text = system_pre_prompt.format_prompt(
            SYSTEM_PROMPT=self.target_prompt.format_prompt(**target_prompt_params)
        ).text
        self.system_prompt = PromptTemplate(
            template=system_prompt_text,
            input_variables=["history", "input"],
            parent_alias=system_pre_prompt.alias,
        )
        self.tracing_layer = PromptelligenceTracer(prompt=system_pre_prompt)

        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=llm_name, temperature=self.temperature)
        self.chain = CustomConversationChain(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=6)
        )
        super().__init__(*args, **kwargs)

    async def initialize_conversation(self):
        self.cost_tracker_layer = CostCalculationTracer()
        system_message = Message(
            author=MessageType.SYSTEM,
            text=self.system_prompt.template,
        )
        self.chat_history.append(system_message)
        app_response = await self.chain.arun(
            history=prep_history(self.chat_history),
            input=self.USER_PSEUDO_REQUEST_FOR_APP_WELCOME_MESSAGE,
            callbacks=[self.tracing_layer, self.cost_tracker_layer],
        )
        app_message = Message(
            author=MessageType.ASSISTANT,
            text=app_response["response"],
            run_id=str(app_response["__run"].run_id)
        )
        self.chat_history.append(app_message)
        self.state = ConversationState.STARTED
        return app_message

    async def next_message(self, user_message: Message) -> Message:
        self.chat_history.append(user_message)
        app_response = await self.chain.arun(
            history=prep_history(self.chat_history),
            input=user_message.text,
            callbacks=[self.tracing_layer, self.cost_tracker_layer],
        )
        app_message = Message(
            author=MessageType.ASSISTANT,
            text=app_response["response"].split(">> Human:")[0],
            run_id=str(app_response["__run"].run_id)
        )
        self.chat_history.append(app_message)
        # print(MessageType.ASSISTANT, ":\n")
        # print(app_response)
        # print(":\n")
        return app_message
