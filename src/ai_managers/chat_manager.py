import os
import httpx
from dataclasses import asdict
from uuid import uuid4
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from .utils.chain import CustomConversationChain
from ..entities.synthetic_user import SyntheticUser
from ..entities.managers import MessageType, ConversationState, Message
from ..utils import load_prompt, extract_fields, prep_history
from .base.chat_manager import ChatManagerBase
from ..tracing.promtelligence_tracing import PromptTemplate, PromptelligenceTracer


SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")


class SyntheticUserChatManager(ChatManagerBase):
    def __init__(self, user: SyntheticUser, openai_api_key):
        self.user = user
        self.metrics = user.metrics
        self.openai_api_key = openai_api_key
        self.chat_id = str(uuid4())
        self.base_url = SPELLFORGE_HOST
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {SPELLFORGE_API_KEY}",
        }
        self._session = httpx.AsyncClient()
        self.simulation_lab_job_id = None
    #     super().__init__(role=MessageType.USER, opposite_role=MessageType.ASSISTANT)

    async def initialize_conversation(self, app_welcome_message: Message) -> Message:
        return await self.next_message(app_welcome_message)

    async def next_message(self, app_message: Message = None) -> Message:
        user_response_message = await self._request_next_message(app_message)
        return Message(**user_response_message)

    async def _request_next_message(self, data):
        message = asdict(data)
        message["author"] = data.author.value
        if not self.simulation_lab_job_id:
            raise Exception("Setup simulation_lab_job_id first")
        message["simulation_lab_job_id"] = self.simulation_lab_job_id
        try:
            response = await self._session.post(
                self.base_url + "api/generate-next-message/", json=message, headers=self.header
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"Error response: {e.response.content}")
            raise e
        except Exception as e:
            print(str(e))
            raise e
        return response.json()

    def finish(self):
        """
        Finish the conversation here
        :return: bool
        """
        self.state = ConversationState.FINISHED


class AIModelDefaultChatManager(ChatManagerBase):
    USER_PSEUDO_REQUEST_FOR_APP_WELCOME_MESSAGE = "Give me your welcome message"
    def __init__(self, target_prompt, llm_name, openai_api_key, target_prompt_params={}, temperature=0.5, *args, **kwargs):
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

        llm = OpenAI(openai_api_key=openai_api_key, model_name=llm_name, temperature=self.temperature)
        self.chain = CustomConversationChain(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=6)
        )
        super().__init__(*args, **kwargs)

    async def initialize_conversation(self):
        system_message = Message(
            author=MessageType.SYSTEM,
            text=self.system_prompt.template,
        )
        self.chat_history.append(system_message)
        app_response = await self.chain.arun(
            history=prep_history(self.chat_history),
            input=self.USER_PSEUDO_REQUEST_FOR_APP_WELCOME_MESSAGE,
            callbacks=[self.tracing_layer],
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
            callbacks=[self.tracing_layer],
        )
        app_message = Message(
            author=MessageType.ASSISTANT,
            text=app_response["response"].split(">> Human:")[0],
            run_id=str(app_response["__run"].run_id)
        )
        self.chat_history.append(app_message)
        print(MessageType.ASSISTANT, ":\n")
        print(app_response)
        print(":\n")
        return app_message
