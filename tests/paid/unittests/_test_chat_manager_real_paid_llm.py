import os

import pytest
import pytest_asyncio
from spelltest_v2.src.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest_v2.src.entities.managers import Message, MessageType
from spelltest_v2.src.ai_managers.chat_manager import SyntheticUserChatManager, AIModelDefaultChatManager, \
    CustomConversationChain
from spelltest_v2.src.ai_managers.raw_completion_manager import CustomLLMChain
from unittest.mock import AsyncMock
from spelltest_v2.src.tracing.promtelligence_tracing import PromptelligenceClient
from langchain.llms.fake import FakeListLLM


tracing = PromptelligenceClient()

TARGET_PROMPT = "Craft an email for me"

@pytest.mark.asyncio
async def test_next_message_synthetic_user():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="gpt-3.5-turbo",
        description="You're sales man that want to sell product X.",
        expectation="Well-crafted but concise  email with selling",
        user_knowledge_about_app="The app crafts emails"
    )
    user = SyntheticUser(
        name="Mail to sell X",
        params=user_params,
        metrics=[
            MetricDefinition(
                name="metric1",
                definition="Information relevance according the product accuracy score from 0 to 100"
            )
        ]
    )
    manager = SyntheticUserChatManager(user, openai_api_key)

    welcome_app_message = Message(
        author=MessageType.ASSISTANT,
        text="Hi! How I can help you",
        run_id="some self.chain.run_id"
    )
    user_first_message = await manager.initialize_conversation(welcome_app_message)
    app_message = Message(author=MessageType.ASSISTANT, text="Do you want me to make if more or less agressive?")
    user_next_message = await manager.next_message(app_message)

    assert user_first_message.author == MessageType.USER
    assert type(user_first_message.text) is str
    assert user_next_message.author == MessageType.USER
    assert type(user_next_message.text) is str


@pytest.mark.asyncio
async def test_next_message_default_chat_manager():
    llm_name = "gpt-3.5-turbo"
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    manager = AIModelDefaultChatManager(TARGET_PROMPT, llm_name, openai_api_key, role="AI", opposite_role="Human")

    app_welcome_message = await manager.initialize_conversation()

    assert app_welcome_message.author == MessageType.ASSISTANT
    assert type(app_welcome_message.text) is str
    user_message = Message(author=MessageType.USER, text="I want you to help me to sell one container of apples")
    app_response = await manager.next_message(user_message)

    assert app_response.author == MessageType.ASSISTANT
    assert type(app_response.text) is str