import json
import os

import pytest
import pytest_asyncio
from spelltest_v2.src.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest_v2.src.entities.managers import Message, MessageType
from spelltest_v2.src.ai_managers.raw_completion_manager import SyntheticUserCompletionManager, AIModelDefaultCompletionManager, CustomLLMChain
from unittest.mock import AsyncMock
from spelltest_v2.src.tracing.promtelligence_tracing import PromptelligenceClient
from langchain.llms.fake import FakeListLLM


tracing = PromptelligenceClient()

# TODO: test that we control number of messages

@pytest.mark.asyncio
async def test_initialize_completion_synthetic_user():
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
    user_manager = SyntheticUserCompletionManager(user, "My name is {name}. My request is {request}", openai_api_key)

    message = await user_manager.generate_user_input()
    assert type(json.loads(message.text)) is dict
    assert message.author == MessageType.USER
    assert type(message.text) is str



@pytest.mark.asyncio
async def test_initialize_completion_ai_model_manager():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm_name = "gpt-3.5-turbo"
    target_prompt = "test prompt {Action}"
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name=llm_name,
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
    user_manager = SyntheticUserCompletionManager(user, target_prompt, openai_api_key)
    user_message = await user_manager.generate_user_input()
    input_variables = json.loads(user_message.text)
    assert type(input_variables) is dict
    assert user_message.author == MessageType.USER
    assert type(user_message.text) is str

    ai_model_manager = AIModelDefaultCompletionManager(target_prompt, llm_name, openai_api_key, role="AI", opposite_role="Human")
    app_response_message = await ai_model_manager.generate_completion(input_variables)

    assert app_response_message.author == MessageType.ASSISTANT
    assert type(app_response_message.text) is str