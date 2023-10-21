import os
import pytest
from spelltest.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest.entities.managers import Message, MessageType
from spelltest.ai_managers.chat_manager import SyntheticUserChatManager, AIModelDefaultChatManager, \
    CustomConversationChain
from spelltest.ai_managers.tracing.promtelligence_tracing import PromptelligenceClient
from langchain.llms.fake import FakeListLLM

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
tracing = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)

# TODO: test that we control number of messages

@pytest.mark.asyncio
async def test_initialize_conversation_synthetic_user():
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="some_llm",
        description="test user",
        expectation="hello",
        user_knowledge_about_app="app knowledge"
    )

    user = SyntheticUser(name="name", params=user_params, metrics=[MetricDefinition(name="metric1", definition="definition1")])
    manager = SyntheticUserChatManager(user, "test_key")
    manager.set_cost_tracker_layer()
    # Mocking chain
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    manager.chain = CustomConversationChain(llm=llm, prompt=manager.system_prompt)

    welcome_message = Message(
        author=MessageType.ASSISTANT,
        text="Hi! I'm you assistant",
        run_id="some self.chain.run_id"
    )
    message = await manager.initialize_conversation(welcome_message)

    assert message.author == MessageType.USER
    assert message.text in responses

@pytest.mark.asyncio
async def test_next_message_synthetic_user():
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="some_llm",
        description="test user",
        expectation="hello",
        user_knowledge_about_app="app knowledge"
    )

    user = SyntheticUser(name="anme", params=user_params, metrics=[MetricDefinition(name="metric1", definition="definition1")])
    manager = SyntheticUserChatManager(user, "test_key")

    # Mocking chain
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    manager.chain = CustomConversationChain(llm=llm, prompt=manager.system_prompt)

    welcome_app_message = Message(
        author=MessageType.ASSISTANT,
        text="Hi! I'm you assistant",
        run_id="some self.chain.run_id"
    )
    user_first_message = await manager.initialize_conversation(welcome_app_message)
    app_message = Message(author=MessageType.ASSISTANT, text="Here is what I cen do for you ... bal bla bla ...")
    user_next_message = await manager.next_message(app_message)

    assert user_first_message.author == MessageType.USER
    assert user_first_message.text in responses
    assert user_next_message.author == MessageType.USER
    assert user_next_message.text in responses


@pytest.mark.asyncio
async def test_initialize_conversation_default_chat_manager():
    llm_name = "gpt-3.5-turbo"
    manager = AIModelDefaultChatManager("test prompt", llm_name, "test_key", role="AI", opposite_role="Human")

    # Mocking chain
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=llm_name)
    manager.chain = CustomConversationChain(llm=llm, prompt=manager.system_prompt)

    app_message = await manager.initialize_conversation()

    assert app_message.author == MessageType.ASSISTANT
    assert app_message.text in responses


@pytest.mark.asyncio
async def test_next_message_default_chat_manager():
    llm_name = "gpt-3.5-turbo"
    manager = AIModelDefaultChatManager("test prompt", llm_name, "test_key", role="AI", opposite_role="Human")

    # Mocking chain
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=llm_name)
    manager.chain = CustomConversationChain(llm=llm, prompt=manager.system_prompt)

    app_welcome_message = await manager.initialize_conversation()

    assert app_welcome_message.author == MessageType.ASSISTANT
    assert app_welcome_message.text in responses

    user_message = Message(author=MessageType.USER, text="Can you assist?")
    app_response = await manager.next_message(user_message)

    assert app_response.author == MessageType.ASSISTANT
    assert app_response.text in responses