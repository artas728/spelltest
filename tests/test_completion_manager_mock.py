import json
import os

import pytest
from spelltest.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest.entities.managers import Message, MessageType
from spelltest.ai_managers.raw_completion_manager import SyntheticUserCompletionManager, AIModelDefaultCompletionManager, CustomLLMChain
from spelltest.ai_managers.tracing.promtelligence_tracing import PromptelligenceClient
from langchain.llms.fake import FakeListLLM

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
tracing = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)

# TODO: test that we control number of messages

@pytest.mark.asyncio
async def test_initialize_completion_synthetic_user():
    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="some_llm",
        description="test user",
        expectation="hello",
        user_knowledge_about_app="app knowledge"
    )

    user = SyntheticUser(name="asedf", params=user_params, metrics=[MetricDefinition(name="metric1", definition="definition1")])
    user_manager = SyntheticUserCompletionManager(user, "My name is {name}. My request is {request}", "test_key")
    user_manager.set_cost_tracker_layer()
    # Mocking chain
    responses = [json.dumps({"Action": "Python REPL\nAction Input: print(2 + 2)"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    user_manager.chain = CustomLLMChain(llm=llm, prompt=user_manager.system_prompt)
    message = await user_manager.generate_user_input()
    assert type(json.loads(message.text)) is dict
    assert message.author == MessageType.USER
    assert message.text in responses



@pytest.mark.asyncio
async def test_initialize_completion_ai_model_manager():
    llm_name = "gpt-3.5-turbo"

    user_params = SyntheticUserParams(
        temperature=0.7,
        llm_name="some_llm",
        description="test user",
        expectation="hello",
        user_knowledge_about_app="app knowledge"
    )
    target_prompt = "test prompt {Action}"
    user = SyntheticUser(
        name="nasd",
        params=user_params,
        metrics=[MetricDefinition(name="metric1", definition="definition1")])
    user_manager = SyntheticUserCompletionManager(user, target_prompt, "test_key")

    # Mocking chain
    responses = [json.dumps({"Action": "Python REPL\nAction Input: print(2 + 2)"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    user_manager.chain = CustomLLMChain(llm=llm, prompt=user_manager.system_prompt)
    user_message = await user_manager.generate_user_input()
    input_variables = json.loads(user_message.text)
    assert type(input_variables) is dict
    assert user_message.author == MessageType.USER
    assert user_message.text in responses

    ai_model_manager = AIModelDefaultCompletionManager(target_prompt, llm_name, "test_key", role="AI", opposite_role="Human")

    # Mocking chain
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=llm_name)
    ai_model_manager.chain = CustomLLMChain(llm=llm, prompt=ai_model_manager.system_prompt)
    app_response_message = await ai_model_manager.generate_completion(input_variables)

    assert app_response_message.author == MessageType.ASSISTANT
    assert app_response_message.text in responses