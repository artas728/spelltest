import os
import uuid
from typing import List
import pytest
import openai
from rich.console import Console
from unittest.mock import Mock, patch, AsyncMock
from spelltest.ai_managers.evaluation_manager import EvaluationManager, Message, MessageType
from spelltest.ai_managers.tracing.promtelligence_tracing import PromptelligenceClient
from spelltest.ai_managers.tracing.cost_calculation_tracing import CostCalculationManager
from spelltest.ai_managers.raw_completion_manager import CustomLLMChain
from langchain.llms.fake import FakeListLLM

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
console = Console()
cost_calculation_manager = CostCalculationManager(console=console)
tracing = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)
@pytest.fixture
def setup_manager():
    manager = EvaluationManager(
        metric_definitions=[Mock()],
        openai_api_key='test_api_key',
        synthetic_user_persona_manager=Mock(),
        llm_name_default='llm_default',
    )
    manager.initialize_evaluation()
    manager.perfect_chat_response_key = "text"
    return manager

@pytest.mark.asyncio
async def test_init_rationale_chain(setup_manager):
    setup_manager._init_rationale_chain()
    assert setup_manager.rationale_chain is not None

@pytest.mark.asyncio
async def test_init_accuracy_chain(setup_manager):
    setup_manager._init_accuracy_chain()
    assert setup_manager.accuracy_chain is not None

@pytest.mark.asyncio
async def test_accuracy_calculation(setup_manager):
    responses = ["87.0", "87.0", "87.0", "87.0", "87.0", "87.0"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_accuracy)
    setup_manager.accuracy_chain = CustomLLMChain(llm=llm, prompt=setup_manager.accuracy_prompt)
    accuracy, deviation = await setup_manager._accuracy('simulation_text')

    assert accuracy == 0.87
    assert deviation == 0.0


@pytest.mark.asyncio
async def test_accuracy_value_error(setup_manager):
    responses = ["87.0", "ERROR", "ERROR", "87.0", "87.0", "87.0"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_accuracy)
    setup_manager.accuracy_chain = CustomLLMChain(llm=llm, prompt=setup_manager.accuracy_prompt)
    accuracy, deviation = await setup_manager._accuracy('simulation_text')

    assert accuracy == 0.87
    assert deviation == 0.0


@pytest.mark.asyncio
async def test_perfect_completion(setup_manager):
    responses = ["bla bla bla bla", "bla bla bla bla", "bla bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_perfect)
    setup_manager.perfect_completion_chain = CustomLLMChain(llm=llm, prompt=setup_manager.perfect_completion_prompt)
    prompt = Message(
        author=MessageType.USER,
        text="text"
    )
    completion = Message(
        author=MessageType.ASSISTANT,
        text="text"
    )
    perfect_message: Message = await setup_manager._generate_perfect_completion(
        prompt,
        completion
    )

    assert perfect_message.text == 'bla bla bla bla'
    assert perfect_message.author == MessageType.ASSISTANT



@pytest.mark.asyncio
async def test_evaluate(setup_manager):
    responses = ["87.0", "87.0", "87.0", "87.0", "87.0", "87.0"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_accuracy)
    setup_manager.accuracy_chain = CustomLLMChain(llm=llm, prompt=setup_manager.accuracy_prompt)

    rationale_responses = ["rationale"]
    rationale_llm = FakeListLLM(responses=rationale_responses, model_name=setup_manager.llm_name_rationale)
    setup_manager.rationale_chain = CustomLLMChain(llm=rationale_llm, prompt=setup_manager.rationale_prompt)

    result = await setup_manager._evaluate('chat_history',  'perfect_chat_history')
    for simulation in result:
        assert simulation.rationale_output == 'rationale'
        assert simulation.accuracy == 0.87
        assert simulation.accuracy_deviation == 0.0



@pytest.mark.asyncio
async def test_evaluate(setup_manager):
    responses = ["87.0", "87.0", "87.0", "87.0", "87.0", "87.0"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_accuracy)
    setup_manager.accuracy_chain = CustomLLMChain(llm=llm, prompt=setup_manager.accuracy_prompt)

    rationale_responses = ["rationale"]
    rationale_llm = FakeListLLM(responses=rationale_responses, model_name=setup_manager.llm_name_rationale)
    setup_manager.rationale_chain = CustomLLMChain(llm=rationale_llm, prompt=setup_manager.rationale_prompt)

    result = await setup_manager._evaluate('chat_history')
    for simulation in result:
        assert simulation.rationale == 'rationale'
        assert simulation.accuracy == 0.87
        assert simulation.accuracy_deviation == 0.0