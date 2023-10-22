import os
import pytest
from spelltest.ai_managers.chat_manager import SyntheticUserChatManager
from spelltest.ai_managers.evaluation_manager import EvaluationManager, Message, MessageType
from spelltest.entities.metric import MetricDefinition
from spelltest.entities.synthetic_user import SyntheticUserParams, SyntheticUser
from spelltest.tracing.promtelligence_tracing import PromptelligenceClient
from spelltest.ai_managers.raw_completion_manager import CustomLLMChain, SyntheticUserCompletionManager

IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
tracing = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)

@pytest.fixture
def setup_manager_chat():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm_name = "gpt-3.5-turbo"
    TARGET_PROMPT = "Craft an email for me"
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
    user_chat_manager = SyntheticUserChatManager(user, openai_api_key)


    metric_definition = MetricDefinition(
        name="custom accuracy",
        definition="How accurate result is from 0 to 100 BE VERY CONCISE"
    )
    manager = EvaluationManager(
        metric_definitions=[metric_definition],
        openai_api_key=openai_api_key,
        synthetic_user_persona_manager=user_chat_manager,
        llm_name_default=llm_name,
    )
    manager.initialize_evaluation()
    return manager


@pytest.fixture
def setup_manager_completion():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm_name = "gpt-3.5-turbo"
    TARGET_PROMPT = "Craft an email for me"
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
    user_completion_manager = SyntheticUserCompletionManager(user, TARGET_PROMPT, openai_api_key)


    metric_definition = MetricDefinition(
        name="custom accuracy",
        definition="How accurate result is from 0 to 100"
    )
    manager = EvaluationManager(
        metric_definitions=[metric_definition],
        openai_api_key=openai_api_key,
        synthetic_user_persona_manager=user_completion_manager,
        llm_name_default=llm_name,
    )
    manager.initialize_evaluation()
    return manager


@pytest.mark.asyncio
async def test_accuracy(setup_manager_chat):
    accuracy, deviation = await setup_manager_chat._accuracy('<...all simulation text..>')
    assert type(accuracy) is float
    assert 0 < accuracy < 1.0
    assert type(deviation) is float
    assert 0 < deviation < 1.0


@pytest.mark.asyncio
async def test_perfect_completion(setup_manager_completion):
    prompt = Message(
        author=MessageType.USER,
        text="text"
    )
    completion = Message(
        author=MessageType.ASSISTANT,
        text="text"
    )
    perfect_message: Message = await setup_manager_completion._generate_perfect_completion(
        prompt,
        completion
    )

    assert type(perfect_message.text) is str
    assert perfect_message.author == MessageType.ASSISTANT

@pytest.mark.asyncio
async def test_perfect_chat_message(setup_manager_chat):
    chat_history = [
        Message(
            author=MessageType.USER,
            text="text"
        ),
        Message(
            author=MessageType.ASSISTANT,
            text="text2"
        )
    ]
    perfect_message: Message = await setup_manager_chat._generate_perfect_chat_message(
        Message(
            author=MessageType.USER,
            text="perfect text3"
        ),
        chat_history
    )

    assert type(perfect_message.text) is str
    assert perfect_message.author == MessageType.ASSISTANT


@pytest.mark.asyncio
async def test_evaluate(setup_manager_chat):
    chat_history = [
        Message(
            author=MessageType.USER,
            text="text"
        ),
        Message(
            author=MessageType.ASSISTANT,
            text="text2"
        )
    ]
    perfect_chat_history = [
        Message(
            author=MessageType.USER,
            text="text"
        ),
        Message(
            author=MessageType.ASSISTANT,
            text="text2"
        )
    ]

    result = await setup_manager_chat._evaluate(chat_history, perfect_chat_history)
    for evaluation in result:
        assert type(evaluation.rationale) is str
        assert type(evaluation.accuracy) is float
        assert 0 <= evaluation.accuracy <= 1.0
        assert type(evaluation.deviation) is float
        assert 0 <= evaluation.deviation <= 1.0