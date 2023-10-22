import json
import pytest
from typing import List
from unittest.mock import patch
from spelltest.ai_managers.utils.chain import CustomConversationChain
from spelltest.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest.spelltest import spelltest
from spelltest.ai_managers.chat_manager import ChatManagerBase
from spelltest.ai_managers.evaluation_manager import EvaluationManager, EvaluationManagerBase, EvaluationResult
from spelltest.ai_managers.chat_manager import Message
from spelltest.ai_managers.raw_completion_manager import CustomLLMChain
from langchain.llms.fake import FakeListLLM


PROMPT = "Write personalized sales email according user requirements"
metric = MetricDefinition(
    name="accuracy",
    definition="accuracy",
)
user = SyntheticUser(
        name='user',
        params=SyntheticUserParams(
            temperature=0.8,
            llm_name="gpt-3.5-turbo",
            description="User description",
            expectation="User expectation",
            user_knowledge_about_app="User knowledge about app",
        ),
        metrics=[
            metric
        ]
)


class CustomChatManager(ChatManagerBase):
    user = user
    metrics = user.metrics
    prompt_version_id = ...
    def initialize_conversation(self):
        return Message()

    def next_message(self, message: Message) -> Message:
        return Message()

    def finish(self) -> bool:
        return True

    def conversation_state(self):
        return "Finished"


class CustomEvaluationManager(EvaluationManagerBase):

    def evaluate_chat(self, *args, **kwargs) -> List[EvaluationResult]:
        return [EvaluationResult(MetricDefinition("custom_test", "custom test definition"), 0.8)]

    def evaluate_raw_completion(self, *args, **kwargs) -> List[EvaluationResult]:
        return [EvaluationResult(MetricDefinition("custom_test", "custom test definition"), 0.8)]


custom_user_persona_manager = CustomChatManager(role="Human", opposite_role="AI")
custom_ai_model_manager = CustomChatManager(role="AI", opposite_role="Human")
custom_evaluation_manager = CustomEvaluationManager()

################################################ SETUP
from spelltest.ai_managers.evaluation_manager import EvaluationManager
from spelltest.ai_managers.chat_manager import SyntheticUserChatManager, AIModelDefaultChatManager
from spelltest.ai_managers.raw_completion_manager import AIModelDefaultCompletionManager, SyntheticUserCompletionManager

# Assuming you've already defined/imported FakeListLLM, CustomLLMChain, and responses

def mock_init(original_init, behavior_func):
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        behavior_func(self)
    return new_init

# Define behavior functions for each class:

def behavior_for_evaluation_manager(instance):
    instance.perfect_chat_response_key = "text"
    # mock perfect chat message
    responses = ["bla bla bla bla", "bla bla bla bla", "bla bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_perfect)
    instance.perfect_chat_chain = CustomLLMChain(llm=llm, prompt=setup_manager.perfect_chat_prompt)

    # mock perfect_completion_chain
    responses = ["bla bla bla bla", "bla bla bla bla", "bla bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_perfect)
    instance.perfect_completion_chain = CustomLLMChain(llm=llm, prompt=setup_manager.perfect_completion_prompt)

    # mock accuracy chain
    responses = ["87.0", "87.0", "87.0", "87.0", "87.0", "87.0"]
    llm = FakeListLLM(responses=responses, model_name=setup_manager.llm_name_accuracy)
    instance.accuracy_chain = CustomLLMChain(llm=llm, prompt=setup_manager.accuracy_prompt)

    # rationale chain
    rationale_responses = ["rationale"]
    rationale_llm = FakeListLLM(responses=rationale_responses, model_name=setup_manager.llm_name_rationale)
    instance.rationale_chain = CustomLLMChain(llm=rationale_llm, prompt=setup_manager.rationale_prompt)

def behavior_for_synthetic_user_chat_manager(instance):
    # Define custom behavior for SyntheticUserChatManager
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomConversationChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_ai_model_default_chat_manager(instance):
    # Define custom behavior for AIModelDefaultChatManager
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4", "Bla bla bla"]
    llm = FakeListLLM(responses=responses, model_name=instance.llm_name)
    instance.chain = CustomConversationChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_ai_model_default_completion_manager(instance):
    responses = [json.dumps({"Action": "Python REPL\nAction Input: print(2 + 2)"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomLLMChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_synthetic_user_completion_manager(instance):
    responses = [json.dumps({"Action": "Python REPL\nAction Input: print(2 + 2)"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomLLMChain(llm=llm, prompt=instance.system_prompt)

@pytest.fixture
def setup_manager():
    with patch.object(EvaluationManager, "__init__", mock_init(EvaluationManager.__init__, behavior_for_evaluation_manager)):
        with patch.object(SyntheticUserChatManager, "__init__", mock_init(SyntheticUserChatManager.__init__, behavior_for_synthetic_user_chat_manager)):
            with patch.object(AIModelDefaultChatManager, "__init__", mock_init(AIModelDefaultChatManager.__init__, behavior_for_ai_model_default_chat_manager)):
                with patch.object(AIModelDefaultCompletionManager, "__init__", mock_init(AIModelDefaultCompletionManager.__init__, behavior_for_ai_model_default_completion_manager)):
                    with patch.object(SyntheticUserCompletionManager, "__init__", mock_init(SyntheticUserCompletionManager.__init__, behavior_for_synthetic_user_completion_manager)):
                        yield manager


#################################################
@pytest.mark.parametrize("test_target,user,custom_user_persona_manager,custom_evaluation_manager", [
    (PROMPT, user, None, None),
    (PROMPT, None, custom_user_persona_manager, None),
    (custom_ai_model_manager, user, None, None),
    (PROMPT, user, None, custom_evaluation_manager),
    (custom_ai_model_manager, user, None, custom_evaluation_manager),
    (custom_ai_model_manager, None, custom_user_persona_manager, custom_evaluation_manager),
    (PROMPT, None, custom_user_persona_manager, custom_evaluation_manager),
])
def test_spelltest(test_target, user, custom_user_persona_manager, custom_evaluation_manager):
    decorator = spelltest()
    wrapper = decorator(None)
    result = wrapper(
        test_target=test_target,
        users=[user] if user else None,
        llm_name="gpt-3.5-turbo",
        openai_api_key="test key",
        size=5,
        temperature=0.8,
        chat_mode=False,
        custom_user_persona_manager=custom_user_persona_manager,
        evaluation_manager=custom_evaluation_manager,
    )

    assert result.mean_accuracy == 0.9
    assert result.deviation == 0.1
    assert result.top_999_percentile_accuracy == 0.9
    assert result.top_99_percentile_accuracy == 0.9
    assert result.top_95_percentile_accuracy == 0.9
    assert result.top_50_percentile_accuracy == 0.9
    assert len(result.simulations) == 5
