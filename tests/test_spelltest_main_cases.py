import json
import uuid
import pytest
from typing import List
from unittest.mock import patch
from spelltest.ai_managers.utils.chain import CustomConversationChain
from spelltest.entities.managers import MessageType
from spelltest.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from spelltest.spelltest import spelltest
from spelltest.ai_managers.evaluation_manager import EvaluationManager, EvaluationManagerBase, EvaluationResult
from spelltest.ai_managers.chat_manager import SyntheticUserChatManager, AIModelDefaultChatManager, ChatManagerBase
from spelltest.ai_managers.raw_completion_manager import AIModelDefaultCompletionManager, AIModelDefaultCompletionManagerBase, \
    SyntheticUserCompletionManager, SyntheticUserRawCompletionManagerBase
from spelltest.ai_managers.chat_manager import Message
from spelltest.ai_managers.raw_completion_manager import CustomLLMChain
from spelltest.ai_managers.tracing.promtelligence_tracing import PromptTemplate as TracedPromptTemplate
from langchain.llms.fake import FakeListLLM as DefaultFakeListLLM


class FakeListLLM(DefaultFakeListLLM):
    async def _acall(
        self,
        prompt,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> str:
        """Return next response"""
        response = self.responses[0]
        self.i += 1
        return response


PROMPT_CHAT = "Write personalized sales email according user requirements"
PROMPT_COMPLETION = "Write personalized sales email according user requirements {Action}"

metric = MetricDefinition(
    name="metric name",
    definition="metric expectation",
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

class CustomAIModelChatManager(ChatManagerBase):
    def __init__(self, *args, **kwargs):
        self.user = user
        self.metrics = [metric]
        self.chat_id = str(uuid.uuid4())
        self.target_prompt = TracedPromptTemplate(
            template=PROMPT_CHAT,
            input_variables=[],
            alias='pytestCustomAIModelChatManager'
        )
        super().__init__(*args, **kwargs)

    async def initialize_conversation(self, initialize_conversation: Message = None) -> Message:
        return Message(author=MessageType.ASSISTANT, text="Hihi! How can I help you?", run_id=str(uuid.uuid4()))

    async def next_message(self, message: Message) -> Message:
        return Message(author=MessageType.ASSISTANT, text="This is next AI message for you", run_id=str(uuid.uuid4()))

    def finish(self) -> bool:
        return True

    def conversation_state(self):
        return "Finished"

class CustomAIModelCompletionManager(AIModelDefaultCompletionManagerBase):
    def __init__(self, *args, **kwargs):
        self.user = user
        self.metrics = [metric]
        self.target_prompt = TracedPromptTemplate(
            template=PROMPT_COMPLETION,
            input_variables=['Action',],
            alias='pytestCustomAIModelCompletionManager'
        )
        super().__init__(*args, **kwargs)
    async def generate_completion(self, user_input):
        return Message(author=MessageType.ASSISTANT, text="Hi! This is an AI message for you", run_id=str(uuid.uuid4()))



class CustomUserRawCompletionManager(SyntheticUserRawCompletionManagerBase):
    def __init__(self, *args, **kwargs):
        self.user = user
        self.metrics = [metric]
        super().__init__(*args, **kwargs)
    async def generate_user_input(self):
        return Message(author=MessageType.USER, text=json.dumps({"Action": "This is user message for you"}), run_id=str(uuid.uuid4()))


class CustomEvaluationManager(EvaluationManagerBase):

    async def evaluate_chat(self, *args, **kwargs) -> List[EvaluationResult]:
        return [EvaluationResult(MetricDefinition("custom_test", "custom test definition"), 0.8, 0.1, "rationale")]

    async def evaluate_raw_completion(self, *args, **kwargs) -> List[EvaluationResult]:
        return [EvaluationResult(MetricDefinition("custom_test", "custom test definition"), 0.8, 0.1, "rationale")]

custom_user_persona_chat_manager = CustomAIModelChatManager(role="Human", opposite_role="AI")
custom_ai_model_chat_manager = CustomAIModelChatManager(role="AI", opposite_role="Human")

custom_user_persona_raw_completion_manager = CustomUserRawCompletionManager()
custom_ai_model_raw_completion_manager = CustomAIModelCompletionManager()

custom_evaluation_manager = CustomEvaluationManager()



def mock_init(original_init, behavior_func):
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        behavior_func(self)
    return new_init

# Define behavior functions for each class:

def behavior_for_evaluation_manager(instance):
    instance.perfect_chat_response_key = "text"
    # # mock perfect chat message
    # responses = ["bla bla bla bla"]
    # llm = FakeListLLM(responses=responses, model_name=instance.llm_name_perfect)
    # instance.perfect_chat_chain = CustomLLMChain(llm=llm, prompt=instance.perfect_chat_prompt)
    #
    # # mock perfect_completion_chain
    # responses = ["bla bla bla bla"]
    # llm = FakeListLLM(responses=responses, model_name=instance.llm_name_perfect)
    # instance.perfect_completion_chain = CustomLLMChain(llm=llm, prompt=instance.perfect_completion_prompt)
    instance.enable_cost_tracker_layer()
    instance._init_perfect_chain()
    instance._init_rationale_chain()
    instance._init_accuracy_chain()
    # mock accuracy chain
    responses = ["0.78"]
    llm = FakeListLLM(responses=responses, model_name=instance.llm_name_accuracy)
    instance.accuracy_chain = CustomLLMChain(llm=llm, prompt=instance.accuracy_prompt)

    # rationale chain
    rationale_responses = ["rationale"]
    rationale_llm = FakeListLLM(responses=rationale_responses, model_name=instance.llm_name_rationale)
    instance.rationale_chain = CustomLLMChain(llm=rationale_llm, prompt=instance.rationale_prompt)

def behavior_for_synthetic_user_chat_manager(instance):
    # Define custom behavior for SyntheticUserChatManager
    responses = ["synthetic user chat manager llm fake output" for _ in range(1)]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomConversationChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_ai_model_default_chat_manager(instance):
    # Define custom behavior for AIModelDefaultChatManager
    responses = ["Ai default chat manager llm fake output"]
    llm = FakeListLLM(responses=responses, model_name=instance.llm_name)
    instance.chain = CustomConversationChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_ai_model_default_completion_manager(instance):
    responses = [json.dumps({"Action": "Ai default completion manager llm fake output"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomLLMChain(llm=llm, prompt=instance.system_prompt)

def behavior_for_synthetic_user_completion_manager(instance):
    responses = [json.dumps({"Action": "Synthetic user default completion manager llm fake output"})]
    llm = FakeListLLM(responses=responses, model_name=user.params.llm_name)
    instance.chain = CustomLLMChain(llm=llm, prompt=instance.system_prompt)

@pytest.fixture
def setup_manager():
    with patch.object(EvaluationManager, "__init__", mock_init(EvaluationManager.__init__, behavior_for_evaluation_manager)), \
            patch.object(EvaluationManager, "initialize_evaluation",
                         mock_init(EvaluationManager.initialize_evaluation, behavior_for_evaluation_manager)), \
            patch.object(SyntheticUserChatManager, "__init__", mock_init(SyntheticUserChatManager.__init__, behavior_for_synthetic_user_chat_manager)), \
         patch.object(AIModelDefaultChatManager, "__init__", mock_init(AIModelDefaultChatManager.__init__, behavior_for_ai_model_default_chat_manager)), \
         patch.object(AIModelDefaultCompletionManager, "__init__", mock_init(AIModelDefaultCompletionManager.__init__, behavior_for_ai_model_default_completion_manager)), \
         patch.object(SyntheticUserCompletionManager, "__init__", mock_init(SyntheticUserCompletionManager.__init__, behavior_for_synthetic_user_completion_manager)):
        yield


#################################################
@pytest.mark.parametrize("test_target,user,custom_ai_model_manager,custom_user_persona_chat_manager,custom_evaluation_manager", [
    (PROMPT_CHAT, user, None, None, None),
    (PROMPT_CHAT, None, None, custom_user_persona_chat_manager, None),
    (None, None, custom_ai_model_chat_manager, custom_user_persona_chat_manager, None),
    (PROMPT_CHAT, user, None, None, custom_evaluation_manager),
    (None, None, custom_ai_model_chat_manager, custom_user_persona_chat_manager, custom_evaluation_manager),
    (PROMPT_CHAT, None, None, custom_user_persona_chat_manager, custom_evaluation_manager),
])
def test_spelltest_chat(setup_manager, test_target, user, custom_ai_model_manager, custom_user_persona_chat_manager, custom_evaluation_manager):
    decorator = spelltest()
    wrapper = decorator(None)
    result = wrapper(
        prompt=test_target,
        users=[user] if user else None,
        llm_name="gpt-3.5-turbo",
        openai_api_key="test key",
        size=5,
        temperature=0.8,
        chat_mode=True,
        custom_ai_model_manager=custom_ai_model_manager,
        custom_user_persona_manager=custom_user_persona_chat_manager,
        custom_evaluation_manager=custom_evaluation_manager,
    )



@pytest.mark.parametrize("test_target,user,custom_ai_model_manager,custom_user_persona_raw_completion_manager,custom_evaluation_manager", [
    (PROMPT_COMPLETION, user, None, None, None),
    (PROMPT_COMPLETION, None, None, custom_user_persona_raw_completion_manager, None),
    (PROMPT_COMPLETION, user, None, None, custom_evaluation_manager),
    (None, None, custom_ai_model_raw_completion_manager, custom_user_persona_raw_completion_manager, custom_evaluation_manager),
    (PROMPT_COMPLETION, None, None, custom_user_persona_raw_completion_manager, custom_evaluation_manager),
])
def test_spelltest_raw_completion(setup_manager, test_target, user, custom_ai_model_manager, custom_user_persona_raw_completion_manager, custom_evaluation_manager):
    decorator = spelltest()
    wrapper = decorator(None)
    result = wrapper(
        prompt=test_target,
        users=[user] if user else None,
        llm_name="gpt-3.5-turbo",
        openai_api_key="test key",
        size=5,
        temperature=0.8,
        chat_mode=False,
        custom_ai_model_manager=custom_ai_model_manager,
        custom_user_persona_manager=custom_user_persona_raw_completion_manager,
        custom_evaluation_manager=custom_evaluation_manager,
    )

