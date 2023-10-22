from ..spelltest.entities.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from ..spelltest.spelltest import spelltest
from ..spelltest.ai_managers.base.chat_manager import ChatManagerBase
from ..spelltest.ai_managers.evaluation_manager import EvaluationManager, EvaluationManagerBase, EvaluationResult
from ..spelltest.ai_managers.chat_manager import Message


class CustomChatManager(ChatManagerBase):
    def initialize_conversation(self):
        return Message()

    def next_message(self, message: Message) -> Message:
        return Message()

    def finish(self) -> bool:
        return True

    def conversation_state(self):
        return "Finished"


class CustomEvaluationManager(EvaluationManagerBase):
    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        return EvaluationResult(MetricDefinition("custom_test", "custom test definition"), 0.8)




custom_user_persona_manager = CustomChatManager(role="Human", opposite_role="AI")
custom_ai_model_manager = CustomChatManager(role="AI", opposite_role="Human")
custom_evaluation_manager = CustomEvaluationManager()





@spelltest(
        llm_name="gpt-3.5-turbo",
        size=5,
        temperature=0.8,
        chat_mode=False,
        custom_ai_model_manager=custom_ai_model_manager,
        custom_user_persona_manager=custom_user_persona_manager,
        evaluation_manager=custom_evaluation_manager,
    )
def spelltest_result_function(simulation_result):
    pass