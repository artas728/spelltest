import os
from typing import List, Union
from uuid import uuid4

from rich.console import Console

from .ai_managers.base.chat_manager import ChatManagerBase
from .ai_managers.base.raw_completion_manager import SyntheticUserRawCompletionManagerBase, \
    AIModelDefaultCompletionManagerBase
from .ai_managers.base.evaluation_manager import EvaluationManagerBase
from .ai_managers.chat_manager import SyntheticUserChatManager, AIModelDefaultChatManager, ConversationState
from .ai_managers.raw_completion_manager import AIModelDefaultCompletionManager, SyntheticUserCompletionManager
from .ai_managers.tracing.cost_calculation_tracing import CostCalculationManager, CostCalculationTracer
from .entities.general import Mode
from .entities.simulation import Simulation, ReasonType
from .entities.synthetic_user import SyntheticUser, SyntheticUserParams
from .result_processing import process_simulation_result
from .spelltest_execution import spelltest_async_together
from .ai_managers.tracing.promtelligence_tracing import PromptelligenceClient

DEFAULT_LLM = 'gpt-3.5-turbo'
IGNORE_DATA_COLLECTING = bool(os.environ.get("IGNORE_DATA_COLLECTING", "True"))
tracing_client = PromptelligenceClient(ignore=IGNORE_DATA_COLLECTING)
console = Console()

def spelltest(*args, **kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return spelltest_run_simulation(*args, **kwargs)
        return wrapper
    return decorator



def spelltest_run_simulation(
        project_name: str = "default",
        prompt: str = None,
        users: List[SyntheticUser] = None,
        openai_api_key: str = os.environ.get("OPENAI_API_KEY"),
        size: int = 1,
        temperature: float = 0.8,
        chat_mode: bool = False,
        chat_mode_max_messages: int = None,
        custom_ai_model_manager: Union[AIModelDefaultCompletionManagerBase, ChatManagerBase] = None,
        custom_user_persona_manager: Union[ChatManagerBase, SyntheticUserRawCompletionManagerBase] = None,
        custom_evaluation_manager: EvaluationManagerBase = None,
        llm_name: str = DEFAULT_LLM,
        evaluation_llm_name: str = None,
        evaluation_llm_name_perfect: str = None,
        evaluation_llm_name_rationale: str = None,
        evaluation_llm_name_accuracy: str = None,
        reason: str = ReasonType.MANUAL,
        reason_value: str = str(uuid4()),
):
    if prompt:
        default_ai_manager_cls = AIModelDefaultChatManager if chat_mode else AIModelDefaultCompletionManager
        app_manager = default_ai_manager_cls(
            target_prompt=prompt,
            llm_name=llm_name,
            openai_api_key=openai_api_key,
            temperature=temperature,
            role="AI",
            opposite_role="Human"
        )
    elif custom_ai_model_manager:
        if not isinstance(custom_ai_model_manager, ChatManagerBase) and \
                not isinstance(custom_ai_model_manager, AIModelDefaultCompletionManagerBase):
            raise Exception("Since test_target is custom class, "
                            "it is expected to be child of ChatManagerBase or AIModelDefaultCompletionManagerBase")
        if users:
            raise Exception("You can't pass users with custom AI model as test_target")
        app_manager = custom_ai_model_manager
    else:
        raise Exception("There are 'prompt' or 'custom_ai_model_manager' parameters required to run simulation")
    if not users:
        if not isinstance(custom_user_persona_manager, ChatManagerBase) and not \
                isinstance(custom_user_persona_manager, SyntheticUserRawCompletionManagerBase):
            raise Exception("Since there is no users, "
                            "`custom_user_persona_manager`is expected "
                            "(based on ChatManagerBase or SyntheticUserRawCompletionManagerBase) ")
        user_persona_managers = [custom_user_persona_manager]
    else:
        user_persona_managers = []
        for user in users:
            if chat_mode:
                user_persona_manager = SyntheticUserChatManager(
                    user=user,
                    openai_api_key=openai_api_key,
                )
            else:
                user_persona_manager = SyntheticUserCompletionManager(
                    target_prompt=prompt,
                    user=user,
                    openai_api_key=openai_api_key,
                )
            user_persona_managers.append(user_persona_manager)

    simulation_result: List[Simulation] = spelltest_async_together(
        target_prompt=prompt,
        app_manager=app_manager,
        user_persona_managers=user_persona_managers,
        evaluation_manager=custom_evaluation_manager,
        size=size,
        mode=Mode.CHAT if chat_mode else Mode.RAW_COMPLETION,
        chat_mode_max_messages=chat_mode_max_messages,
        openai_api_key=openai_api_key,
        llm_name=llm_name,
        evaluation_llm_name=evaluation_llm_name,
        evaluation_llm_name_perfect=evaluation_llm_name_perfect,
        evaluation_llm_name_rationale=evaluation_llm_name_rationale,
        evaluation_llm_name_accuracy=evaluation_llm_name_accuracy,
        console=console,
    )
    return process_simulation_result(
        project_name=project_name,
        simulations=simulation_result,
        llm_name=llm_name,
        size=size,
        chat_mode=chat_mode,
        temperature=temperature,
        reason=reason,
        reason_value=reason_value
    )