from ..my_app import app.langchain_chain as chain
from ..src.synthetic_user import SyntheticUser, SyntheticUserParams, MetricDefinition
from ..src.spelltest import spelltest
from ..src.chat_manager import ChatManagerBase
from ..src.evaluation_manager import EvaluationManager, EvaluationManagerBase, EvaluationResult
from ..src.chat_manager import Message


user = SyntheticUser(
        params=SyntheticUserParams(
            temperature=...,
            llm_name=...,
            description=...,
            expectation=...,
            user_knowledge_about_app=...,
        ),
        metrics=[
            MetricDefinition(
                name=...,
                definition=...,
            )
        ]
    )





@spelltest(
        test_target=chain,
        user=user,
        llm_name="gpt-3.5-turbo",
        size=5,
        temperature=0.8,
        chat_mode=False,
    )
def spelltest_result_function(simulation_result):
    pass