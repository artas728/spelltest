import copy
import os
import asyncio
import json
import time
import uuid
from typing import List

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn
import httpx
from dataclasses import dataclass, asdict, field
from .ai_managers.chat_manager import ConversationState
from .ai_managers.evaluation_manager import EvaluationManager
from .entities.general import Mode
from .entities.managers import EvaluationResult
from .entities.simulation import Simulation, ChatSimulationMessageStorage, CompletionSimulationMessageStorage

CHAT_MAX_MESSAGES_DEFAULT = 6
SPELLFORGE_HOST = os.environ.get("SPELLFORGE_HOST", "http://spellforge.ai/")
SPELLFORGE_API_KEY = os.environ.get("SPELLFORGE_API_KEY")

def spelltest_async_together(
    target_prompt,
    app_manager,
    user_persona_managers,
    evaluation_manager,
    size,
    mode,
    chat_mode_max_messages,
    openai_api_key,
    llm_name,
    evaluation_llm_name,
    evaluation_llm_name_perfect,
    evaluation_llm_name_rationale,
    evaluation_llm_name_accuracy,
) -> List[Simulation]:
    console = Console()
    progress = Progress(TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TextColumn("[progress.completed]{task.completed} of {task.total}"),
        transient=True,)
    with progress:
        console.print("🚀 Starting simulations!", style="bold red")
        evaluation_llm_name = evaluation_llm_name if evaluation_llm_name else llm_name
        tasks = []
        for sim_num in range(size):
            for user_persona_manager in user_persona_managers:
                if not evaluation_manager:
                    evaluation_manager = EvaluationManager(
                        metric_definitions=user_persona_manager.metrics,
                        openai_api_key=openai_api_key,
                        synthetic_user_persona_manager=user_persona_manager,
                        llm_name_default=evaluation_llm_name,
                        llm_name_perfect=evaluation_llm_name_perfect if evaluation_llm_name_perfect else evaluation_llm_name,
                        llm_name_rationale=evaluation_llm_name_rationale if evaluation_llm_name_rationale else evaluation_llm_name,
                        llm_name_accuracy=evaluation_llm_name_accuracy if evaluation_llm_name_accuracy else evaluation_llm_name,
                    )
                console_render_task_id = progress.add_task(f"[cyan]Simulating({sim_num})...", total=3)
                tasks.append(_asimulate(app_manager,
                                        user_persona_manager,
                                        evaluation_manager,
                                        mode,
                                        chat_mode_max_messages,
                                        progress,
                                        console_render_task_id,
                                        sim_num
                                        ))
        loop = asyncio.get_event_loop()
        simulations = loop.run_until_complete(asyncio.gather(*tasks))
        # console.print("Debug: All simulations reported done, clearing console and printing final message")
        time.sleep(1)  # wait for 1 second
        console.clear()
        console.print("🏁 Simulations finished!", style="bold green")
        return simulations


async def _asimulate(
        app_manager,
        user_persona_manager,
        evaluation_manager,
        mode,
        chat_mode_max_messages,
        progress,
        console_render_task_id,
        sim_num
):
    app_manager = copy.deepcopy(app_manager)
    user_persona_manager = copy.deepcopy(user_persona_manager)
    evaluation_manager = copy.deepcopy(evaluation_manager)
    if mode is Mode.CHAT:
        progress.update(console_render_task_id, advance=1, description=f" Simulation {sim_num}, Step 2: [bold]Generating Chat[/bold] ⏳ ")
        chat_history = await _generate_chat(app_manager, user_persona_manager, chat_mode_max_messages)
        progress.update(console_render_task_id, advance=1, description=f"Simulation {sim_num}, Step 3: [bold]Evaluation[/bold] ⏳ ")
        evaluations: List[EvaluationResult] = await evaluation_manager.evaluate_chat(
            chat_history,
            user_persona_manager
        )
        progress.update(console_render_task_id, advance=1, description=f"Simulation {sim_num}, [bold]Simulation completed[/bold] ✅ ")
        return Simulation(
            prompt_version_id=app_manager.target_prompt.promptelligence_params.db_version_id,
            app_user_persona_id=user_persona_manager.user.db_id,
            run_ids=[message.run_id for message in chat_history if message.run_id is not None],
            length_complexity=0.0,   # TODO
            chat_id=user_persona_manager.chat_id,
            evaluations=evaluations,
            granular_evaluation=False,
            message_storage=ChatSimulationMessageStorage(
                chat_history=chat_history,
                perfect_chat_history=evaluation_manager.perfect_chat_history
                if hasattr(evaluation_manager, "perfect_chat_history") else None,
            )
        )
    elif mode is Mode.RAW_COMPLETION:
        progress.update(console_render_task_id, advance=0, description=f" Simulation {sim_num}, Step 2: [bold]Generating Completion[/bold] ⏳ ")
        prompt, completion = await _generate_raw_completion(app_manager, user_persona_manager)
        progress.update(console_render_task_id, advance=1, description=f"Simulation {sim_num}, Step 3: [bold]Evaluation[/bold] ⏳ ")
        evaluations: List[EvaluationResult] = await evaluation_manager.evaluate_raw_completion(
            prompt,
            completion,
            # app_manager,
            user_persona_manager)
        progress.update(console_render_task_id, advance=2, description=f"Simulation {sim_num}, [bold]Simulatoin completed[/bold] ✅ ")
        return Simulation(
            prompt_version_id=app_manager.prompt_version_id,    #app_manager.target_prompt.promptelligence_params.db_version_id,
            app_user_persona_id=user_persona_manager.user.db_id,
            run_ids=[prompt.run_id, completion.run_id],
            length_complexity=0.0,  # TODO
            chat_id=None,
            evaluations=evaluations,
            granular_evaluation=False,
            message_storage=CompletionSimulationMessageStorage(
                prompt=prompt,
                completion=completion,
                perfect_completion=evaluation_manager.perfect_completion
                if hasattr(evaluation_manager, "perfect_completion") else None,
            )
        )
    else:
        raise Exception(f"Unexpected mode: {mode}")


async def _generate_chat(app_chat_manager, user_persona_manager, max_messages):
    app_message = await app_chat_manager.initialize_conversation()
    user_message = await user_persona_manager.initialize_conversation(app_message)
    message_count = 0
    if not max_messages:
        max_messages = CHAT_MAX_MESSAGES_DEFAULT
    # Continuously call "next_message" function between two sides until UserPersonaManager decides that conversation is over
    while user_persona_manager.conversation_state() is not ConversationState.FINISHED and message_count < max_messages:
        app_message = await app_chat_manager.next_message(user_message)
        user_message = await user_persona_manager.next_message(app_message)
        message_count+=2
    return user_persona_manager.chat_history  # TODO: make local history implementation in order to not count on custom implementations

async def _generate_raw_completion(app_completion_manager, user_persona_manager):
    user_input_message = await user_persona_manager.generate_user_input()
    user_input = json.loads(user_input_message.text)
    return user_input_message, await app_completion_manager.generate_completion(user_input)