import os
import tiktoken
from abc import ABC
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.box import Box

invisible_box = Box(
    "    \n"  # top
    "    \n"  # head
    "    \n"  # head_row
    "    \n"  # mid
    "    \n"  # row
    "    \n"  # foot_row
    "    \n"  # foot
    "    "    # bottom (no newline character at the end)
)

class CostCalculationManager:
    _instance = None

    def __new__(cls, console=None):
        if cls._instance is None:
            if console is None:
                raise Exception("You have to pass 'console' object when you initialize 'CostCalculationDisplay' class first time")
            cls._instance = super(CostCalculationManager, cls).__new__(cls)
            cls._instance.cost_usd = 0.0
            cls._instance.live = Live(cls._instance._render(), console=console, auto_refresh=True)
            cls._instance.update_cost()
        return cls._instance

    def _render(self):
        return Panel(Text(f"Cost: ${self.cost_usd:.8f}", justify="right"),
                     box=invisible_box,
                     padding=(1, 2),
                     )

    def update_cost(self):
        self.live.update(self._render())


class CostCalculationTracer(BaseCallbackHandler, ABC):
    """Tracing for each individual model."""
    PRICE_MAP = {  # per 1000 tokens
            "gpt-3.5-turbo-15k": {"prompt": 0.003, "completion": 0.004},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-davinci-003": {"prompt": 0.002, "completion": 0.002},
            "text-babbage-001": {"prompt": 0.002, "completion": 0.0024},
            "text-curie-001": {"prompt": 0.002, "completion": 0.0120},
            "text-ada-001": {"prompt": 0.002, "completion": 0.0016},
            "text-embedding-ada-002": {"prompt": 0.002, "completion": 0.0004},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},  # 8k token size
            "gpt-4-32k": {"prompt": 0.06, "completion": 0.12}  # 32k token size
    }

    def __init__(self):
        self.cost_manager = CostCalculationManager()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id,
        parent_run_id: Optional = None,
        **kwargs: Any,
    ) -> Any:
        model_name = kwargs["invocation_params"]["model_name"]
        for prompt in prompts:
            token_len = self._calculate_token_len(model_name, prompt)
            self.cost_manager.cost_usd += self._calculate_cost_usd(model_name, token_len, "prompt")
            self.cost_manager.update_cost()
    def on_llm_end(
        self, response: LLMResult, run_id, parent_run_id: Optional = None, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        for generation_group in response.generations:
            for generation in generation_group:
                if not response.llm_output["token_usage"]:
                    model_name = kwargs["invocation_params"]["model_name"]
                    # TODO: validate this approach calculates tokens properly
                    token_length = self._calculate_token_len(
                        model_name=model_name,
                        text=generation.text
                    )
                else:
                    model_name = response.llm_output["model_name"]
                    token_length = response.llm_output["token_usage"][
                            "completion_tokens"
                        ]
                self.cost_manager.cost_usd += self._calculate_cost_usd(model_name, token_length, "completion")
                self.cost_manager.update_cost()
    def _calculate_token_len(self, model_name, text):
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def _calculate_cost_usd(self, model_name, token_len, side):
        return token_len * (self.PRICE_MAP[model_name][side] / 1000)
