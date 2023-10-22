import atexit
import os
import requests
import tiktoken
import uuid
from abc import ABC
from typing import Any, Dict, List, Optional
from langchain import Prompt as DefaultPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING
from langchain.schema import LLMResult
from pydantic import BaseModel, Field


client = None   # TODO:  refactor this


class PromptelligenceParams(BaseModel):
    db_id: Optional[int] = None
    db_version_id: Optional[int] = None
    version: Optional[str] = None
    accuracy_threshold: Optional[float] = None
    custom_definition_accuracy: Optional[str] = None


class PromptVersionUsageLog(BaseModel):
    prompt_version_id: int
    run_id: str
    parent_run_id: str
    llm_name: str
    prompt: str
    invocation_params: Dict


class LLMUsageLog(BaseModel):
    prompt_version_id: str
    run_id: str
    parent_run_id: str
    llm_name: str
    completion: str
    generation_info: Optional[Dict] = None
    tokens_total: Optional[int] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None


class PromptTemplate(DefaultPromptTemplate):
    """Use this one instead default PromptTemplate"""

    alias: str = None
    promptelligence_params: PromptelligenceParams = Field(default_factory=PromptelligenceParams)
    parent_alias: str = None

    def __init__(self, **kwargs):
        alias_error = "You must define one between`alias` or `parent_alias` but not both"
        alias = kwargs.get("alias")
        parent_alias = kwargs.get("parent_alias")
        assert not type(alias) is type(parent_alias), alias_error
        if client is None:
            raise Exception(
                "You have to initialize PromptelligenceClient before calling any PromptTemplate"
            )
        super().__init__(**kwargs)
        if not client.ignore_tracing:
            client.sync_prompt(self)

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs and log the usage to LangChainClient."""
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        formatted_prompt = DEFAULT_FORMATTER_MAPPING[self.template_format](
            self.template, **kwargs
        )
        return formatted_prompt


class PromptelligenceClient:
    def __init__(
        self,
        project: str = 'Default',
        api_key: str or None = None,
        environment: str or None = None,
        base_url: str = "http://127.0.0.1:8000",
        ignore: bool = False
    ):
        self.project_name = project
        self.environment = environment
        if ignore:
            self.ignore_tracing = True
        else:
            self.ignore_tracing = False
            self.base_url = base_url
            if not api_key:
                api_key = os.environ.get("PROMTELLIGENCE_API_KEY")
            self.header = {
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {api_key}",
            }
            self._session = requests.Session()
            atexit.register(self._cleanup)
            self._get_or_create_project()
            self.uploaded_prompts = {}
            self._upload_prompts()

        global client
        client = self

    def sync_prompt(self, prompt: PromptTemplate):
        if prompt.alias and prompt.alias not in self.uploaded_prompts:
            self._register_new_prompt(prompt)
        elif prompt.parent_alias and prompt.parent_alias in self.uploaded_prompts:
            self._sync_with_parent_prompt(prompt)
        else:
            self._sync_prompt(prompt)

    def delete_prompt(self, prompt: PromptTemplate):
        response = self._session.delete(
            self.base_url + f"/api/prompts/{prompt.promptelligence_params.db_id}/",
            headers=self.header,
        )
        response.raise_for_status()
        del self.uploaded_prompts[prompt.alias]

    def send_prompt_version_usage_log(self, usage_log: PromptVersionUsageLog):
        response = self._session.post(
            self.base_url + "/api/prompt-version-usage-logs/",
            json=usage_log.dict(),
            headers=self.header,
        )
        response.raise_for_status()
        return response.json()["id"]

    def send_llm_usage_log(self, usage_log: LLMUsageLog):
        response = self._session.post(
            self.base_url + "/api/llm-usage-logs/",
            json=usage_log.dict(),
            headers=self.header,
        )
        response.raise_for_status()
        return response.json()["id"]

    def _sync_prompt(self, prompt: PromptTemplate):
        if not prompt.promptelligence_params.db_id:
            prompt.promptelligence_params.db_id = self.uploaded_prompts[prompt.alias]["id"]
            prompt.promptelligence_params.db_version_id = self.uploaded_prompts[
                prompt.alias
            ]["last_version"]["id"]
        if self._is_prompt_changed(prompt):
            self._register_new_prompt_version(prompt)

    def _sync_with_parent_prompt(self, prompt):
        if not prompt.promptelligence_params.db_id:
            prompt.promptelligence_params.db_id = self.uploaded_prompts[prompt.parent_alias]["id"]
            prompt.promptelligence_params.db_version_id = self.uploaded_prompts[
                prompt.parent_alias
            ]["last_version"]["id"]

    def _get_or_create_project(self):
        params = {
            "name": self.project_name,
            "environment": self.environment if self.environment else "dev",
        }
        response = self._session.get(
            self.base_url + "/api/projects/", params=params, headers=self.header
        )
        response.raise_for_status()
        result = response.json()
        if len(result) == 0:
            response = self._session.post(
                self.base_url + "/api/projects/", json=params, headers=self.header
            )
            response.raise_for_status()
            result = response.json()
        else:
            if len(result) == 1:
                Exception(f"There is more then one project with name {self.project_name}")
            result = result[0]
        self.project_id = result["id"]

    def _upload_prompts(self):
        response = self._session.get(
            self.base_url + "/api/prompts/",
            params={"project_id": self.project_id},
            headers=self.header,
        )
        response.raise_for_status()
        prompts = response.json()
        for i in prompts:
            i["last_version"] = self._get_last_prompt_version(i["id"])
            self.uploaded_prompts[i["alias"]] = i

    def _get_last_prompt_version(self, prompt_id):
        response = self._session.get(
            self.base_url + f"/api/prompt-versions/{prompt_id}/latest/",
            headers=self.header,
        )
        response.raise_for_status()
        return response.json()

    def _is_prompt_changed(self, prompt: PromptTemplate):
        return (
            prompt.template
            != self.uploaded_prompts[prompt.alias]["last_version"]["template_body"]
        )

    def _register_new_prompt_version(
        self, prompt: PromptTemplate, first_version: bool = False
    ):
        if first_version:
            prompt.promptelligence_params.version = "v1"
        else:
            uploaded_prompt_version = self.uploaded_prompts[prompt.alias][
                "last_version"
            ]
            splitted_old_version = uploaded_prompt_version[
                "version_or_git_commit"
            ].split("v")
            if splitted_old_version[0] == "" and splitted_old_version[-1].isdigit():
                new_version = f"v{int(splitted_old_version[-1])+1}"
            else:
                new_version = str(uuid.uuid4())[:10]
            prompt.promptelligence_params.version = new_version
        data = {
            "prompt_id": prompt.promptelligence_params.db_id,
            "version_or_git_commit": prompt.promptelligence_params.version,
        }
        if len(prompt.template) > 0:
            data["template_body"] = prompt.template
        response = self._session.post(
            self.base_url + "/api/prompt-versions/", json=data, headers=self.header
        )
        try:
            response.raise_for_status()
        except Exception as e:
            print(str(e))
            raise e
        result = response.json()
        self.uploaded_prompts[prompt.alias]["last_version"] = result
        prompt.promptelligence_params.db_version_id = result["id"]

    def _register_new_prompt(self, prompt: PromptTemplate):
        data = {
            "project_id": self.project_id,
            "name": prompt.alias,
            "alias": prompt.alias,
            "accuracy_threshold": prompt.promptelligence_params.accuracy_threshold,
            "custom_definition_accuracy": prompt.promptelligence_params.custom_definition_accuracy,
        }
        response = self._session.post(
            self.base_url + "/api/prompts/", json=data, headers=self.header
        )
        response.raise_for_status()
        result = response.json()
        prompt.promptelligence_params.db_id = result["id"]
        if prompt.alias not in self.uploaded_prompts:
            self.uploaded_prompts[prompt.alias] = result
        self._register_new_prompt_version(prompt, first_version=True)

    def _cleanup(self):
        self._session.close()


class PromptelligenceTracer(BaseCallbackHandler, ABC):
    """Base interface for tracers."""

    def __init__(self, prompt):
        self.prompt = prompt
        self.prompt_version_id = prompt.promptelligence_params.db_version_id

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id,
        parent_run_id: Optional = None,
        **kwargs: Any,
    ) -> Any:
        if client.ignore_tracing:
            return
        if bool(os.environ.get("PYTEST_RUN_CONFIG", 'False')) is True:
            kwargs["invocation_params"]["model_name"] = "text-davinci-003"
        for prompt in prompts:
            client.send_prompt_version_usage_log(
                PromptVersionUsageLog(
                    prompt_version_id=self.prompt_version_id,
                    llm_name=kwargs["invocation_params"]["model_name"],
                    run_id=str(run_id),
                    parent_run_id=str(parent_run_id),
                    invocation_params=kwargs["invocation_params"],
                    prompt=prompt,
                )
            )

    def on_llm_end(
        self, response: LLMResult, run_id, parent_run_id: Optional = None, **kwargs: Any
    ) -> None:
        """End a trace for an LLM run."""
        if client.ignore_tracing:
            return
        if bool(os.environ.get("PYTEST_RUN_CONFIG", 'False')) is True and response.llm_output is None:
            response.llm_output = {"model_name": "text-davinci-003", "token_usage": None}
        previous_generations_token_length = 0
        llm_usage_log_ids = []
        for generation_group in response.generations:
            for generation in generation_group:
                if not response.llm_output["token_usage"]:
                    # TODO: validate this approach calculates tokens properly
                    previous_generations_token_length += self._calculate_token_len(
                        model_name=response.llm_output["model_name"],
                        text=generation.text
                    )
                    response.llm_output["token_usage"] = {
                        "completion_tokens": previous_generations_token_length,
                    }
                    llm_usage_log = LLMUsageLog(
                        prompt_version_id=self.prompt_version_id,
                        llm_name=response.llm_output["model_name"],
                        run_id=str(run_id),
                        parent_run_id=str(parent_run_id),
                        completion=generation.text,
                        tokens_completion=response.llm_output["token_usage"][
                            "completion_tokens"
                        ],
                        generation_info=generation.generation_info,
                    )
                else:
                    llm_usage_log = LLMUsageLog(
                        prompt_version_id=self.prompt_version_id,
                        llm_name=response.llm_output["model_name"],
                        run_id=str(run_id),
                        parent_run_id=str(parent_run_id),
                        completion=generation.text,
                        tokens_total=response.llm_output["token_usage"]["total_tokens"],
                        tokens_prompt=response.llm_output["token_usage"][
                            "prompt_tokens"
                        ],
                        tokens_completion=response.llm_output["token_usage"][
                            "completion_tokens"
                        ],
                        generation_info=generation.generation_info,
                    )
                llm_usage_log_id = client.send_llm_usage_log(llm_usage_log)
                llm_usage_log_ids.append(llm_usage_log_id)
        response.Config.fields["llm_usage_log_ids"] = llm_usage_log_ids

    def _calculate_token_len(self, model_name, text):
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

Prompt = PromptTemplate
