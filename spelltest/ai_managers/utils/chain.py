import asyncio
from typing import Any, Optional, List, Dict, Union

import openai
from langchain.callbacks.manager import Callbacks, AsyncCallbackManager, AsyncCallbackManagerForChainRun
from langchain.chains import ConversationChain, LLMChain

class CustomConversationChain(ConversationChain):
    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            if len(self.output_keys) != 1:
                raise ValueError(
                    f"`run` not supported when there is not exactly "
                    f"one output key. Got {self.output_keys}."
                )
            elif args and not kwargs:
                if len(args) != 1:
                    raise ValueError("`run` supports only one positional argument.")
                return await self.acall(
                        args[0], callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=True
                    )
            if kwargs and not args:
                return await self.acall(
                        kwargs, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=True
                    )
            raise ValueError(
                f"`run` supported with either positional arguments or keyword arguments"
                f" but not both. Got args: {args} and kwargs: {kwargs}."
            )
        except openai.error.RateLimitError as e:
            print(str(e))
            wait_and_repeat_if_limit = kwargs.get("wait_and_repeat_if_limit", 1)
            if wait_and_repeat_if_limit:
                await asyncio.sleep(wait_and_repeat_if_limit)
                kwargs["wait_and_repeat_if_limit"] = wait_and_repeat_if_limit * 2
                return await self.arun(
                    *args,
                    callbacks=callbacks,
                    tags=tags,
                    metadata=metadata,
                    **kwargs
                )
            raise openai.error.RateLimitError

class CustomLLMChain(LLMChain):
    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            if len(self.output_keys) != 1:
                raise ValueError(
                    f"`run` not supported when there is not exactly "
                    f"one output key. Got {self.output_keys}."
                )
            elif args and not kwargs:
                if len(args) != 1:
                    raise ValueError("`run` supports only one positional argument.")
                return await self.acall(
                        args[0], callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=True
                    )
            if kwargs and not args:
                return await self.acall(
                        kwargs, callbacks=callbacks, tags=tags, metadata=metadata, include_run_info=True
                    )
            raise ValueError(
                f"`run` supported with either positional arguments or keyword arguments"
                f" but not both. Got args: {args} and kwargs: {kwargs}."
            )
        except openai.error.RateLimitError as e:
            print(str(e))
            wait_and_repeat_if_limit = kwargs.get("wait_and_repeat_if_limit", 1)
            if wait_and_repeat_if_limit:
                await asyncio.sleep(wait_and_repeat_if_limit)
                kwargs["wait_and_repeat_if_limit"] = wait_and_repeat_if_limit * 2
                return await self.arun(
                    *args,
                    callbacks=callbacks,
                    tags=tags,
                    metadata=metadata,
                    **kwargs
                )
            raise openai.error.RateLimitError