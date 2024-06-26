import logging
import os
import random
import threading
from typing import Optional, Literal, Any
import openai
import backoff
import dspy
import requests
from dsp.modules.gpt3 import GPT3
from dsp import ERRORS, backoff_hdlr, giveup_hdlr
from dsp.modules.hf import openai_to_hf
from dsp.modules.hf_client import (
    send_hfvllm_request_v00,
    send_hftgi_request_v01_wrapped,
)
from transformers import AutoTokenizer

try:
    from anthropic import RateLimitError
except ImportError:
    RateLimitError = None


class DSClient(GPT3):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        api_provider: Literal["openai"] = "openai",
        api_base: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"
        openai.api_type = api_provider

        self.system_prompt = system_prompt

        assert (
            api_provider != "azure"
        ), "Azure functionality with base OpenAI has been deprecated, please use dspy.AzureOpenAI instead."

        default_model_type = (
            "chat"
            if ("gpt-3.5" in model or "turbo" in model or "gpt-4" in model)
            and ("instruct" not in model)
            else "text"
        )
        self.model_type = model_type if model_type else default_model_type

        if api_key:
            openai.api_key = api_key

            openai.api_base = api_base

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions
