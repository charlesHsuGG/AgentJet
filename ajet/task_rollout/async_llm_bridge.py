import copy
import os
import time
import uuid
from typing import (TYPE_CHECKING, Any, Awaitable, Callable, Dict, List,
                    Literal, Union)

from agentscope.model import ChatResponse as AgentScopeChatResponse
from loguru import logger
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import \
    ChatCompletion as OpenAIChatCompletion
from pydantic import BaseModel

from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker
from ajet.schema.convertion import (
    convert_llm_proxy_response_to_agentscope_response,
    convert_llm_proxy_response_to_oai_response)
from ajet.schema.logprob import TokenAndProb
from ajet.utils.tokenizer import ajet_apply_chat_template

if TYPE_CHECKING:
    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    except ImportError:
        from sglang.srt.entrypoints.openai.protocol import \
            ChatCompletionRequest

ChatResponse = Union[OpenAIChatCompletion, AgentScopeChatResponse]


class AjetStandardLlmBridgeRequest(BaseModel):
    messages: List[Dict[str, str]]
    custom_sampling_params: dict = {}
    tools: List = []
    request_id: str = ""


class AjetStandardLlmBridgeResponse(BaseModel):
    role: str = "assistant"
    request_id: str = ""
    content: str = ""
    tool_calls: List[Dict] = []
    tokens: List[TokenAndProb] = []


# -------------------------------------------------------------------------------------
# ------------------------ Unify LLM for Verl + Trinity + Vllm ------------------------
# -------------------------------------------------------------------------------------

class AsyncLlmBridge(object):
    def __init__(
        self,
        config: DictConfig,
        async_rollout_manager: Any,
        tokenizer: Any,
        llm_mode: Literal["local", "remote", "trinity"] = "local",
        max_llm_retries: int = 3,
    ):
        self.config = config
        self.async_rollout_manager = async_rollout_manager
        self.tokenizer = tokenizer
        self.llm_mode = llm_mode
        self.max_llm_retries = max_llm_retries

        self.address_mapping = {server_address: 0 for server_address in self.async_rollout_manager._server_id_to_handle.keys()}

    def get_llm_inference_fn_async(self, sampling_params: dict = {}) -> Callable:  # noqa: C901

        async def llm_chat_verl(
            messages: List[Dict[str, str]], custom_sampling_params: dict = {}, tools=[],
            request_id: str = "",
        ) -> dict:

            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            input_messages = copy.deepcopy(messages)
            # the input (prompt) sequence as text
            prompt_text = ajet_apply_chat_template(
                tokenizer=self.tokenizer,
                conversation=input_messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            # the input (prompt) sequence as input_ids
            prompt_token_ids = self.tokenizer(prompt_text)["input_ids"]

            if "max_completion_tokens" not in updated_sampling_params:
                updated_sampling_params["max_completion_tokens"] = self.config.ajet.rollout.max_response_length_in_one_turn

            updated_sampling_params.update({"logprobs": 1, "return_token_ids": True, "return_tokens_as_token_ids": True})

            server_address = min(self.address_mapping, key=self.address_mapping.get)
            client = AsyncOpenAI(base_url=f"http://{server_address}/v1", api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"))

            if tools:
                completion = await client.chat.completions.create(
                    model=self.config.ajet.model.path,
                    messages=messages,
                    tools=tools,
                    extra_body=updated_sampling_params,
                )
            else:
                completion = await client.chat.completions.create(
                    model=self.config.ajet.model.path,
                    messages=messages,
                    extra_body=updated_sampling_params,
                )

            message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            token_logprobs = []
            token_ids = completion.choices[0].token_ids if hasattr(completion.choices[0], "token_ids") else []
            logprobs = completion.choices[0].logprobs.content if completion.choices[0].logprobs else []
            if logprobs:
                for i, logprob in enumerate(logprobs):
                    token_logprobs.append(TokenAndProb(
                        token_id=token_ids[i] if i < len(token_ids) else -100,
                        logprob=logprob.logprob,    # Warning: vllm logprob does not participant training (not reliable enough), for log only.
                        decoded_string=logprob.token,
                    ))

            # sometimes tool use message has no content field
            if "content" not in message:
                message["content"] = ""

            usage = {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else None,  # type: ignore
                "total_tokens": completion.usage.total_tokens if completion.usage else None,  # type: ignore
            }

            return {
                "role": message["role"],
                "request_id": request_id,
                "content": message["content"],
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "tool_calls": message.get("tool_calls", []),
                "finish_reason": completion.choices[0].finish_reason,
                "usage": usage,
                "tokens": token_logprobs,
            }

        async def llm_chat_remote(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    # this function is defined in `ajet/backbone/main_vllm.py`
                    output_message = await self.async_rollout_manager.submit_chat_completions_async(
                        messages=input_messages,
                        sampling_params=updated_sampling_params,
                        tools=tools,
                        request_id=request_id,
                    )
                    break
                except Exception as e:
                    logger.bind(exception=True).exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)
            return output_message[-1]  # type: ignore

        async def llm_chat_trinity(
            messages: List[Dict[str, str]],
            custom_sampling_params: dict = {},
            tools=[],
            request_id: str = "",
        ) -> dict:
            async def main():
                updated_sampling_params = {}
                if sampling_params:
                    updated_sampling_params.update(sampling_params)
                if custom_sampling_params:
                    updated_sampling_params.update(custom_sampling_params)
                updated_sampling_params.pop("min_tokens")

                if tools:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        tools=tools,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                else:
                    response = await self.async_rollout_manager.chat.completions.create(
                        model=self.async_rollout_manager.model_path,
                        messages=messages,
                        logprobs=True,
                        top_logprobs=0,
                        **updated_sampling_params,
                    )
                return response

            response = await main()
            prompt_text = self.tokenizer.decode(response.model_extra["prompt_token_ids"])
            prompt_token_ids = response.model_extra["prompt_token_ids"]
            content = response.choices[0].message.content
            message = response.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)

            if content is None:
                content = ""

            if ("<tool_call>" in content) and (not message.get("tool_calls", None)):
                # logger.bind(exception=True).exception(f"Bad toolcall discovered \n\nprompt_text:\n{prompt_text}\n\nrepsonse:\n{content}")
                logger.warning(f"Bad toolcall discovered: {content}")

            tool_calls = message.get("tool_calls", [])
            max_response_length_in_one_turn = self.config.ajet.rollout.max_response_length_in_one_turn
            max_model_len: int = self.config.ajet.rollout.max_model_len
            max_seq_length: int = max_model_len - max_response_length_in_one_turn
            if len(prompt_token_ids) >= max_seq_length:
                finish_reason = "length"
            else:
                finish_reason = "stop"
            if tool_calls:
                finish_reason = "tool_calls"
            usage = {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(response.choices[0].token_ids),  # type: ignore
                "total_tokens": len(prompt_token_ids) + len(response.choices[0].token_ids),  # type: ignore
            }
            return {
                "role": "assistant",
                "request_id": response.id,
                "content": content,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "usage": usage,
                "tokens": [
                    TokenAndProb(
                        token_id=token,
                        logprob=tokenlogprob.logprob,  # Warning: vllm logprob does not participant training, for log only.
                        decoded_string=tokenlogprob.token,
                    )
                    for tokenlogprob, token in zip(
                        response.choices[0].logprobs.content,
                        response.choices[0].token_ids,
                    )
                ],
            }

        if self.llm_mode == "remote":
            return llm_chat_remote
        if self.llm_mode == "trinity":
            return llm_chat_trinity
        else:
            return llm_chat_verl


# ----------------------------------------------------------------------------------------------
# ------------------------ call async llm with context tracker (OpenAI) ------------------------
# ----------------------------------------------------------------------------------------------

class OpenaiLlmProxyWithTracker(object):
    """
    An essential wrapper to connect AsyncLlmBridge with AgentScope

    User_user_workflow <-> AsyncLlmBridge <-> Context Tracker.
    """

    def __init__(
        self,
        llm_inference_fn: Callable[..., Awaitable[Dict]],  # Callable[AjetStandardLlmBridgeRequest, AjetStandardLlmBridgeResponse]
        context_tracker: MultiAgentContextTracker,
        config,
    ) -> None:
        self.context_tracker = context_tracker
        self.llm_inference_fn = llm_inference_fn
        self.config = config

    async def chat_completion_request(
        self,
        req: "ChatCompletionRequest",
        timeline_uuid: str,
        agent_name: str,
        target_tag: str,
        episode_uuid: str,
    ):
        from openai.types.chat.chat_completion import ChatCompletion
        req_as_dict = req.model_dump(mode='json')

        # infer + process with context tracker
        llm_output = await self.run_infer(
            messages=req_as_dict["messages"],
            tools=req_as_dict["tools"],
            tool_choice="auto",
        )
        # convert to OpenAI ChatCompletion format
        response: ChatCompletion = convert_llm_proxy_response_to_oai_response(llm_output)
        # this is an important id assignment
        response.id = timeline_uuid
        assert isinstance(response, ChatCompletion)
        return response

    async def __call__(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        **kwargs,
    ) -> ChatResponse:
        llm_output = await self.run_infer(messages, tools, tool_choice, **kwargs)
        return convert_llm_proxy_response_to_oai_response(llm_output)

    async def run_infer(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",      # always auto
        **kwargs,
    ) -> Dict:
        # generate timeline uuid
        timeline_uuid = uuid.uuid4().hex

        # prepare context tracker, check context safety
        (
            context_safe,
            token_overflow,
            info,
            converted_message,
            custom_sampling_params,
            tools,
        ) = self.context_tracker.step_prepare(messages, tools, timeline_uuid=timeline_uuid)

        # if context not safe to infer further
        if not context_safe:
            logger.warning(f"[{info}] detected.")
            self.context_tracker.context_overflow = True
            if token_overflow:
                # ajet_action_when_overflow = self.config.ajet.rollout.ajet_action_when_overflow
                # cannot proceed due to context overflow
                return self.construct_overflow_response(info)
            # else:
            #     otherwise, for abnormal output, can still proceed, but we do not track output anymore

        # run llm inference ✨ (llm_chat_verl)
        llm_output = await self.llm_inference_fn(converted_message, custom_sampling_params, tools)

        # context tracking
        self.context_tracker.step_track(llm_output, context_safe, converted_message, tools, timeline_uuid=timeline_uuid)
        return llm_output

    def construct_overflow_response(self, info):
        return {
            "role": "assistant",
            "request_id": "overflow_response",
            "content": f"AgentJet: Exceeded max model context length. {info}",
            "tool_calls": None,
            "finish_reason": "length",
            "tokens": [],
        }


# ----------------------------------------------------------------------------------------------
# ------------------------ call async llm with context tracker (AgentScope) --------------------
# ----------------------------------------------------------------------------------------------

class AgentScopeLlmProxyWithTracker(OpenaiLlmProxyWithTracker):

    async def __call__(
        self,
        messages: List[dict],
        tools: List = [],
        tool_choice: str = "auto",
        structured_model=None,
        **kwargs,
    ) -> AgentScopeChatResponse:

        llm_output = await self.run_infer(messages, tools, tool_choice)
        response = convert_llm_proxy_response_to_agentscope_response(llm_output, structured_model=structured_model)
        return response
