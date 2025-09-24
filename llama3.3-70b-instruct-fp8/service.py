import os
import typing as t
from typing import AsyncGenerator, Optional

import bentoml
import httpx
from annotated_types import Ge, Le
from typing_extensions import Annotated

import fastapi
openai_api_app = fastapi.FastAPI()

MAX_SESSION_LEN = int(os.environ.get("MAX_SESSION_LEN", 32*1024))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 16*1024))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 1))
PORT = int(os.environ.get("PORT", "3000"))

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"

sys_pkg_cmd = "apt-get -y update && apt-get -y install libopenmpi-dev git python3-pip libnuma-dev"
runtime_image = bentoml.images.Image(
    base_image="docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    lock_python_packages=False,
).run(sys_pkg_cmd).requirements_file("requirements.txt")

@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentosglang-llama3.3-70b-instruct-fp8-service",
    image=runtime_image,
    envs=[
        {"name": "MAX_SESSION_LEN", "value": f"{MAX_SESSION_LEN}"},
        {"name": "MAX_TOKENS", "value": f"{MAX_TOKENS}"},
        {"name": "NUM_GPUS", "value": f"{NUM_GPUS}"}
    ],
    traffic={
        "timeout": 150,
        "concurrency": 30,
    },
    resources={
        "gpu": NUM_GPUS,
        "gpu_type": "nvidia-h200-141gb",
    },
)
class SGL:

    hf_model = bentoml.models.HuggingFaceModel(
        MODEL_ID,
        exclude=['*.pth', '*.pt', 'original/**/*'],
    )

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        import sglang as sgl
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils import add_prometheus_middleware

        server_args = ServerArgs(
            model_path=self.hf_model,
            served_model_name=MODEL_ID,
            tool_call_parser="pythonic",
            context_length=MAX_SESSION_LEN,
            mem_fraction_static=0.85,
            tp_size=NUM_GPUS,
            enable_metrics=True,
        )
        self.engine = sgl.Engine(
            server_args=server_args
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

        # OpenAI endpoints
        from fastapi import Request
        from fastapi.responses import ORJSONResponse
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            CompletionRequest,
            ModelCard,
            ModelList,
        )
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )

        openai_serving_completion = OpenAIServingCompletion(
            self.engine.tokenizer_manager, self.engine.template_manager
        )
        openai_serving_chat = OpenAIServingChat(
            self.engine.tokenizer_manager, self.engine.template_manager
        )

        @openai_api_app.post("/completions")
        async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
            return await openai_serving_completion.handle_request(request, raw_request)

        @openai_api_app.post("/chat/completions")
        async def openai_v1_chat_completions(
            request: ChatCompletionRequest, raw_request: Request
        ):
            return await openai_serving_chat.handle_request(request, raw_request)

        @openai_api_app.get("/models", response_class=ORJSONResponse)
        def available_models():
            """Show available models."""
            served_model_names = [self.engine.tokenizer_manager.served_model_name]
            model_cards = []
            for served_model_name in served_model_names:
                model_cards.append(
                    ModelCard(
                        id=served_model_name,
                        root=served_model_name,
                        max_model_len=self.engine.tokenizer_manager.model_config.context_len,
                    )
                )
            return ModelList(data=model_cards)

        add_prometheus_middleware(openai_api_app)


    @bentoml.on_shutdown
    def shutdown(self):
        self.engine.shutdown()


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
        sampling_params: Optional[t.Dict[str, t.Any]] = None,
    ) -> AsyncGenerator[str, None]:

        if sampling_params is None:
            sampling_params = dict()
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        sampling_params["max_new_tokens"] = sampling_params.get("max_new_tokens", max_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        stream = await self.engine.async_generate(
            prompt, sampling_params=sampling_params, stream=True
        )

        cursor = 0
        async for request_output in stream:
            text = request_output["text"][cursor:]
            cursor += len(text)
            yield text

    async def __metrics__(self, original_metrics: str) -> str:
        async with httpx.AsyncClient() as client:
            sglang = (await client.get(f"http://127.0.0.1:{PORT}/v1/metrics")).text
        return f"{original_metrics}{sglang}"
