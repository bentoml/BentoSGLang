import typing as t
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

MAX_MODEL_LEN = 4096
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@bentoml.service(
    name="bentosglang-llama3.1-8b-instruct-service",
    traffic={
        "timeout": 1200,
        "concurrency": 256,  # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class SGL:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        import sglang as sgl

        self.engine = sgl.Engine(
            model_path=MODEL_ID,
            context_length=(MAX_MODEL_LEN - MAX_TOKENS),
            mem_fraction_static=0.85,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    @bentoml.on_shutdown
    def shutdown(self):
        from sglang.srt.utils import kill_child_process
        kill_child_process()

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

        async for request_output in stream:
            text = request_output["text"]
            yield text
