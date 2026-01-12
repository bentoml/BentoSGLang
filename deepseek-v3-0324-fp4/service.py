import os
import typing as t

import bentoml

MAX_SESSION_LEN = int(os.environ.get("MAX_SESSION_LEN", 32*1024))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 16*1024))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 4))

DEBUG = False

if os.environ.get("DEBUG") in ("True", "1"):
    DEBUG = True

MODEL_ID = "nvidia/DeepSeek-V3-0324-NVFP4"

BASE_IMAGE = "docker.io/lmsysorg/sglang:dev"
#BASE_IMAGE = "lmsysorg/sglang:v0.5.6.post2-cu130-amd64-runtime"
#BASE_IMAGE = "lmsysorg/sglang:v0.5.6.post2"
runtime_image = bentoml.images.Image(
    base_image=BASE_IMAGE,
).requirements_file("requirements.txt")


@bentoml.service(
    name="bentosglang-deepseek-v3-0324-fp4",
    image=runtime_image,
    envs=[
        {"name": "MAX_SESSION_LEN", "value": f"{32*1024}"},
        {"name": "MAX_TOKENS", "value": f"{16*1024}"},
        {"name": "NUM_GPUS", "value": f"{NUM_GPUS}"},
        {"name": "PYTHONPATH", "value": "/sgl-workspace/sglang/python/:/app/.venv/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.12/dist-packages:/usr/lib/python312.zip:/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload"},
    ],
    traffic={
        "timeout": 300,
        "concurrency": 40,
    },
    resources={
        "gpu": NUM_GPUS,
        "gpu_type": "nvidia-b200",
    },
)
class SGL:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __command__(self) -> list[str]:
        import os

        extra_params = []
        kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE")
        if kv_cache_dtype:
            extra_params.extend(["--kv-cache-dtype", kv_cache_dtype])

        attention_backend = os.environ.get("ATTENTION_BACKEND")
        if attention_backend:
            extra_params.extend(["--attention-backend", attention_backend])
        
        enable_mixed_chunk = os.environ.get("ENABLE_MIXED_CHUNK")
        if enable_mixed_chunk:
            extra_params.extend(["--enable-mixed-chunk"])
        
        extra_params_list_json = os.environ.get("EXTRA_PARAMS_LIST_JSON")
        if extra_params_list_json:
            import json
            try:
                extra_params_list = json.loads(extra_params_list_json)
                if extra_params_list:
                    extra_params.extend(extra_params_list)
            except json.decoder.JSONDecodeError:
                print("extra params list loading error")
        
        return [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.hf_model,
            "--served-model-name", MODEL_ID,
            "--port", "8000",
            "--context-length", str(MAX_SESSION_LEN),
            "--tensor-parallel-size", str(NUM_GPUS),
            "--ep-size", str(NUM_GPUS),
            "--cuda-graph-max-bs", str(256),
            "--max-running-requests", str(256),
            "--enable-symm-mem",
            "--stream-interval", str(10),
            "--scheduler-recv-interval", str(30),
        ] + extra_params
