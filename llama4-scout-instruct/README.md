<div align="center">
    <h1 align="center">Self-host Llama 4 Scout with SGLang and BentoML</h1>
</div>

This example demonstrates how to serve and deploy Llama-4-Scout-17B-16E-Instruct using [SGLang](https://github.com/sgl-project/sglang), a fast serving framework for LLMs and VLMs.

💡 You can use this example as a base for advanced code customization. For simple LLM hosting with OpenAI-compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

- You have gained access to Llama-4-Scout-17B-16E-Instruct on [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct).
- If you want to test the Service locally, we recommend you use Nvidia GPUs with at least 80GBx2 VRAM (e.g about 2 H100 GPUs).

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoSGLang.git
cd BentoSGLang/llama4-scout-instruct

# Recommend Python 3.11
pip install -r requirements.txt

export HF_TOEKN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve service:SGL

2025-04-07T02:47:06+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SGL" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -s -X POST \
    'http://localhost:3000/generate' \
    -H 'Content-Type: application/json' \
    -d '{
        "max_tokens": 1024,
        "prompt": "Explain superconductors in plain English",
        "sampling_params": null,
        "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'"'"'t know the answer to a question, please don'"'"'t share false information."
    }' 
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.generate(
        max_tokens=1024,
        prompt="Explain superconductors in plain English",
        sampling_params=None,
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share False information.",
    )
```

</details>

<details>

<summary>OpenAI-compatible endpoints</summary>

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

chat_completion = client.chat.completions.create(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors in plain English"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

</details>

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
