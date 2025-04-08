<div align="center">
    <h1 align="center">Self-host Llama 3.1 8B with SGLang and BentoML</h1>
</div>

This example demonstrates how to serve and deploy Llama 3.1 8B using [SGLang](https://github.com/sgl-project/sglang), a fast serving framework for LLMs and VLMs.

💡 You can use this example as a base for advanced code customization. For simple LLM hosting with OpenAI-compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

- You have gained access to Llama 3.1 8B on [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoSGLang.git
cd BentoSGLang/llama3.1-8b-instruct

# Recommend Python 3.11
pip install -r requirements.txt

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve

2024-11-12T02:47:06+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:SGL" listening on http://localhost:3000 (Press CTRL+C to quit)
2024-11-12T02:49:31+0000 [INFO] [entry_service:bentosglang-llama3.1-8b-instruct-service:1] Service bentosglang-llama3.1-8b-instruct-service initialized
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors in plain English",
  "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.",
  "max_tokens": 1024,
  "sampling_params": null
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors in plain English",
        max_tokens=1024
    )
    for response in response_generator:
        print(response, end='')
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
    model="meta-llama/Llama-3.1-8B-Instruct",
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

Make sure you have [logged in to BentoCloud](hhttps://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
