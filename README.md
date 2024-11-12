<div align="center">
    <h1 align="center">Self-host LLMs with SGLang and BentoML</h1>
</div>

This repository contains a group of BentoML example projects, showing you how to serve and deploy open-source LLMs using [SGLang](https://github.com/sgl-project/sglang), a fast serving framework for LLMs and VLMs.

💡 You can use these examples as bases for advanced code customization. For simple LLM hosting with OpenAI-compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/use-cases/index.html) for a full list of BentoML example projects.

The following is an example of serving one of the LLMs in this repository: Llama 3.1 8B.

## Prerequisites

- You have gained access to Llama 3.1 8B on [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoSGLang.git
cd BentoSGLang/llama3.1-8b-instruct

# Recommend Python 3.11
pip install -r requirements.txt

export HF_TOEKN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

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

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy . --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).