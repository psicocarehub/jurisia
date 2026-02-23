"""
Deploy do GAIA Legal Reasoning no Modal.com.

Modal:
- Serverless, pay per second
- L4 GPU for GAIA 4B (quantized)
- Persistent volumes for weights
- Auto-scale 0-3 replicas (min 0 saves cost when idle)

Usage:
    modal serve training/serve/modal_serve.py   # Dev mode
    modal deploy training/serve/modal_serve.py  # Production
"""

import time
import uuid
from pathlib import Path
from typing import Any

import modal

app = modal.App("jurisai-gaia-legal-reasoning")
volume = modal.Volume.from_name("gaia-weights", create_if_missing=True)

MODEL_ID = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.4.0",
        "transformers",
        "huggingface_hub",
        "fastapi",
    )
)


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format messages into Gemma 3 chat template."""
    prompt = ""
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt


@app.cls(
    image=image,
    gpu="L4",
    volumes={"/models": volume},
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    min_containers=0,
    max_containers=3,
)
class GaiaServe:
    """vLLM-powered inference for GAIA Legal Reasoning."""

    @modal.enter()
    def load_model(self) -> None:
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        model_path = "/models/gaia-legal-reasoning"

        if not Path(model_path).exists():
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL_ID, local_dir=model_path)

        self.llm = LLM(
            model=model_path,
            dtype="auto",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )
        outputs = self.llm.generate([prompt], params)
        out = outputs[0]
        return {
            "text": out.outputs[0].text,
            "tokens": len(out.outputs[0].token_ids),
            "finish_reason": out.outputs[0].finish_reason,
        }

    @modal.method()
    def health(self) -> dict[str, str]:
        return {"status": "ok", "model": MODEL_ID}


@app.function(image=image, allow_concurrent_inputs=20)
@modal.web_endpoint(method="POST")
def v1_chat_completions(request: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-compatible /v1/chat/completions endpoint."""
    gaia = GaiaServe()
    messages = request.get("messages", [])
    prompt = format_chat_prompt(messages)
    max_tokens = request.get("max_tokens", 4096)
    temperature = request.get("temperature", 0.7)

    t0 = time.time()
    result = gaia.generate.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - t0

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "completion_tokens": result["tokens"],
            "prompt_tokens": len(prompt) // 4,
            "total_tokens": result["tokens"] + len(prompt) // 4,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_ID, "service": "jurisai-gaia"}
