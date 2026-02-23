"""
Deploy do GAIA Legal Reasoning no Modal.com.

Modal:
- Serverless, pay per second
- L4 $0.80/hr para GAIA 4B quantizado
- Volumes persistentes para weights
- min_containers=1 para evitar cold start

Usage:
    modal serve training/serve/modal_serve.py
    modal deploy training/serve/modal_serve.py
"""

import json
from pathlib import Path
from typing import Any

import modal

# ============================================
# APP & IMAGE
# ============================================

app = modal.App("jurisai-gaia-serve")
volume = modal.Volume.from_name("gaia-weights", create_if_missing=True)

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


# ============================================
# PROMPT FORMATTING
# ============================================

def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format messages into Gemma chat template."""
    prompt = ""
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt


# ============================================
# GAIA SERVE CLASS
# ============================================

@app.cls(
    image=image,
    gpu="L4",
    volumes={"/models": volume},
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    min_containers=1,
    max_containers=5,
)
class GaiaServe:
    """vLLM-powered inference for GAIA Legal Reasoning."""

    @modal.enter()
    def load_model(self) -> None:
        """Load model on container start."""
        from vllm import LLM
        from vllm import SamplingParams

        self.SamplingParams = SamplingParams
        model_path = "/models/gaia-legal-reasoning"

        if not Path(model_path).exists():
            from huggingface_hub import snapshot_download
            snapshot_download(
                "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
                local_dir=model_path,
            )

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
        """Generate completion (non-streaming)."""
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
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate completion (returns list of tokens for streaming)."""
        params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )
        # vLLM synchronous generate; for true streaming use llm.generate with stream
        outputs = self.llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        # Return as single chunk for simplicity (true streaming needs async)
        return [text] if text else []


# ============================================
# OPENAI-COMPATIBLE ENDPOINT
# ============================================

@app.function(image=image, allow_concurrent_inputs=20)
@modal.web_endpoint(method="POST")
def v1_chat_completions(request: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Forwards to GaiaServe GPU class.
    """
    gaia = GaiaServe()
    messages = request.get("messages", [])
    prompt = format_chat_prompt(messages)
    max_tokens = request.get("max_tokens", 4096)
    temperature = request.get("temperature", 0.7)

    if request.get("stream", False):
        chunks = gaia.generate_stream.remote(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Return as single response for simplicity (streaming needs SSE)
        full_text = "".join(chunks) if chunks else ""
        return {
            "choices": [{"message": {"role": "assistant", "content": full_text}}],
            "usage": {"completion_tokens": len(full_text) // 4},
        }

    result = gaia.generate.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    ).get()
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "completion_tokens": result["tokens"],
            "prompt_tokens": 0,
            "total_tokens": result["tokens"],
        },
    }
