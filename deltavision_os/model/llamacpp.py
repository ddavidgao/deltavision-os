"""
llama.cpp server backend. Thin subclass of OpenAIModel with a fixed base_url.

llama.cpp's server (`llama-server`) exposes an OpenAI-compatible endpoint on
`/v1/chat/completions`, so the V1 OpenAIModel handles all the heavy lifting
— we just point it at the local server and skip the API key.

Usage:
    model = LlamaCppModel(host="100.70.57.66", port=8080, model="mai-ui-8b")

Target models (Windows RTX 5080 Laptop, 16GB VRAM):
  - MAI-UI-8B (best ScreenSpot-Pro as of 2026-04)
  - Qwen3-VL-8B (strong general VLM)
  - UI-TARS-1.5-7B (Ollama fallback if llama.cpp gguf unavailable)
"""

from .openai import OpenAIModel


class LlamaCppModel(OpenAIModel):
    """
    llama.cpp server runs on David's Windows box at Tailscale IP 100.70.57.66.
    Default local port 8080; can be remote via Tailscale.
    """

    def __init__(
        self,
        model: str = "default",
        host: str = "localhost",
        port: int = 8080,
        api_key: str = "not-needed",
    ):
        base_url = f"http://{host}:{port}/v1"
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.host = host
        self.port = port

    def __repr__(self):
        return f"LlamaCppModel(host={self.host!r}, port={self.port}, model={self.model!r})"
