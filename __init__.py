# ComfyUI_LLama - llama.cpp integration for ComfyUI
# Provides LLM inference nodes using llama-server

from .llamacpp_inference import ComfyLLama
from .gguf_loader import GGUFLoader
from .media_paths_loader import MediaPathsLoader
from .llamacpp_inference_server import (
    ComfyLLamaServer, 
    ComfyLLamaTextInput, 
    ComfyLLamaTextConcat,
    ComfyLLamaPreviewText,
    ComfyLLamaSaveText,
    ComfyLLamaPromptBuilder,
)

NODE_CLASS_MAPPINGS = {
    "ComfyLLama": ComfyLLama,
    "GGUFLoader": GGUFLoader,
    "MediaPathsLoader": MediaPathsLoader,
    "ComfyLLamaServer": ComfyLLamaServer,
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
    "ComfyLLamaPreviewText": ComfyLLamaPreviewText,
    "ComfyLLamaSaveText": ComfyLLamaSaveText,
    "ComfyLLamaPromptBuilder": ComfyLLamaPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLama": "🦙 ComfyUI-LLama CLI [Deprecated]",
    "GGUFLoader": "📦 ComfyUI-LLama GGUF Loader",
    "MediaPathsLoader": "🖼️ ComfyUI-LLama Media Paths",
    "ComfyLLamaServer": "🦙 ComfyUI-LLama Server",
    "ComfyLLamaTextInput": "📝 ComfyUI-LLama Text Input",
    "ComfyLLamaTextConcat": "🔗 ComfyUI-LLama Text Concat",
    "ComfyLLamaPreviewText": "👁️ ComfyUI-LLama Preview Text",
    "ComfyLLamaSaveText": "💾 ComfyUI-LLama Save Text",
    "ComfyLLamaPromptBuilder": "🛠️ ComfyUI-LLama Prompt Builder",
}

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web/comfyui"