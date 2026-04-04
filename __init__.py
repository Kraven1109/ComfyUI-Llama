# ComfyUI_LLama - LLM integration for ComfyUI
# Supports both llama.cpp (GGUF) and HuggingFace models

from .gguf_loader import GGUFLoader
from .media_paths_loader import MediaPathsLoader
from .llamacpp_inference_server import (
    ComfyLLamaServerConfig,
    ComfyLLamaServer, 
    ComfyLLamaTextInput, 
    ComfyLLamaTextConcat,
    ComfyLLamaPreviewText,
    ComfyLLamaSaveText,
    ComfyLLamaPromptBuilder,
)
from .hf_model_loader import HFModelLoader, HFModelUnloader
from .hf_inference import HFVLInference, HFTextInference

NODE_CLASS_MAPPINGS = {
    # GGUF / llama.cpp nodes
    "GGUFLoader": GGUFLoader,
    "MediaPathsLoader": MediaPathsLoader,
    "ComfyLLamaServerConfig": ComfyLLamaServerConfig,
    "ComfyLLamaServer": ComfyLLamaServer,
    # Text utility nodes
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
    "ComfyLLamaPreviewText": ComfyLLamaPreviewText,
    "ComfyLLamaSaveText": ComfyLLamaSaveText,
    "ComfyLLamaPromptBuilder": ComfyLLamaPromptBuilder,
    # HuggingFace nodes
    "HFModelLoader": HFModelLoader,
    "HFModelUnloader": HFModelUnloader,
    "HFVLInference": HFVLInference,
    "HFTextInference": HFTextInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # GGUF / llama.cpp
    "GGUFLoader": "📦 GGUF Loader",
    "MediaPathsLoader": "🖼️ Media Paths",
    "ComfyLLamaServerConfig": "⚙️ LLama Server Config",
    "ComfyLLamaServer": "🦙 LLama Server (GGUF)",
    # Text utilities
    "ComfyLLamaTextInput": "📝 Text Input",
    "ComfyLLamaTextConcat": "🔗 Text Concat",
    "ComfyLLamaPreviewText": "👁️ Preview Text",
    "ComfyLLamaSaveText": "💾 Save Text",
    "ComfyLLamaPromptBuilder": "🛠️ Prompt Builder",
    # HuggingFace
    "HFModelLoader": "🤗 HF Model Loader",
    "HFModelUnloader": "🗑️ HF Model Unloader",
    "HFVLInference": "🤗 HF Vision-Language",
    "HFTextInference": "🤗 HF Text Generation",
}

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web/comfyui"