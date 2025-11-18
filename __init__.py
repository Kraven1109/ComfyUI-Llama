# Minimal ComfyUI plugin for llama.cpp one-shot nodes
from .llamacpp_inference import ComfyLLama
from .gguf_loader import GGUFLoader
from .media_paths_loader import MediaPathsLoader

NODE_CLASS_MAPPINGS = {
    "ComfyLLama": ComfyLLama,
    "GGUFLoader": GGUFLoader,
    
    "MediaPathsLoader": MediaPathsLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLama": "ComfyLLama (llama.cpp)",
    "GGUFLoader": "GGUF Loader (llama.cpp)",
     
    "MediaPathsLoader": "Media Paths Loader (llama.cpp)",
}

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web/comfyui"