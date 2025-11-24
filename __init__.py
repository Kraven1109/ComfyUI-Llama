# Minimal ComfyUI plugin for llama.cpp one-shot nodes
from .llamacpp_inference import ComfyLLama
from .gguf_loader import GGUFLoader
from .media_paths_loader import MediaPathsLoader
from .llamacpp_inference_server import ComfyLLamaServer, ComfyLLamaTextInput, ComfyLLamaTextConcat

NODE_CLASS_MAPPINGS = {
    "ComfyLLama": ComfyLLama,
    "GGUFLoader": GGUFLoader,
    
    "MediaPathsLoader": MediaPathsLoader,
    "ComfyLLamaServer": ComfyLLamaServer,
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLama": "ComfyLLama (llama.cpp) [DEPRECATED - Use Server]",
    "GGUFLoader": "GGUF Loader (llama.cpp)",
     
    "MediaPathsLoader": "Media Paths Loader (llama.cpp)",
    "ComfyLLamaServer": "ComfyLLama (llama-server)",
    "ComfyLLamaTextInput": "Text Input",
    "ComfyLLamaTextConcat": "Text Concat",
}

# WEB_DIRECTORY is the comfyui nodes directory that ComfyUI will link and auto-load.
WEB_DIRECTORY = "./web/comfyui"