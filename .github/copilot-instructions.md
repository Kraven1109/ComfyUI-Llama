# ComfyUI-LLama Copilot Instructions

## Project Overview
ComfyUI-LLama is a custom node package that provides LLM inference for ComfyUI workflows. Supports two backends:
1. **llama.cpp (GGUF)**: Uses llama-server.exe HTTP API for GGUF model inference
2. **HuggingFace**: Loads models directly from HuggingFace Hub with bitsandbytes quantization

## Core Architecture
- **GGUF**: One-shot pattern via llama-server.exe HTTP API (server starts → runs inference → terminates)
- **HuggingFace**: Smart caching - model stays in global cache across runs, unloads on demand
- **Multimodal Support**: Images via base64 (GGUF) or PIL (HuggingFace)
- **Node-Based UI**: Custom ComfyUI nodes with web extensions for enhanced UX

## Key Components

### GGUF / llama.cpp Nodes

#### Model Loading
- `GGUFLoader`: Scans `folder_paths.models_dir` recursively for `.gguf` files
- Returns: `{"path": "/full/path/to/model.gguf"}` dict (not just string)
- Use `gguf_path_override` for custom paths outside models directory

#### Server Configuration
- `ComfyLLamaServerConfig`: Sampling parameters node
- Outputs `SERVER_CONFIG` dict with all llama-server parameters
- Separate from inference node for reusability

#### Main Inference
- `ComfyLLamaServer`: Core inference node using llama-server.exe
- Requires: `gguf_model` (from loader) + `config` (from config node) + `prompt`
- Optional: `image` (tensor) + `mmproj_model` (vision projector)
- Returns: Generated text string
- **Auto-terminates server after inference**

### HuggingFace Nodes

#### HFModelLoader
- Loads VL models from HuggingFace Hub
- Downloads to `[ComfyUI]/models/LLM/`
- Supports 4-bit/8-bit quantization via bitsandbytes
- Optional Flash Attention 2 support
- Returns: `HF_MODEL` + `HF_PROCESSOR` dicts

#### HFVLInference
- Vision-Language inference with HuggingFace models
- Supports image + text multimodal input
- **Smart caching**: Model stays in global `_HF_MODEL_CACHE` across runs
- `auto_unload` parameter (default: True) - unloads from cache after inference

#### HFTextInference
- Text-only inference (no image processing)
- Wrapper around HFVLInference with image=None

#### HFModelUnloader
- Manual model unloader (optional, for explicit cleanup)

### Text Processing Nodes
- `ComfyLLamaTextInput`: Simple multiline text input
- `ComfyLLamaTextConcat`: Join multiple texts with custom delimiter
- `ComfyLLamaPromptBuilder`: Template substitution with `{var1}`, `{input}` placeholders
- `ComfyLLamaPreviewText`: UI preview with copy button and **persistence across reloads**
- `ComfyLLamaSaveText`: Save to file with timestamp option

### Media Handling
- `MediaPathsLoader`: Combine multiple image/audio paths into list

## Node Connection Patterns

### GGUF Text Generation
```
GGUFLoader → ComfyLLamaServerConfig → ComfyLLamaServer → ComfyLLamaPreviewText
```

### GGUF Vision
```
LoadImage → ComfyLLamaServer
           ↑
GGUFLoader → ComfyLLamaServerConfig
           ↑
GGUFLoader (mmproj)
```

### HuggingFace Vision
```
HFModelLoader → HFVLInference → ComfyLLamaPreviewText
                ↑
          LoadImage
```

### HuggingFace Text-Only
```
HFModelLoader → HFTextInference → ComfyLLamaPreviewText
```

## Code Conventions

### Model Path Handling (GGUF)
```python
def _resolve_path(self, model_input):
    if isinstance(model_input, str):
        return model_input
    elif isinstance(model_input, dict) and 'path' in model_input:
        return model_input['path']
    return None
```

### Text Input Combination
```python
final_prompt = prompt
if text_input:
    final_prompt = f"{final_prompt}\n{text_input}"
```

### Image Processing (ComfyUI tensor to PIL)
```python
def _tensor_to_pil(self, image_tensor):
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]
    img_np = (255. * image_tensor.cpu().numpy()).astype(np.uint8)
    return Image.fromarray(img_np)
```

### Auto-Unload Pattern (HuggingFace)
```python
# Global cache for model persistence across runs
_HF_MODEL_CACHE = {
    "model": None,
    "processor": None,
    "model_id": None,
}

try:
    # ... inference code ...
    return (result,)
finally:
    if auto_unload:
        _unload_cached_model()
```

### Strip Thinking Tags
```python
# For "thinking" models (Huihui-Step3-VL, Qwen3-VL-Thinking)
# Chat template adds <think> automatically: '<|im_start|>assistant\n<think>\n'
def _strip_thinking_tags(self, text):
    if '</think>' in text:
        # Take content after </think>
        parts = text.split('</think>')
        return parts[-1].strip()
    return text
```

## Model-Specific Notes

### Thinking vs Non-Thinking Models
**Thinking models** (Huihui-Step3-VL, Qwen3-VL-Thinking):
- Chat template: `add_generation_prompt` adds `<|im_start|>assistant\n<think>\n`
- Output includes internal reasoning wrapped in `<think>...</think>`
- Use `strip_thinking=True` to get final answer only

**Non-thinking models** (Qwen2.5-VL, Llava):
- Chat template: `add_generation_prompt` adds `<|im_start|>assistant\n`
- Output is direct answer
- `strip_thinking` has no effect (safe to leave True)

### Recommended Generation Settings
**For VL tasks (image description, etc.):**
- `do_sample=False` (greedy, deterministic)
- `repetition_penalty=1.0` (no penalty, especially for JSON)
- `max_new_tokens=1024`

**For creative text:**
- `do_sample=True`
- `temperature=0.7`, `top_p=0.95`, `min_p=0.05`, `top_k=20`

### Server Lifecycle (GGUF)
- Start server with model loading
- Wait for `/health` endpoint (200 status)
- Send POST to `/chat/completions` with OpenAI-compatible payload
- **Always terminate server** in finally block to free VRAM

## Environment Notes
- **ComfyUI portable location**: `D:\Apps\ComfyUI_portable`
- **llama.cpp location**: `D:\Apps\llama-cuda`
- **Node symlink**: `[ComfyUI_Root]/custom_nodes/ComfyUI-Llama`
- **HuggingFace cache**: `[ComfyUI]/models/LLM/`

## Development Environment
- **OS**: Windows
- **Package manager**: `uv` (project uses portable environment)
- **Python path**: `D:\Apps\ComfyUI_portable\.venv\Scripts\python.exe`
- **ComfyUI portable**: `D:\Apps\ComfyUI_portable`
- **llama.cpp location**: `D:\Apps\llama-cuda`
- **Source code**: `D:\quang_dev\ComfyUI-LLama`
- **Symlink target**: `D:\Apps\ComfyUI_portable\ComfyUI\custom_nodes\ComfyUI-Llama`
- **Run command example**:
  ```powershell
  cd D:\Apps\ComfyUI_portable
  .\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'ComfyUI'); ..."
  ```

## Web Extensions
- Located in `web/comfyui/`
- `preview_text.js`: Adds scrollable text preview with copy functionality
- **Persistence**: Uses `onSerialize`/`onConfigure` to save/restore preview text

## File Structure
```
__init__.py              # Node registration
gguf_loader.py           # GGUF model discovery
llamacpp_inference_server.py  # GGUF inference via llama-server
hf_model_loader.py       # HuggingFace model loading
hf_inference.py          # HuggingFace inference
media_paths_loader.py    # Media path handling
web/comfyui/
  preview_text.js        # Preview Text UI extension
  media_paths_loader.js  # Media paths UI
_archive/                # Deprecated files (CLI-based inference)
```

## Dependencies
- Core: `torch`, `torchvision`, `pillow`, `numpy`, `requests`
- GGUF: `llama-server.exe` from llama.cpp build
- HuggingFace: `transformers`, `accelerate`, `bitsandbytes` (optional)

## Common Patterns
- **One-Shot**: GGUF loads, infers, unloads automatically per run
- **Smart Caching**: HuggingFace caches model globally, reuses across runs
- **Sampling Params**: GGUF uses ServerConfig node; HuggingFace uses inline params
- **Multimodal**: GGUF requires mmproj; HuggingFace uses native VL models
- **VRAM Management**: auto_unload=True frees VRAM; False keeps model cached
- **Timeouts**: 120s for text, 300s for multimodal loading

## Troubleshooting
- **GGUF port busy**: Change `server_port` or kill existing process
- **HuggingFace OOM**: Use 4bit quantization
- **Slow first load**: HuggingFace downloads ~15GB on first use
- **Model not reused**: Check if auto_unload was True; set False for persistence
