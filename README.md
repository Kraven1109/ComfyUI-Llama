# 🦙 ComfyUI-LLama

Lightweight ComfyUI plugin for LLM inference with multimodal support. Supports both **llama.cpp (GGUF)** and **HuggingFace** models.

## Nodes

### GGUF / llama.cpp Nodes
| Node | Icon | Description |
|------|------|-------------|
| `ComfyLLamaServer` | 🦙 | LLM inference via llama-server HTTP API |
| `ComfyLLamaServerConfig` | ⚙️ | Server configuration (sampling params) |
| `GGUFLoader` | 📦 | Load GGUF model files |
| `MediaPathsLoader` | 🖼️ | Combine multiple media paths |

### HuggingFace Nodes
| Node | Icon | Description |
|------|------|-------------|
| `HFModelLoader` | 🤗 | Load HuggingFace VL models with quantization |
| `HFVLInference` | 🤗 | Vision-Language inference (auto-unload) |
| `HFTextInference` | 🤗 | Text-only inference |
| `HFModelUnloader` | 🗑️ | Manual model unloader |

### Utility Nodes
| Node | Icon | Description |
|------|------|-------------|
| `ComfyLLamaTextInput` | 📝 | Simple text input |
| `ComfyLLamaTextConcat` | 🔗 | Concatenate multiple texts |
| `ComfyLLamaPreviewText` | 👁️ | Preview text output with copy button |
| `ComfyLLamaSaveText` | 💾 | Save text to file |
| `ComfyLLamaPromptBuilder` | 🛠️ | Build prompts with templates |

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "ComfyUI-Llama" or paste: `https://github.com/Kraven1109/ComfyUI-Llama.git`
3. Click Install and restart ComfyUI

### Manual Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Kraven1109/ComfyUI-Llama.git ComfyUI-LLama
pip install -r ComfyUI-LLama/requirements.txt
```

Restart ComfyUI after installation.

## Requirements

### Core
- `torch`, `torchvision`, `pillow`, `numpy`

### For GGUF / llama.cpp
- llama.cpp with `llama-server.exe` built
- GGUF model files

### For HuggingFace Models
- `transformers>=4.40.0`
- `accelerate`
- `bitsandbytes` (for 4-bit/8-bit quantization)
- `flash-attn` (optional, for faster inference)

Install HuggingFace dependencies:
```bash
pip install transformers accelerate bitsandbytes
pip install flash-attn --no-build-isolation  # optional
```

## Usage

### Method 1: GGUF with llama-server (highly recommended)

Best for: Local GGUF models, maximum control over sampling

```
[📦 GGUF Loader] → [⚙️ Server Config] → [🦙 LLama Server] → [👁️ Preview Text]
```

### Method 2: HuggingFace Models

Best for: Easy access to HuggingFace models, automatic quantization

```
[🤗 HF Model Loader] → [🤗 HF Vision-Language] → [👁️ Preview Text]
                    ↑
              [LoadImage]
```

**Note:** 
- **GGUF**: One-shot approach - server starts, runs inference, then terminates.
- **HuggingFace**: Smart caching - model stays in VRAM across runs. Set `auto_unload=True` to free VRAM after each run, or `False` to keep model loaded for faster repeated runs.

## Node Details

### 🤗 HF Model Loader

Load Vision-Language models from HuggingFace with optional quantization.

**Key Inputs:**
- `model_preset`: Popular VL models dropdown
- `quantization`: none / 4bit / 8bit
- `torch_dtype`: auto / float16 / bfloat16
- `flash_attention`: Enable Flash Attention 2

**Supported Models:**
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `microsoft/Phi-4-multimodal-instruct`
- Any HuggingFace VL model (use Custom preset)

**Performance Tips:**
- Enable `flash_attention` for faster inference
- SDPA mode auto-leverages Sage/Radial/Sparse attention if available in ComfyUI
- 4-bit quantization provides best balance of speed and quality

**VRAM Usage (7B model):**
- 4bit: ~4-6 GB
- 8bit: ~8-10 GB
- none: ~14-16 GB

### 🤗 HF Vision-Language

Run inference with VL models. Supports smart caching for repeated runs.

**Key Inputs:**
- `model`/`processor`: From HFModelLoader
- `prompt`: User prompt
- `image`: Optional IMAGE tensor
- `auto_unload`: When `True` (default), frees VRAM after inference. Set to `False` to keep model cached for faster repeated runs.

**Smart Caching:**
- Model is cached globally after first load
- Subsequent runs reuse the cached model (much faster)
- Cache is invalidated when switching to a different model
- Set `auto_unload=False` in the last run to keep model for next queue

### 🦙 LLama Server (GGUF)

GGUF inference using llama-server HTTP API. Server starts, runs inference, then terminates.

**Key Inputs:**
- `gguf_model`: From GGUF Loader
- `config`: From ServerConfig node
- `prompt`: User prompt
- `image`: Optional image for vision models
- `mmproj_model`: Required for vision (from GGUF Loader)

### 👁️ Preview Text

Preview text output with persistence across workflow reloads.

**Features:**
- Scrollable text preview
- 📋 Copy button
- **Text persists after reload/switching workflows**
- Pass-through output for chaining

## Model Storage

- **GGUF models**: `[ComfyUI]/models/` (any subfolder)
- **HuggingFace models**: `[ComfyUI]/models/LLM/` (auto-downloaded)

## Troubleshooting

### HuggingFace Issues

| Problem | Solution |
|---------|----------|
| Out of VRAM | Use 4bit quantization |
| Slow first load | Normal - models are downloading |
| Flash Attention error | Install `flash-attn` or disable |
| Gated model access | Add HuggingFace token |

### GGUF Issues

| Problem | Solution |
|---------|----------|
| Port busy | Change `server_port` or kill existing process |
| Model not found | Check `folder_paths.models_dir` |
| Long load times | Multimodal models take longer (300s timeout) |

## License

MIT License

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct)
