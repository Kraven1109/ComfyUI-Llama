# ComfyLLama

Lightweight ComfyUI plugin exposing llama.cpp-based one-shot Qwen VQA nodes.

Files:
- `llamacpp_inference.py` - main inference node using llama.cpp cli
- `gguf_loader.py` - helper to enumerate GGUF models in the ComfyUI models dir
 - `llama-help.md` - local copy of llama.cpp + llama-mtmd-cli help (source of truth as of 2025-11-16)
 - `media_paths_loader.py` - `MediaPathsLoader` node for flexible multiple media inputs

Usage:
 - Copy the `ComfyLLama` folder into `ComfyUI/custom_nodes` and restart ComfyUI.
 - Use `MediaPathsLoader` node to combine multiple media paths, then connect to `ComfyLLama` `media_paths` input.
 - For single media: Connect an IMAGE tensor to `image` input, AUDIO tensor to `audio` input.
 - Set `llama_cpp_folder` to the path containing llama-cli.exe and llama-mtmd-cli.exe (required).

## Nodes

### ComfyLLama (llama.cpp)
Core inference node for llama.cpp-based text generation with optional multimodal support (images/audio).

**Key Inputs:**
- `text`: Prompt string
- `gguf_model`: GGUF model file (connect from GGUF Loader)
- `llama_cpp_folder`: Path to llama-cli.exe and llama-mtmd-cli.exe
- Optional: `image` (single IMAGE tensor), `audio` (single AUDIO), `media_paths` (multiple paths from Media Paths Loader), `mmproj_model` (multimodal projector)

**Outputs:** Generated text (STRING)

**Example Workflow:**
```
[GGUF Loader] --> [ComfyLLama]
[Media Paths Loader] --> [ComfyLLama (media_paths)]
```

### GGUF Loader (llama.cpp)
Loads GGUF model files from ComfyUI's models directory.

**Inputs:**
- `gguf_name`: Dropdown of discovered .gguf files
- Optional: `gguf_path_override`: Custom file path

**Outputs:** MODEL (dict with file path)

### Media Paths Loader (llama.cpp)
Combines multiple file paths into a list for batch media processing.

**Inputs:** Dynamic path inputs (path_0, path_1, etc. - add via node UI)

**Outputs:** List of paths (PATH)

**Example:** Add multiple image/audio file paths, connect to ComfyLLama's `media_paths` for multimodal inference with several files.

Inputs & Multimodal behavior:
- Required: `llama_cpp_folder` (STRING) - path to folder containing llama-cli.exe and llama-mtmd-cli.exe
- Separate inputs for different media types:
	 - `image` (IMAGE) — single image tensor.
	 - `audio` (AUDIO) — single audio tensor or dict.
	 - `media_paths` (PATH) — multiple paths (use MediaPathsLoader for flexible input).
- Single media inputs (`image` or `audio`) are limited to one each and cannot be combined with `media_paths`. Use `media_paths` (via MediaPathsLoader) for multiple media files, as the underlying CLI supports repeated `--image` and `--audio` flags.
- Supported media: images (.png, .jpg, etc.) and audio (.wav, .mp3, etc.).
- If one or more images or audio files are provided (or `mmproj` is set), ComfyLLama will use `llama-mtmd-cli` for multimodal inference. In this strict mode the node will fail with a clear message if `llama-mtmd-cli` is not installed or not found in `llama_cpp_folder`.
- If no image/audio/mmproj is provided, the node will use the text-only `llama-cli`.
- `jinja_chat_template` (STRING) - Jinja chat template name (built-in) or custom template string (validated at runtime).
- `no_display_prompt` is hardcoded to True for cleaner output.

Notes:
- `llama-help.md` in this folder is the source of truth for available CLI flags (as of 2025-11-16).
- This plugin **does not** auto-detect flags from the binary; it relies on the included `llama-help.md` for documentation.
- If you need additional CLI options not exposed as UI fields, provide them via the `system_prompt` / other fields or update the node to include `extra_cli_args` in a future PR.
- Provide `llama_cpp_folder` path (required input) or ensure llama-cli.exe is in PATH.

Notes:
- This plugin is intentionally minimal and optimized for one-shot non-interactive workflows.
- It includes multimodal support (image+mmproj) and a local `GGUFLoader`.

## Acknowledgements

Many ideas and implementations in this project were inspired by and adapted from
[IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct). Thank you to
[@IuvenisSapiens](https://github.com/IuvenisSapiens) and contributors for the
original work; this project builds on those concepts with a different focus and
implementation details.

If you have feedback on attribution or spot anything that needs clarification, feel free to open an issue — I'm happy to adjust!

## Installation

Install this plugin into `ComfyUI/custom_nodes` using one of the methods below.

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager in your ComfyUI interface.
2. Search for "ComfyUI-Llama" or paste the repo URL: `https://github.com/Kraven1109/ComfyUI-Llama.git`
3. Click Install and restart ComfyUI.

### Manual Installation

Clone the repository directly into your `custom_nodes` folder:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Kraven1109/ComfyUI-Llama.git
```

Restart ComfyUI after installation.

### Updates

To update the plugin:

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-Llama
git pull
```

Restart ComfyUI after updating.