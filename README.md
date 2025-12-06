# 🦙 ComfyUI_LLama

Lightweight ComfyUI plugin for llama.cpp-based LLM inference with multimodal support.

## Features

- 🦙 **LLM Inference** via llama-server HTTP API
- 🖼️ **Multimodal Support** for images and audio
- 📦 **GGUF Model Loading** from ComfyUI models directory
- 🛠️ **Prompt Building** with templates and variables
- 👁️ **Text Preview** with copy functionality
- 💾 **Text Saving** to files

## Nodes

| Node | Icon | Description |
|------|------|-------------|
| `ComfyLLamaServer` | 🦙 | LLM inference via llama-server HTTP API |
| `GGUFLoader` | 📦 | Load GGUF model files |
| `MediaPathsLoader` | 🖼️ | Combine multiple media paths |
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
git clone https://github.com/Kraven1109/ComfyUI-Llama.git ComfyUI_LLama
```

Restart ComfyUI after installation.

## Usage

### Basic Text Generation

```
[📦 GGUF Loader] → [🦙 LLama Server] → [👁️ Preview Text]
```

### Image Description

```
[LoadImage] → [🦙 LLama Server] → [👁️ Preview Text]
              ↑
[📦 GGUF Loader] + [mmproj model]
```

### Prompt Building

```
[📝 Text Input] → [🛠️ Prompt Builder] → [🦙 LLama Server]
```

### Text to Speech (with TTSS)

```
[🦙 LLama Server] → [🔊 Text to Speech] → [🎧 Preview Audio]
```

## Node Details

### 🦙 LLama Server

Core inference node using llama-server HTTP API.

**Key Inputs:**
- `gguf_model`: GGUF model file (from GGUF Loader)
- `text_input`: Prompt text
- `image`: Optional IMAGE tensor for vision models
- `mmproj_model`: Multimodal projector for vision
- `server_port`: HTTP port (default: 8080)
- `temperature`, `max_tokens`, `seed`: Generation parameters

**Outputs:** Generated text (STRING)

### 📦 GGUF Loader

Loads GGUF model files from ComfyUI's models directory.

**Inputs:**
- `gguf_name`: Dropdown of discovered .gguf files
- `gguf_path_override`: Custom file path (optional)

**Outputs:** MODEL (dict with file path)

### 🛠️ Prompt Builder

Build prompts using templates with variable substitution.

**Template Syntax:**
- `{input}` - Main input placeholder
- `{var1_name}`, `{var2_name}`, `{var3_name}` - Custom variables

**Example Template:**
```
Describe the following image:

{input}

Style: {style}
Format: {format}
```

### 👁️ Preview Text

Preview text output in the node interface.

**Features:**
- Monospace font for code/text
- Scrollable for long content
- 📋 Copy button for easy copying
- Pass-through output for chaining

### 💾 Save Text

Save text content to a file.

**Options:**
- `filename`: Output filename
- `append`: Append to existing file (default: false)
- `add_timestamp`: Add timestamp to filename (default: true)

### 🔗 Text Concat

Concatenate multiple text inputs with custom delimiter.

**Supports escape sequences:**
- `\n` - newline
- `\t` - tab
- `\r\n` - Windows line endings

## Requirements

- llama.cpp with llama-server.exe
- GGUF model files
- Optional: mmproj file for vision models

## Related Projects

- **[TTSS](https://github.com/...)** - Text-to-Speech nodes for ComfyUI (works great with LLama output!)

## Acknowledgements

Inspired by [IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct](https://github.com/IuvenisSapiens/ComfyUI_Qwen3-VL-Instruct).

## License

MIT License