import os
import sys
import shutil
import subprocess
import tempfile
import time
import socket
import io
import base64

import torch
import numpy as np
import requests
import folder_paths
from PIL import Image


def _llama_server_binary(llama_cpp_folder: str) -> str:
    """Return the absolute path to llama-server, auto-discovering from PATH when folder is blank."""
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    if llama_cpp_folder:
        return os.path.join(llama_cpp_folder, binary_name)
    # Try to find in PATH
    found = shutil.which(binary_name)
    return found if found else binary_name  # fall back to bare name so error message is readable


class ComfyLLamaServerConfig:
    """Configuration node for llama-server settings.
    
    Separates server/sampling configuration from the main inference node
    for better organization and reusability.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # === Server Settings ===
                "llama_cpp_folder": ("STRING", {
                    "default": r"d:\Apps\llama-cuda" if sys.platform == "win32" else "",
                    "tooltip": (
                        "Folder containing the llama-server binary. "
                        "Leave empty to auto-detect from PATH (Linux/macOS). "
                        "Windows example: d:\\Apps\\llama-cuda"
                    ),
                }),
                "server_port": ("INT", {
                    "default": 8080, 
                    "min": 1024, 
                    "tooltip": "Port for the HTTP server. Use different ports if running multiple instances."
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1, 
                    "min": -1,
                    "tooltip": "-1 = offload all layers to GPU, 0 = CPU only, N = offload N layers"
                }),
                "ctx_size": ("INT", {
                    "default": 32768, 
                    "min": 128,
                    "tooltip": "Context size in tokens. Higher = more memory. No max limit - depends on your GPU VRAM."
                }),
                "n_predict": ("INT", {
                    "default": -1, 
                    "min": -2,
                    "tooltip": "-1 or 0 = until EOS (recommended), -2 = use remaining context (25% of ctx_size), N > 0 = exact token limit (auto-capped to 25% of ctx_size to prevent overflow)"
                }),
                "reasoning_budget": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": (
                        "Controls reasoning/thinking for thinking models (Qwen3-VL, DeepSeek-R1, etc.).\n"
                        "Passed to llama-server as --reasoning-budget N on startup.\n"
                        "\n"
                        "  -1 = unlimited thinking (default)\n"
                        "   0 = disable thinking entirely\n"
                        "  N>0 = token budget (experimental)"
                    ),
                }),
                
                # === Basic Sampling ===
                "temperature": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": "Randomness. 0 = deterministic, 0.3-0.7 = balanced, 1.0+ = creative/chaotic"
                }),
                "top_k": ("INT", {
                    "default": 40, 
                    "min": 0,
                    "tooltip": "Keep only top K tokens. 0 = disabled. Lower = more focused, higher = more diverse"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Nucleus sampling. Keep tokens with cumulative prob <= top_p. 1.0 = disabled"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold. Tokens below min_p * max_prob are filtered. 0.0 = disabled"
                }),
                
                # === Anti-Repetition ===
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, 
                    "min": 1.0, 
                    "max": 3.0, 
                    "step": 0.05,
                    "tooltip": "Penalize repeated tokens. 1.0 = disabled, 1.1-1.3 = mild, 1.5+ = strong"
                }),
                "repeat_last_n": ("INT", {
                    "default": 256, 
                    "min": 0,
                    "tooltip": "How many recent tokens to check for repetition. 0 = disabled, -1 = ctx_size"
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": "Penalize tokens based on frequency in text. 0.0 = disabled. Reduces common word spam."
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": "Penalize tokens that appeared at all. 0.0 = disabled. Encourages topic diversity."
                }),
                
                # === DRY Sampling (Don't Repeat Yourself) ===
                "dry_multiplier": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "DRY sampling strength. 0.0 = disabled. 0.5-1.0 = recommended. Prevents phrase repetition."
                }),
                "dry_base": ("FLOAT", {
                    "default": 1.75, 
                    "min": 1.0, 
                    "max": 3.0, 
                    "step": 0.05,
                    "tooltip": "DRY exponential base. Higher = stronger penalty for longer repeated sequences."
                }),
                "dry_allowed_length": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 10,
                    "tooltip": "Allow repetition of sequences up to this length. 2 = allow bigrams, 3 = allow trigrams"
                }),
                "dry_penalty_last_n": ("INT", {
                    "default": -1, 
                    "min": -1,
                    "tooltip": "Tokens to check for DRY. -1 = ctx_size, 0 = disabled"
                }),
            },
            "optional": {
                # === Hardware Tuning ===
                "kv_cache_type": (["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"], {
                    "default": "f16",
                    "tooltip": (
                        "KV cache quantization type. Reduces VRAM at slight precision cost.\n"
                        "f16 = full precision (default, safest)\n"
                        "q8_0 = 8-bit, good quality/VRAM balance\n"
                        "q4_0 = 4-bit, ~50% VRAM reduction (recommended for long-ctx models)\n"
                        "Applied to both K and V caches."
                    ),
                }),
                "batch_size": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "tooltip": "Prompt processing batch size (--batch-size). Higher = faster prompt ingestion, more VRAM."
                }),
                "ubatch_size": ("INT", {
                    "default": 512,
                    "min": 1,
                    "tooltip": "Micro-batch size for generation (--ubatch-size). Usually batch_size/4."
                }),
                # === Advanced Sampling ===
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "-1 = random seed each run. Set specific value for reproducible outputs."
                }),
                "typical_p": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Locally typical sampling. 1.0 = disabled. Lower values = more coherent but less creative."
                }),
                "mirostat": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 2,
                    "tooltip": "Mirostat algorithm. 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. Auto-adjusts sampling."
                }),
                "mirostat_tau": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Mirostat target entropy. Lower = more focused, higher = more diverse."
                }),
                "mirostat_eta": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Mirostat learning rate. How fast it adapts."
                }),
            },
        }
    
    RETURN_TYPES = ("SERVER_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = """Server configuration for llama-server.

🌐 **Cross-platform binary discovery:**
• Windows: Set llama_cpp_folder to e.g. d:\\Apps\\llama-cuda
• Linux/macOS: Leave empty to auto-detect from PATH, or provide the folder

🧠 **Thinking models (Qwen3-VL, DeepSeek-R1, etc.):**
• reasoning_budget=-1 → unlimited thinking (default)
• reasoning_budget=0  → disable thinking entirely

📌 **Quick Presets:**
• Creative Writing: temp=0.8, repeat_penalty=1.2, dry=0.5
• Factual/Code: temp=0.3, repeat_penalty=1.1, dry=0.0
• Anti-Repetition: repeat_penalty=1.3, dry=0.8, freq=0.1, pres=0.1

⚠️ **Troubleshooting:**
• Output repeating? → Increase dry_multiplier, repeat_penalty
• Output too random? → Lower temperature, increase top_k
• Output cut off? → Increase n_predict or set to -1
• Out of VRAM? → Lower ctx_size or use kv_cache_type=q4_0

💡 **Tips:**
• Chat template settings (jinja, custom template) are configured in the GGUF Loader node
• ctx_size and n_predict have no max limit - depends on your GPU
• Use external INT nodes to pipe custom values if needed"""

    def create_config(
        self,
        llama_cpp_folder,
        server_port,
        n_gpu_layers,
        ctx_size,
        n_predict,
        reasoning_budget,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        repeat_last_n,
        frequency_penalty,
        presence_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_penalty_last_n,
        kv_cache_type="f16",
        batch_size=2048,
        ubatch_size=512,
        seed=-1,
        typical_p=1.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
    ):
        config = {
            # Server hardware
            "llama_cpp_folder": llama_cpp_folder,
            "server_port": server_port,
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "n_predict": n_predict,
            "reasoning_budget": reasoning_budget,
            # KV cache & batch
            "kv_cache_type": kv_cache_type,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            # Basic sampling
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            # Anti-repetition
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            # DRY
            "dry_multiplier": dry_multiplier,
            "dry_base": dry_base,
            "dry_allowed_length": dry_allowed_length,
            "dry_penalty_last_n": dry_penalty_last_n,
            # Advanced sampling
            "seed": seed,
            "typical_p": typical_p,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
        }
        return (config,)


class ComfyLLamaServer:
    """ComfyLLama node using llama-server.exe for GGUF inference via HTTP API.
    
    Server-based version for potentially better performance and API compatibility.
    """

    def _strip_thinking_tags(self, text: str) -> str:
        """Strip <think>...</think> blocks from model output.
        
        With --reasoning-budget -1: server extracts thinking → reasoning_content,
        content is already clean. This handles the case where thinking leaks into content.
        """
        if '</think>' in text:
            # Take everything after the last </think> tag
            return text.split('</think>')[-1].strip()
        # If only <think> opening (no close), strip the opening tag too
        if text.startswith('<think>'):
            return text[len('<think>'):].strip()
        return text

    def _resolve_path(self, model_input):
        if isinstance(model_input, str):
            return model_input
        elif isinstance(model_input, dict) and 'path' in model_input:
            return model_input['path']
        return None

    def _tensor_to_base64(self, image_tensor):
        """Converts ComfyUI Image Tensor to base64 strings."""
        encoded_images = []
        
        # Handle batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        for i in range(image_tensor.shape[0]):
            # Convert Tensor (0-1 float) to uint8
            img_np = (255. * image_tensor[i].cpu().numpy()).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Save to memory buffer
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            
            # Encode
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            encoded_images.append(f"data:image/jpeg;base64,{img_str}")
            
        return encoded_images

    def _is_port_free(self, port: int) -> bool:
        """Return True when nothing is listening on the given port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.connect_ex(('127.0.0.1', port)) != 0

    def _find_free_port(self, preferred: int, search_range: int = 20) -> int:
        """Return the preferred port if free, otherwise the first free port in range."""
        if self._is_port_free(preferred):
            return preferred
        print(f"ComfyLLama: Port {preferred} busy, searching for a free port...")
        for offset in range(1, search_range + 1):
            candidate = preferred + offset
            if self._is_port_free(candidate):
                print(f"ComfyLLama: Using port {candidate} instead.")
                return candidate
        raise RuntimeError(
            f"No free port found in range {preferred}–{preferred + search_range}. "
            "Stop other processes or change server_port."
        )

    def _kill_process(self, proc) -> None:
        """Terminate a subprocess gracefully, escalating to SIGKILL / kill() as needed."""
        try:
            if sys.platform != "win32":
                import signal, os as _os
                try:
                    _os.killpg(_os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass

    def run_inference(self, gguf_model, config, prompt,
                     image=None, system_prompt="", stop_string=""):
        """
        Run inference with full sampling parameters from config.

        Args:
            gguf_model: MODEL dict from GGUFLoader (may contain 'mmproj_path')
            config: SERVER_CONFIG dict from ComfyLLamaServerConfig
            prompt: The user prompt text
            image: Optional ComfyUI IMAGE tensor for multimodal
            system_prompt: Optional system prompt
            stop_string: Optional stop sequence
        """
        # Extract config values
        llama_cpp_folder = config.get("llama_cpp_folder", "")
        server_port = config.get("server_port", 8080)
        n_gpu_layers = config.get("n_gpu_layers", -1)
        ctx_size = config.get("ctx_size", 32768)
        n_predict = config.get("n_predict", -1)
        
        server_process = None
        stderr_log = None
        try:

            # 1. Resolve paths
            gguf_path = self._resolve_path(gguf_model)
            if not gguf_path or not os.path.exists(gguf_path):
                return (f"Error: GGUF model not found: {gguf_path}",)

            # mmproj comes from the loader dict (preferred) — no separate input node needed
            mmproj_path = gguf_model.get('mmproj_path') if isinstance(gguf_model, dict) else None

            server_exe = _llama_server_binary(llama_cpp_folder)
            if not os.path.isfile(server_exe):
                binary_name = os.path.basename(server_exe)
                found_in_path = shutil.which(binary_name)
                if not found_in_path:
                    return (f"Error: llama-server binary not found. "
                            f"Tried: '{server_exe}'. Set llama_cpp_folder or add it to PATH.",)
                server_exe = found_in_path

            # Auto-select a free port
            server_port = self._find_free_port(server_port)

            # 2. Process Images (if any)
            image_data = []
            if image is not None:
                if not mmproj_path:
                    return ("Error: Image input requires an mmproj model. "
                            "Select mmproj_name in the GGUF Loader node.",)
                image_data = self._tensor_to_base64(image)

            # 3. Build Command — hardware and model loading only (no sampling params)
            cmd = [
                server_exe,
                "-m", gguf_path,
                "--port", str(server_port),
                "--ctx-size", str(ctx_size),
                "--n-gpu-layers", str(n_gpu_layers),
                "--flash-attn", "on",
                "--batch-size", str(config.get("batch_size", 2048)),
                "--ubatch-size", str(config.get("ubatch_size", 512)),
            ]

            # KV cache quantization — always apply both k and v together
            kv_cache_type = config.get("kv_cache_type", "f16")
            if kv_cache_type and kv_cache_type != "f16":
                cmd.extend([
                    "--cache-type-k", kv_cache_type,
                    "--cache-type-v", kv_cache_type,
                ])
            
            # Add chat template settings — from gguf_model dict (set in GGUF Loader)
            model_dict = gguf_model if isinstance(gguf_model, dict) else {}
            use_jinja = model_dict.get("use_jinja", True)
            chat_template = model_dict.get("chat_template", "auto")
            custom_chat_template = model_dict.get("custom_chat_template", "").strip()

            # custom_chat_template can be:
            #   a) a path to a .jinja file  →  use --chat-template-file <path>
            #   b) an inline Jinja2 string  →  use --chat-template <string>
            # In both cases --jinja must be present.
            if custom_chat_template:
                cmd.append("--jinja")
                if os.path.isfile(custom_chat_template):
                    cmd.extend(["--chat-template-file", custom_chat_template])
                    print(f"ComfyLLama: Using chat template file: {custom_chat_template}")
                else:
                    cmd.extend(["--chat-template", custom_chat_template])
            else:
                # No custom template — respect use_jinja and the preset dropdown
                if use_jinja:
                    cmd.append("--jinja")
                if chat_template != "auto":
                    cmd.extend(["--chat-template", chat_template])
            
            # Only load mmproj if image is provided
            if mmproj_path and image is not None:
                cmd.extend(["--mmproj", mmproj_path])

            # Reasoning budget — always pass explicitly so behavior is predictable
            # -1 = unlimited thinking, 0 = disable thinking, N>0 = token cap (experimental)
            reasoning_budget = config.get("reasoning_budget", -1)
            cmd.extend(["--reasoning-budget", str(reasoning_budget)])

            # 4. Start Server — capture stderr to a temp file so we can show it on crash
            stderr_log = tempfile.NamedTemporaryFile(
                mode='w', suffix='.log', prefix='llama_server_',
                delete=False
            )
            print(f"ComfyLLama: Starting One-Shot Server on port {server_port}...")
            print(f"ComfyLLama: Command: {' '.join(cmd)}")
            popen_kwargs = {
                "stdout": subprocess.DEVNULL,
                "stderr": stderr_log,
                "text": True,
                "cwd": os.path.dirname(server_exe) or None,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                # New process group so we can cleanly kill the whole tree
                popen_kwargs["start_new_session"] = True
            server_process = subprocess.Popen(cmd, **popen_kwargs)

            # 5. Wait for Health Check (Model Loading)
            load_timeout = 300 if mmproj_path else 120
            start_time = time.time()
            server_ready = False
            print(f"ComfyLLama: Waiting up to {load_timeout} seconds for model to load...")
            while time.time() - start_time < load_timeout:
                if server_process.poll() is not None:
                    # Read crash output from the stderr log
                    crash_msg = "Server crashed during startup."
                    try:
                        stderr_log.flush()
                        with open(stderr_log.name, 'r') as f:
                            tail = f.read()[-2000:]
                        if tail.strip():
                            crash_msg += f"\n--- llama-server output ---\n{tail}"
                    except Exception:
                        pass
                    return (crash_msg,)
                
                try:
                    response = requests.get(f"http://localhost:{server_port}/health", timeout=1)
                    if response.status_code == 200:
                        server_ready = True
                        print("ComfyLLama: Model loaded successfully!")
                        break
                    # Server started but model still loading (e.g. 503) — wait a bit
                    time.sleep(1)
                except:
                    time.sleep(1)
            
            if not server_ready:
                return ("Error: Server timed out while loading model. Multimodal models may take longer to load.",)

            # 6. Send Inference Request
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            user_content = []
            if image_data:
                for img in image_data:
                    user_content.append({"type": "image_url", "image_url": {"url": img}})
            user_content.append({"type": "text", "text": prompt})
            
            messages.append({"role": "user", "content": user_content})
            
            # 75 % of ctx_size is reserved for the prompt;
            # the remaining 25 % is available for generation output.
            # truncate_prompt_tokens tells llama-server to trim the input if it
            # would exceed this many tokens, preventing silent context overflow.
            prompt_token_limit = max(128, int(ctx_size * 0.75))
            # Maximum tokens the model can generate (at most the space left after prompt).
            generation_budget = max(64, ctx_size - prompt_token_limit)

            # Sanitize n_predict for the /chat/completions API:
            #   -1  → unlimited (fine, send as-is)
            #   -2  → "fill context" (llama.cpp CLI flag, not valid in HTTP API;
            #          translate to the remaining context space)
            #   0   → would generate nothing — treat as unlimited
            #   >0  → explicit token limit; cap to generation_budget to avoid overflow
            if n_predict == -2:
                api_n_predict = generation_budget
            elif n_predict == 0 or n_predict == -1:
                api_n_predict = -1   # let the model run to EOS within context
            else:
                api_n_predict = min(n_predict, generation_budget)

            # Build payload — sampling params passed per-request for flexibility
            payload = {
                "messages": messages,
                "n_predict": api_n_predict,
                "truncate_prompt_tokens": prompt_token_limit,
                "stop": [stop_string] if stop_string else [],
                # Basic sampling
                "temperature":       config.get("temperature", 0.6),
                "top_k":             config.get("top_k", 40),
                "top_p":             config.get("top_p", 0.9),
                "min_p":             config.get("min_p", 0.05),
                # Repetition penalties
                "repeat_penalty":    config.get("repeat_penalty", 1.1),
                "repeat_last_n":     config.get("repeat_last_n", 256),
                "frequency_penalty": config.get("frequency_penalty", 0.1),
                "presence_penalty":  config.get("presence_penalty", 0.1),
                # DRY sampling
                "dry_multiplier":    config.get("dry_multiplier", 0.8),
                "dry_base":          config.get("dry_base", 1.75),
                "dry_allowed_length":config.get("dry_allowed_length", 2),
                "dry_penalty_last_n":config.get("dry_penalty_last_n", -1),
            }

            # Optional params
            seed = config.get("seed", -1)
            if seed != -1:
                payload["seed"] = seed

            typical_p = config.get("typical_p", 1.0)
            if typical_p < 1.0:
                payload["typical_p"] = typical_p

            mirostat = config.get("mirostat", 0)
            if mirostat > 0:
                payload["mirostat"] = mirostat
                payload["mirostat_tau"] = config.get("mirostat_tau", 5.0)
                payload["mirostat_eta"] = config.get("mirostat_eta", 0.1)

            print("ComfyLLama: Sending prompt...")
            response = requests.post(f"http://localhost:{server_port}/chat/completions", json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                msg = result["choices"][0]["message"]
                content = self._strip_thinking_tags((msg.get("content") or "").strip())
                reasoning_content = msg.get("reasoning_content") or ""
                if not content and reasoning_content:
                    # Thinking model used all its token budget reasoning; no answer was generated.
                    # This happens when n_predict is too small relative to thinking depth.
                    thinking_len = len(reasoning_content)
                    print(f"ComfyLLama: WARNING — content empty, model spent {thinking_len} chars thinking without answering.")
                    return (
                        f"[No answer generated — model exhausted token budget ({thinking_len} chars) thinking.\n"
                        f"Fix: set reasoning_budget=0 to disable thinking, or increase ctx_size/n_predict.]",
                    )
                return (content,)
            else:
                return (f"Server Error {response.status_code}: {response.text}",)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"Error: {e}",)

        finally:
            # 7. CLEANUP — always release VRAM
            if server_process:
                print("ComfyLLama: Killing server to release VRAM...")
                self._kill_process(server_process)
                print("ComfyLLama: Server shutdown complete.")
            # Remove the temp stderr log file
            if stderr_log:
                try:
                    stderr_log.close()
                    os.unlink(stderr_log.name)
                except Exception:
                    pass


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gguf_model": ("MODEL", {
                    "tooltip": "GGUF model file from the Model Loader node"
                }),
                "config": ("SERVER_CONFIG", {
                    "tooltip": "Server configuration from the ServerConfig node"
                }),
                "prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Main prompt text. Can be combined with text_input."
                }),
            },
            "optional": {
                "text_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Additional text input from other nodes. Will be appended to prompt."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Image input for vision-language models. "
                               "Set mmproj_name in the GGUF Loader to enable VL mode."
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "System prompt to set model behavior/persona."
                }),
                "stop_string": ("STRING", {
                    "default": "",
                    "tooltip": "Stop sequence to end generation. E.g., '</s>' or '\\n\\n'"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_llamacpp_server"
    CATEGORY = "🦙 ComfyUI-LLama"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — inference is side-effectful (starts/stops a server)
        # and the result depends on runtime state, not just input values.
        return float("nan")
    DESCRIPTION = """Server-based Llama inference using the llama-server binary.

🔗 **Inputs:**
• gguf_model → GGUF Loader
• config → ServerConfig node (all sampling params)
• prompt: Your main prompt
• text_input: Optional extra text from other nodes

🖼️ **Vision/Multimodal (VL):**
• Pick mmproj_name in the GGUF Loader — no second loader node needed
• Then connect an image here to enable VL mode

🌐 **Cross-platform:**
• Linux/macOS: leave llama_cpp_folder empty → auto-detect from PATH,
  or set it to the folder containing the llama-server binary
• Windows: set it to e.g. d:\\Apps\\llama-cuda

💡 **Tips:**
• Port is auto-selected if the configured port is already in use
• Server stderr is captured and shown on crash for easier debugging
• Use external INT nodes to override ctx_size/n_predict limits"""

    def inference_llamacpp_server(
        self,
        gguf_model,
        config,
        prompt,
        text_input=None,
        image=None,
        system_prompt="",
        stop_string="",
    ):
        final_prompt = prompt
        if text_input:
            final_prompt = f"{final_prompt}\n{text_input}" if final_prompt else text_input

        return self.run_inference(
            gguf_model=gguf_model,
            config=config,
            prompt=final_prompt,
            image=image,
            system_prompt=system_prompt,
            stop_string=stop_string,
        )


class ComfyLLamaTextInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_input"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Simple text input node. Enter text that can be passed to other nodes."

    def text_input(self, text):
        return (text,)


class ComfyLLamaTextConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delimiter": ("STRING", {"default": "\n"}),
            },
            "optional": {
                "text1": ("STRING", {"forceInput": True}),
                "text2": ("STRING", {"forceInput": True}),
                "text3": ("STRING", {"forceInput": True}),
                "text4": ("STRING", {"forceInput": True}),
                "text5": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_concat"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Concatenate multiple text inputs with a custom delimiter. Supports escape sequences: \\n (newline), \\t (tab), \\r\\n (Windows line endings), \\\\n (literal \\n), \\\" (double quote), \\' (single quote), --- (separator line)."

    def text_concat(self, delimiter, text1=None, text2=None, text3=None, text4=None, text5=None):
        # Unescape delimiter to handle \n, \t, etc.
        try:
            delimiter = delimiter.encode().decode('unicode_escape')
        except:
            pass  # If unescaping fails, use as-is
        texts = [t for t in [text1, text2, text3, text4, text5] if t is not None]
        return (delimiter.join(texts),)


class ComfyLLamaPreviewText:
    """Preview text output in the ComfyUI interface with markdown support.
    
    Can work in two modes:
    1. With input connection: displays and passes through the connected text
    2. Without input: uses the cached_text field (can be edited manually)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Text input from another node. When connected, this takes priority."
                }),
                "cached_text": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Fallback text when no input is connected. You can paste/edit text here manually."
                }),
                "title": ("STRING", {"default": "Output"}),
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "preview_text"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = """Preview and pass-through text with optional caching.

🔗 **Two modes:**
• With input: Displays connected text and passes it through
• Without input: Uses cached_text field (editable)

💡 **Tips:**
• Copy output text to cached_text to keep it available after disconnecting
• Use as a text buffer/clipboard between workflow runs"""

    def preview_text(self, text=None, cached_text="", title="Output"):
        # Use connected text if available, otherwise use cached
        output_text = text if text is not None else cached_text
        
        # Return UI data for display and pass through the text
        return {"ui": {"text": [output_text], "title": [title]}, "result": (output_text,)}


class ComfyLLamaSaveText:
    """Save text content to a file."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": "output.txt"}),
            },
            "optional": {
                "append": ("BOOLEAN", {"default": False}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "save_text"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Save text to a file in the output directory. Option to append or overwrite, and add timestamp to filename."

    def save_text(self, text, filename, append=False, add_timestamp=True):
        import folder_paths
        
        output_dir = folder_paths.get_output_directory()
        
        # Add timestamp if requested
        if add_timestamp:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            if not ext:
                ext = ".txt"
            filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(output_dir, filename)
        
        # Write or append
        mode = "a" if append else "w"
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(text)
            if append:
                f.write("\n")  # Add newline when appending
        
        print(f"[LLama] Text saved to: {filepath}")
        return (filepath,)


class ComfyLLamaPromptBuilder:
    """Build prompts using templates with variable substitution."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "Describe the following in detail:\n\n{input}\n\nBe specific and concise.",
                    "multiline": True
                }),
            },
            "optional": {
                "input": ("STRING", {"forceInput": True}),
                "var1_name": ("STRING", {"default": ""}),
                "var1_value": ("STRING", {"forceInput": True}),
                "var2_name": ("STRING", {"default": ""}),
                "var2_value": ("STRING", {"forceInput": True}),
                "var3_name": ("STRING", {"default": ""}),
                "var3_value": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Build prompts using templates. Use {input} for main input, or define custom variables like {var1_name} with var1_value. Supports multiple variable substitutions."

    def build_prompt(self, template, input=None, 
                     var1_name="", var1_value=None,
                     var2_name="", var2_value=None,
                     var3_name="", var3_value=None):
        result = template
        
        # Replace {input} placeholder
        if input is not None:
            result = result.replace("{input}", input)
        
        # Replace custom variables
        if var1_name and var1_value is not None:
            result = result.replace(f"{{{var1_name}}}", var1_value)
        if var2_name and var2_value is not None:
            result = result.replace(f"{{{var2_name}}}", var2_value)
        if var3_name and var3_value is not None:
            result = result.replace(f"{{{var3_name}}}", var3_value)
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ComfyLLamaServerConfig": ComfyLLamaServerConfig,
    "ComfyLLamaServer": ComfyLLamaServer,
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
    "ComfyLLamaPreviewText": ComfyLLamaPreviewText,
    "ComfyLLamaSaveText": ComfyLLamaSaveText,
    "ComfyLLamaPromptBuilder": ComfyLLamaPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLamaServerConfig": "⚙️ Server Config",
    "ComfyLLamaServer": "🦙 LLama Server",
    "ComfyLLamaTextInput": "📝 Text Input",
    "ComfyLLamaTextConcat": "🔗 Text Concat",
    "ComfyLLamaPreviewText": "👁️ Preview Text",
    "ComfyLLamaSaveText": "💾 Save Text",
    "ComfyLLamaPromptBuilder": "🛠️ Prompt Builder",
}

