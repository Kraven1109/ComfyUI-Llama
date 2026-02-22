import os
import subprocess
import time
import socket
import io
import base64

import torch
import numpy as np
import requests
import folder_paths
from PIL import Image


class ComfyLLamaServerConfig:
    """Configuration node for llama-server sampling parameters.
    
    Separates server/sampling configuration from the main inference node
    for better organization and reusability.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # === Server Settings ===
                "llama_cpp_folder": ("STRING", {
                    "default": r"d:\Apps\llama-cuda",
                    "tooltip": "Path to llama.cpp folder containing llama-server.exe"
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
                    "tooltip": "-1 = infinite (until EOS), -2 = fill context, 0+ = exact token limit. No max limit."
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
                    "default": 1.3, 
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
                # === Advanced ===
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
                # === Chat Template ===
                "use_jinja": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Jinja2 chat template processing. Recommended for chat models."
                }),
                "chat_template": (["auto", "chatml", "llama2", "llama3", "gemma", "phi3", "phi4", 
                                   "mistral-v1", "mistral-v3", "deepseek", "deepseek2", "deepseek3",
                                   "command-r", "vicuna", "zephyr", "openchat", "falcon3", "exaone3"], {
                    "default": "auto",
                    "tooltip": "Chat template format. 'auto' = detect from model metadata. Use specific template if auto-detection fails."
                }),
                "custom_chat_template": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom Jinja2 chat template. Overrides chat_template if provided. Leave empty to use preset."
                }),
            },
        }
    
    RETURN_TYPES = ("SERVER_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = """Server configuration for llama-server.exe.

📌 **Quick Presets:**
• Creative Writing: temp=0.8, repeat_penalty=1.2, dry=0.5
• Factual/Code: temp=0.3, repeat_penalty=1.1, dry=0.0
• Anti-Repetition: repeat_penalty=1.3, dry=0.8, freq=0.1, pres=0.1

⚠️ **Troubleshooting:**
• Output repeating? → Increase dry_multiplier, repeat_penalty
• Output too random? → Lower temperature, increase top_k
• Output cut off? → Increase n_predict or set to -1
• Out of VRAM? → Lower ctx_size or n_gpu_layers

💡 **Tips:**
• ctx_size and n_predict have no max limit - depends on your GPU
• Use external INT nodes to pipe custom values if needed"""

    def create_config(
        self,
        llama_cpp_folder,
        server_port,
        n_gpu_layers,
        ctx_size,
        n_predict,
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
        seed=-1,
        typical_p=1.0,
        mirostat=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        use_jinja=True,
        chat_template="auto",
        custom_chat_template="",
    ):
        config = {
            # Server
            "llama_cpp_folder": llama_cpp_folder,
            "server_port": server_port,
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "n_predict": n_predict,
            # Basic
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
            # Advanced
            "seed": seed,
            "typical_p": typical_p,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            # Chat Template
            "use_jinja": use_jinja,
            "chat_template": chat_template,
            "custom_chat_template": custom_chat_template,
        }
        return (config,)


class ComfyLLamaServer:
    """ComfyLLama node using llama-server.exe for GGUF inference via HTTP API.
    
    Server-based version for potentially better performance and API compatibility.
    """

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

    def _wait_for_port_free(self, port):
        """Ensures the port is free before trying to bind."""
        timeout = 5
        start = time.time()
        while time.time() - start < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return # Port is free
            time.sleep(0.5)
        print(f"Warning: Port {port} seems busy, attempting to start anyway...")

    def run_inference(self, gguf_model, config, prompt, 
                     image=None, mmproj_model=None, system_prompt="", stop_string=""):
        """
        Run inference with full sampling parameters from config.
        
        Args:
            gguf_model: Path to GGUF model file
            config: Dictionary containing all server and sampling parameters
            prompt: The user prompt text
            image: Optional image tensor for multimodal
            mmproj_model: Optional mmproj model path for vision
            system_prompt: Optional system prompt
            stop_string: Optional stop sequence
        """
        server_process = None
        
        # Extract config values
        llama_cpp_folder = config.get("llama_cpp_folder", r"d:\Apps\llama-cuda")
        server_port = config.get("server_port", 8080)
        n_gpu_layers = config.get("n_gpu_layers", -1)
        ctx_size = config.get("ctx_size", 32768)
        n_predict = config.get("n_predict", -1)
        
        try:
            # 1. Setup Paths
            gguf_path = self._resolve_path(gguf_model)
            mmproj_path = self._resolve_path(mmproj_model)

            if not gguf_path or not os.path.exists(gguf_path):
                return (f"Error: GGUF Model not found at {gguf_path}",)

            server_exe = os.path.join(llama_cpp_folder, "llama-server.exe")
            if not os.path.exists(server_exe):
                return (f"Error: llama-server.exe not found at {server_exe}",)

            # 2. Process Images (if any)
            image_data = []
            if image is not None:
                if not mmproj_path:
                    return ("Error: Image input detected but no mmproj_model provided.",)
                image_data = self._tensor_to_base64(image)

            # 3. Build Command with sampling params
            cmd = [
                server_exe,
                "-m", gguf_path,
                "--port", str(server_port),
                "--ctx-size", str(ctx_size),
                "--n-gpu-layers", str(n_gpu_layers),
                "--threads", "-1",
                "--flash-attn", "auto",
                # Sampling params on server startup
                "--temp", str(config.get("temperature", 0.6)),
                "--top-k", str(config.get("top_k", 40)),
                "--top-p", str(config.get("top_p", 0.9)),
                "--min-p", str(config.get("min_p", 0.05)),
                "--repeat-penalty", str(config.get("repeat_penalty", 1.3)),
                "--repeat-last-n", str(config.get("repeat_last_n", 256)),
                "--frequency-penalty", str(config.get("frequency_penalty", 0.1)),
                "--presence-penalty", str(config.get("presence_penalty", 0.1)),
                # DRY sampling
                "--dry-multiplier", str(config.get("dry_multiplier", 0.8)),
                "--dry-base", str(config.get("dry_base", 1.75)),
                "--dry-allowed-length", str(config.get("dry_allowed_length", 2)),
                "--dry-penalty-last-n", str(config.get("dry_penalty_last_n", -1)),
            ]
            
            # Add mirostat if enabled
            mirostat = config.get("mirostat", 0)
            if mirostat > 0:
                cmd.extend([
                    "--mirostat", str(mirostat),
                    "--mirostat-ent", str(config.get("mirostat_tau", 5.0)),
                    "--mirostat-lr", str(config.get("mirostat_eta", 0.1)),
                ])
            
            # Add typical_p if not disabled
            typical_p = config.get("typical_p", 1.0)
            if typical_p < 1.0:
                cmd.extend(["--typical", str(typical_p)])
            
            # Add chat template settings
            use_jinja = config.get("use_jinja", True)
            if use_jinja:
                cmd.append("--jinja")
            
            chat_template = config.get("chat_template", "auto")
            custom_chat_template = config.get("custom_chat_template", "")
            
            # Use custom template if provided, otherwise use preset (if not auto)
            if custom_chat_template:
                cmd.extend(["--chat-template", custom_chat_template])
            elif chat_template != "auto":
                cmd.extend(["--chat-template", chat_template])
            
            # Only load mmproj if image is provided
            if mmproj_path and image is not None:
                cmd.extend(["--mmproj", mmproj_path])

            # 4. Start Server
            self._wait_for_port_free(server_port)
            
            print(f"ComfyLLama: Starting One-Shot Server on port {server_port}...")
            print(f"ComfyLLama: Command: {' '.join(cmd)}")
            server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=os.path.dirname(server_exe))

            # 5. Wait for Health Check (Model Loading)
            load_timeout = 300 if mmproj_path else 120
            start_time = time.time()
            server_ready = False
            print(f"ComfyLLama: Waiting up to {load_timeout} seconds for model to load...")
            while time.time() - start_time < load_timeout:
                if server_process.poll() is not None:
                    return ("Server crashed during startup.",)
                
                try:
                    response = requests.get(f"http://localhost:{server_port}/health", timeout=1)
                    if response.status_code == 200:
                        server_ready = True
                        print("ComfyLLama: Model loaded successfully!")
                        break
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
            
            # Build payload with all sampling params
            payload = {
                "messages": messages,
                "n_predict": n_predict,
                "stop": [stop_string] if stop_string else [],
                # Sampling params (override server defaults if needed)
                "temperature": config.get("temperature", 0.6),
                "top_k": config.get("top_k", 40),
                "top_p": config.get("top_p", 0.9),
                "min_p": config.get("min_p", 0.05),
                "repeat_penalty": config.get("repeat_penalty", 1.3),
                "repeat_last_n": config.get("repeat_last_n", 256),
                "frequency_penalty": config.get("frequency_penalty", 0.1),
                "presence_penalty": config.get("presence_penalty", 0.1),
                # DRY
                "dry_multiplier": config.get("dry_multiplier", 0.8),
                "dry_base": config.get("dry_base", 1.75),
                "dry_allowed_length": config.get("dry_allowed_length", 2),
                "dry_penalty_last_n": config.get("dry_penalty_last_n", -1),
            }
            
            # Add seed if specified
            seed = config.get("seed", -1)
            if seed != -1:
                payload["seed"] = seed
            
            # Add mirostat to payload if enabled
            if mirostat > 0:
                payload["mirostat"] = mirostat
                payload["mirostat_tau"] = config.get("mirostat_tau", 5.0)
                payload["mirostat_eta"] = config.get("mirostat_eta", 0.1)
            
            # Add typical_p if not disabled
            if typical_p < 1.0:
                payload["typical_p"] = typical_p

            print("ComfyLLama: Sending prompt...")
            response = requests.post(f"http://localhost:{server_port}/chat/completions", json=payload, timeout=600)
            
            if response.status_code == 200:
                result = response.json()
                return (result["choices"][0]["message"]["content"],)
            else:
                return (f"Server Error {response.status_code}: {response.text}",)

        except Exception as e:
            return (f"Error: {e}",)

        finally:
            # 7. CLEANUP - STRICT
            if server_process:
                print("ComfyLLama: Killing server to release VRAM...")
                try:
                    server_process.terminate()
                    server_process.wait(timeout=2)
                except:
                    try:
                        server_process.kill()
                    except:
                        pass
                print("ComfyLLama: Server shutdown complete.")


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
                    "tooltip": "Image input for multimodal models. Requires mmproj_model."
                }),
                "mmproj_model": ("MODEL", {
                    "tooltip": "Vision projector model (mmproj) for multimodal. Required when using image input."
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
    DESCRIPTION = """Server-based Llama inference using llama-server.exe.

🔗 **Inputs:**
• gguf_model: Connect from GGUF Loader node
• config: Connect from ServerConfig node (sampling params)
• prompt: Your main prompt text
• text_input: Optional additional text from other nodes

🖼️ **Multimodal:**
• image: Connect image for vision models
• mmproj_model: Required vision projector for image input

💡 **Tips:**
• All sampling params are in the ServerConfig node
• Use external INT nodes to override ctx_size/n_predict limits"""

    def inference_llamacpp_server(
        self,
        gguf_model,
        config,
        prompt,
        text_input=None,
        image=None,
        mmproj_model=None,
        system_prompt="",
        stop_string="",
    ):
        # Combine text inputs
        final_prompt = prompt
        if text_input:
            if final_prompt:
                final_prompt = f"{final_prompt}\n{text_input}"
            else:
                final_prompt = text_input

        # Call run_inference with config
        return self.run_inference(
            gguf_model=gguf_model,
            config=config,
            prompt=final_prompt,
            image=image,
            mmproj_model=mmproj_model,
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

