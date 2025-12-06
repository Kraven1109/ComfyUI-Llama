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

    def run_inference(self, gguf_model, llama_cpp_folder, server_port, prompt, 
                     n_gpu_layers, ctx_size, temperature, max_tokens, repeat_penalty,
                     image=None, mmproj_model=None, system_prompt="", stop_string="", seed=-1):

        server_process = None
        
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

            # 3. Build Command
            cmd = [
                server_exe,
                "-m", gguf_path,
                "--port", str(server_port),
                "--ctx-size", str(ctx_size),
                "--n-gpu-layers", str(n_gpu_layers),
                "--threads", "-1",
                "--flash-attn", "auto",
            ]
            
            # Only load mmproj if image is provided
            if mmproj_path and image is not None:
                cmd.extend(["--mmproj", mmproj_path])

            # 4. Start Server
            self._wait_for_port_free(server_port)
            
            print(f"ComfyLLama: Starting One-Shot Server on port {server_port}...")
            print(f"ComfyLLama: Command: {' '.join(cmd)}")
            # Use separate process group on windows to ensure clean kill, or just Popen
            server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, cwd=os.path.dirname(server_exe))

            # 5. Wait for Health Check (Model Loading)
            # Give it time to load the model into VRAM
            # Multimodal models take longer due to vision component
            load_timeout = 300 if mmproj_path else 120  # 5 min for multimodal, 2 min for text-only
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
                    time.sleep(1)  # Check every 1 seconds for multimodal
            
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
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "n_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
                "stop": [stop_string] if stop_string else [],
            }
            
            if seed != -1:
                payload["seed"] = seed

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
                "prefix": ("STRING", {"default": "", "multiline": True}),
                "gguf_model": ("MODEL",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "ctx_size": ("INT", {"default": 32768, "min": 128, "max": 32768, "step": 128}),
                "server_port": ("INT", {"default": 8080, "min": 1024, "max": 65535, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 3.0, "step": 0.1}),
                "llama_cpp_folder": ("STRING", {"default": r"d:\Apps\llama-cuda"}),
            },
            "optional": {
                # Separate inputs for different media types
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "media_paths": ("PATH",),
                # Model and prompt options (unchanged)
                "mmproj_model": ("MODEL",),
                "stop_string": ("STRING", {"default": ""}),
                "n_predict": ("INT", {"default": -1, "min": -2, "max": 8192, "step": 1, "tooltip": "-1 = infinite (until EOS), -2 = fill context, 0+ = exact token limit"}),
                "use_jinja": ("BOOLEAN", {"default": False}),
                "jinja_chat_template": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": ""}),
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_llamacpp_server"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Server-based Llama inference using llama-server.exe. Faster than CLI for multimodal models. Supports text and image inputs via HTTP API."

    def inference_llamacpp_server(
        self,
        prefix,
        gguf_model,
        temperature,
        seed,
        n_gpu_layers,
        ctx_size,
        server_port,
        repeat_penalty,
        llama_cpp_folder=r"d:\Apps\llama-cuda",
        text_input=None,
        image=None,
        audio=None,
        media_paths=None,
        mmproj_model=None,
        stop_string="",
        n_predict=-1,
        use_jinja=False,
        jinja_chat_template="",
        system_prompt="",
    ):
        # Combine text inputs
        final_prompt = prefix
        if text_input:
            if final_prompt:
                final_prompt = f"{final_prompt}\n{text_input}"
            else:
                final_prompt = text_input

        # For now, only support image input, not audio or media_paths
        if audio is not None or media_paths is not None:
            return ("Error: Audio and media_paths inputs not yet supported in server mode. Use image input only.",)

        # Call the improved run_inference method
        return self.run_inference(
            gguf_model=gguf_model,
            llama_cpp_folder=llama_cpp_folder,
            server_port=server_port,
            prompt=final_prompt,
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
            temperature=temperature,
            max_tokens=n_predict if n_predict != -1 else -1,
            repeat_penalty=repeat_penalty,
            image=image,
            mmproj_model=mmproj_model,
            system_prompt=system_prompt,
            stop_string=stop_string,
            seed=seed
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
    """Preview text output in the ComfyUI interface with markdown support."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "title": ("STRING", {"default": "Output"}),
            },
        }

    INPUT_IS_LIST = False
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "preview_text"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Preview text output in the node. Supports markdown formatting. Pass-through: outputs the same text for chaining."

    def preview_text(self, text, title="Output"):
        # Return UI data for display and pass through the text
        return {"ui": {"text": [text], "title": [title]}, "result": (text,)}


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
    "ComfyLLamaServer": ComfyLLamaServer,
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
    "ComfyLLamaPreviewText": ComfyLLamaPreviewText,
    "ComfyLLamaSaveText": ComfyLLamaSaveText,
    "ComfyLLamaPromptBuilder": ComfyLLamaPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLamaServer": "🦙 LLama Server",
    "ComfyLLamaTextInput": "📝 Text Input",
    "ComfyLLamaTextConcat": "🔗 Text Concat",
    "ComfyLLamaPreviewText": "👁️ Preview Text",
    "ComfyLLamaSaveText": "💾 Save Text",
    "ComfyLLamaPromptBuilder": "🛠️ Prompt Builder",
}

