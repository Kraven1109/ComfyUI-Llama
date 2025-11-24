import os
import tempfile
import torch
import shlex
import shutil
from pathlib import Path
import folder_paths
from torchvision.transforms import ToPILImage
import numpy as np
from scipy.io.wavfile import write as wav_write
import requests
import json
import time
import subprocess
import threading
import socket
import io
import base64
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

    def _prepare_inputs(self, gguf_model, mmproj_model, media_paths, image, audio):
        """Shared input preparation logic for inference and command building."""
        # Handle gguf_model input
        gguf_path = self._resolve_path(gguf_model)
        if not gguf_path or not os.path.exists(gguf_path):
            raise ValueError(f"GGUF file not found: {gguf_path}")

        # Handle mmproj_model input
        mmproj_path = self._resolve_path(mmproj_model)
        if mmproj_model and not mmproj_path:
            raise ValueError(f"Invalid mmproj_model format: {type(mmproj_model)}")
        if mmproj_path and not os.path.exists(mmproj_path):
            raise ValueError(f"MMProj file not found: {mmproj_path}")

        # Enforce exclusive use: cannot use media_paths with any single media
        if media_paths is not None and any([image, audio]):
            raise ValueError("Cannot provide both single media inputs and media_paths. Choose single or multiple.")

        # Normalize and detect multimodal usage (image/audio or mmproj)
        IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}
        AUDIO_EXT = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}

        # Build candidate paths list from provided inputs
        candidate_paths = []
        if media_paths is not None:
            if isinstance(media_paths, list):
                candidate_paths.extend(media_paths)
            else:
                candidate_paths.append(media_paths)

        # Classify by extension
        images = []
        audios = []
        others = []
        for p in candidate_paths:
            if isinstance(p, str) and p:
                ext = os.path.splitext(p.lower())[1]
                if ext in IMAGE_EXT:
                    images.append(p)
                elif ext in AUDIO_EXT:
                    audios.append(p)
                else:
                    others.append(p)
            else:
                others.append(str(p))  # invalid

        # Handle tensor inputs
        if image is not None:
            images.append('__temp_image__')
        if audio is not None:
            audios.append('__temp_audio__')

        # Robust multimodal detection: require mmproj for images/audio
        has_multimodal_input = bool(images) or bool(audios)
        if has_multimodal_input and not mmproj_path:
            raise ValueError("Multimodal input (images/audio) detected but no mmproj model provided. Multimodal models require an mmproj file for vision/audio processing.")
        
        use_multimodal = has_multimodal_input

        return gguf_path, mmproj_path, images, audios, use_multimodal

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
                "n_predict": ("INT", {"default": -1, "min": -2, "max": 8192, "step": 16}),
                "use_jinja": ("BOOLEAN", {"default": False}),
                "jinja_chat_template": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": ""}),
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_llamacpp_server"
    CATEGORY = "ComfyLLama"

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
                "n_predict": ("INT", {"default": -1, "min": -2, "max": 8192, "step": 16}),
                "use_jinja": ("BOOLEAN", {"default": False}),
                "jinja_chat_template": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": ""}),
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_llamacpp_server"
    CATEGORY = "ComfyLLama"

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
    CATEGORY = "ComfyLLama"

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
    CATEGORY = "ComfyLLama"

    def text_concat(self, delimiter, text1=None, text2=None, text3=None, text4=None, text5=None):
        texts = [t for t in [text1, text2, text3, text4, text5] if t is not None]
        return (delimiter.join(texts),)


NODE_CLASS_MAPPINGS = {
    "ComfyLLamaServer": ComfyLLamaServer,
    "ComfyLLamaTextInput": ComfyLLamaTextInput,
    "ComfyLLamaTextConcat": ComfyLLamaTextConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLamaServer": "ComfyLLama (llama-server)",
    "ComfyLLamaTextInput": "Text Input",
    "ComfyLLamaTextConcat": "Text Concat",
}
