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


class ComfyLLamaServer:
    """ComfyLLama node using llama-server.exe for GGUF inference via HTTP API.
    
    Server-based version for potentially better performance and API compatibility.
    """

    def _prepare_inputs(self, gguf_model, mmproj_model, media_paths, image, audio):
        """Shared input preparation logic for inference and command building."""
        # Handle gguf_model input
        if isinstance(gguf_model, str):
            gguf_path = gguf_model
        elif isinstance(gguf_model, dict) and 'path' in gguf_model:
            gguf_path = gguf_model['path']
        else:
            raise ValueError(f"Invalid gguf_model format: {type(gguf_model)}")

        if not gguf_path or not os.path.exists(gguf_path):
            raise ValueError(f"GGUF file not found: {gguf_path}")

        # Handle mmproj_model input
        mmproj_path = None
        if mmproj_model:
            if isinstance(mmproj_model, str):
                mmproj_path = mmproj_model
            elif isinstance(mmproj_model, dict) and 'path' in mmproj_model:
                mmproj_path = mmproj_model['path']
            else:
                raise ValueError(f"Invalid mmproj_model format: {type(mmproj_model)}")
            
            if not os.path.exists(mmproj_path):
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
                "ctx_size": ("INT", {"default": 16384, "min": 128, "max": 32768, "step": 128}),
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

    def _ensure_server_running(self, gguf_path, mmproj_path, n_gpu_layers, ctx_size, server_port, llama_cpp_folder):
        """Ensure llama-server is running with the specified model."""
        # Check if server is already running
        try:
            response = requests.get(f"http://localhost:{server_port}/health", timeout=1)
            if response.status_code == 200:
                # Server is running, check if it's the right model
                # For simplicity, assume it's the same; in production, might need to check
                return
        except:
            pass

        # Start the server
        server_exe = os.path.join(llama_cpp_folder, "llama-server.exe")
        if not os.path.exists(server_exe):
            raise ValueError(f"llama-server.exe not found at {server_exe}")

        cmd = [server_exe, "-m", gguf_path, "--port", str(server_port), "--ctx-size", str(ctx_size)]
        if n_gpu_layers >= 0:
            cmd.extend(["--gpu-layers", str(n_gpu_layers)])
        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])

        # Start in background and capture errors
        try:
            self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Wait a bit and check for immediate errors
            time.sleep(2)
            if self.server_process.poll() is not None:
                # Process exited early
                stdout, stderr = self.server_process.communicate()
                error_msg = stderr.strip() if stderr else "Unknown error"
                raise ValueError(f"llama-server failed to start: {error_msg}")
            
            # Wait for server to be ready
            time.sleep(3)
            
            # Check if started - retry health check
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response = requests.get(f"http://localhost:{server_port}/health", timeout=5)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 503:
                        # Server is starting up, wait longer
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                            continue
                        else:
                            raise ValueError(f"Server health check failed after {max_retries} attempts: {response.status_code}")
                    else:
                        raise ValueError(f"Server health check failed: {response.status_code}")
                except requests.RequestException as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise ValueError(f"Server health check failed after {max_retries} attempts: {e}")
            else:
                raise ValueError("Server health check timed out")
        except subprocess.TimeoutExpired:
            # Process is still running, assume it's starting
            time.sleep(5)
            try:
                response = requests.get(f"http://localhost:{server_port}/health", timeout=5)
                if response.status_code != 200:
                    raise ValueError(f"Server health check failed after timeout: {response.status_code}")
            except requests.RequestException as e:
                raise ValueError(f"Server health check failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to start server: {e}")

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

        # Prepare inputs
        try:
            gguf_path, mmproj_path, images, audios, use_multimodal = self._prepare_inputs(
                gguf_model, mmproj_model, media_paths, image, audio
            )
        except ValueError as e:
            return (f"Error: {e}",)

        # Ensure server is running
        self._ensure_server_running(gguf_path, mmproj_path, n_gpu_layers, ctx_size, server_port, llama_cpp_folder)

        # Prepare media if needed
        temp_files = []
        if '__temp_image__' in images:
            try:
                pil_image = ToPILImage()(image[0].permute(2, 0, 1) if image.dim() == 4 else image.permute(2, 0, 1))
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode="wb")
                pil_image.save(temp_file.name)
                temp_files.append(temp_file)
                image_path = temp_file.name
                for i, val in enumerate(images):
                    if val == '__temp_image__':
                        images[i] = image_path
            except Exception as e:
                return (f"Error preparing image: {e}",)

        # For multimodal, encode images to base64
        image_data = []
        if use_multimodal and images:
            import base64
            for img_path in images:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        image_data.append(f"data:image/png;base64,{base64.b64encode(f.read()).decode()}")

        # Prepare request payload
        payload = {
            "prompt": final_prompt,
            "temperature": temperature,
            "n_predict": n_predict if n_predict != -1 else -1,
            "repeat_penalty": repeat_penalty,
            "stop": [stop_string] if stop_string else [],
        }
        if seed != -1:
            payload["seed"] = seed
        if use_multimodal and image_data:
            payload["image_data"] = image_data

        try:
            response = requests.post(f"http://localhost:{server_port}/completion", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                output = result.get("content", "")
                return (output,)
            else:
                return (f"Server error: {response.status_code} {response.text}",)
        except Exception as e:
            return (f"Error: {e}",)
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except Exception:
                        pass
            # Kill the server process to free resources
            if hasattr(self, 'server_process') and self.server_process:
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                except Exception:
                    pass
                finally:
                    self.server_process = None


NODE_CLASS_MAPPINGS = {
    "ComfyLLamaServer": ComfyLLamaServer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLamaServer": "ComfyLLama (llama-server)",
}