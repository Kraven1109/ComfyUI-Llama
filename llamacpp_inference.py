import os
import subprocess
import tempfile
import torch
import shlex
import shutil
from pathlib import Path
import folder_paths
from torchvision.transforms import ToPILImage
import numpy as np
from scipy.io.wavfile import write as wav_write


class ComfyLLama:
    """ComfyLLama node using llama.cpp for GGUF inference.
    
    General-purpose one-shot node for any GGUF model supported by llama-cli or llama-mtmd-cli.
    Optimized for one-shot ComfyUI workflows (non-interactive mode).
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
                "text": ("STRING", {"default": "", "multiline": True}),
                "gguf_model": ("MODEL",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "ctx_size": ("INT", {"default": 2048, "min": 128, "max": 32768, "step": 128}),
                "cli_timeout": ("INT", {"default": 60, "min": 5, "max": 3600, "step": 5}),
                "interactive": ("BOOLEAN", {"default": False}),
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
                "n_predict": ("INT", {"default": -2, "min": -2, "max": 8192, "step": 16}),
                "use_jinja": ("BOOLEAN", {"default": False}),
                "jinja_chat_template": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": ""}),
            },
        }
        # Note: `paths` is intended to be the output of `MultiplePathsInput`,
        # which typically produces a list of dictionaries like:
        #   [{"type":"image", "image": "C:\path\to\file.png"}, ...]
        # The node will also accept a single path string, a single dict, or a
        # list of plain strings for backwards compatibility.

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_llamacpp"
    CATEGORY = "ComfyLLama"

    def inference_llamacpp(
        self,
        text,
        gguf_model,
        temperature,
        seed,
        n_gpu_layers,
        ctx_size,
        cli_timeout,
        interactive=False,
        repeat_penalty=1.1,
        llama_cpp_folder=r"d:\Apps\llama-cuda",
        image=None,
        audio=None,
        media_paths=None,
        mmproj_model=None,
        stop_string="",
        n_predict=-2,
        use_jinja=False,
        jinja_chat_template="",
        system_prompt="",
    ):
        # Hardcoded settings
        no_display_prompt = True
        strip_prompt = True

        # Jinja template validation
        BUILT_IN_TEMPLATES = {
            "bailing", "bailing-think", "bailing2", "chatglm3", "chatglm4", "chatml", "command-r",
            "deepseek", "deepseek2", "deepseek3", "exaone3", "exaone4", "falcon3", "gemma",
            "gigachat", "glmedge", "gpt-oss", "granite", "grok-2", "hunyuan-dense", "hunyuan-moe",
            "kimi-k2", "llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip", "llama3",
            "llama4", "megrez", "minicpm", "mistral-v1", "mistral-v3", "mistral-v3-tekken",
            "mistral-v7", "mistral-v7-tekken", "monarch", "openchat", "orion", "pangu-embedded",
            "phi3", "phi4", "rwkv-world", "seed_oss", "smolvlm", "vicuna", "vicuna-orca", "yandex", "zephyr"
        }
        if use_jinja:
            if jinja_chat_template in BUILT_IN_TEMPLATES:
                # Use as built-in template
                pass
            elif jinja_chat_template.strip():
                # Validate custom template string
                try:
                    import jinja2
                    jinja2.Template(jinja_chat_template)
                except Exception as e:
                    return (f"Error: Invalid Jinja template: {e}",)
            else:
                # Empty, disable jinja
                use_jinja = False

        # Prepare inputs using shared logic
        try:
            gguf_path, mmproj_path, images, audios, use_multimodal = self._prepare_inputs(
                gguf_model, mmproj_model, media_paths, image, audio
            )
        except ValueError as e:
            return (f"Error: {e}",)

        # Determine CLI executable path
        if not os.path.isdir(llama_cpp_folder):
            return (f"Error: llama_cpp_folder '{llama_cpp_folder}' is not a valid directory.",)
        cli_name = "llama-mtmd-cli.exe" if use_multimodal else "llama-cli.exe"
        llamacli_path = os.path.join(llama_cpp_folder, cli_name)

        if not os.path.exists(llamacli_path):
            return (f"Error: {cli_name} not found at {llamacli_path}. Please ensure llama.cpp is installed in the specified folder.",)

        # Argument candidates based on CLI type
        if use_multimodal:
            arg_candidates = {
                "-m", "-p", "--predict", "--temp", "--ctx-size", "--repeat-penalty",
                "--gpu-layers", "--seed", "--jinja", "--chat-template",
                "--system-prompt", "--mmproj", "--image", "--audio",
                "--no-perf", "--threads"
            }
        else:
            arg_candidates = {
                "-m", "-p", "--predict", "--temp", "--ctx-size", "--repeat-penalty",
                "--single-turn", "--gpu-layers", "--seed", "--jinja", "--chat-template",
                "--system-prompt", "--reverse-prompt", "--no-perf", "--threads",
                "--no-display-prompt"
            }

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
        if '__temp_audio__' in audios:
            try:
                # Handle AUDIO: assume dict with 'waveform' and 'sample_rate'
                if isinstance(audio, dict):
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                else:
                    # Assume tensor
                    waveform = audio
                    sample_rate = 44100  # default
                # Convert to numpy
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                # Normalize to int16
                if waveform.dtype != np.int16:
                    waveform = (waveform * 32767).astype(np.int16)
                # Save as wav
                import scipy.io.wavfile
                temp_file_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb")
                scipy.io.wavfile.write(temp_file_audio.name, sample_rate, waveform)
                temp_files.append(temp_file_audio)
                audio_path = temp_file_audio.name
                for i, val in enumerate(audios):
                    if val == '__temp_audio__':
                        audios[i] = audio_path
            except Exception as e:
                return (f"Error preparing audio: {e}",)

        try:
            # Clamp n_predict to safe range
            if n_predict != -2:
                n_predict = min(max(1, n_predict), ctx_size, 8192)

            # Build the final command
            cmd = [llamacli_path]
            if "-m" in arg_candidates:
                cmd.extend(["-m", gguf_path])
            if "-p" in arg_candidates:
                cmd.extend(["-p", text])
            if "--predict" in arg_candidates:
                cmd.extend(["--predict", str(n_predict)])
            if "--temp" in arg_candidates:
                cmd.extend(["--temp", str(temperature)])
            if "--ctx-size" in arg_candidates:
                cmd.extend(["--ctx-size", str(ctx_size)])
            if "--repeat-penalty" in arg_candidates:
                cmd.extend(["--repeat-penalty", str(repeat_penalty)])
            # Apply single-turn conversation enforcement unless interactive mode requested
            if not interactive and "--single-turn" in arg_candidates:
                cmd.append("--single-turn")

            # Add multimodal support and add multiple images/audio if provided
            if use_multimodal:
                # mmproj if supplied
                if mmproj_path and "--mmproj" in arg_candidates:
                    cmd.extend(["--mmproj", mmproj_path])
                # Add images and audio paths discovered earlier only if CLI supports these args
                if "--image" in arg_candidates and images:
                    for img in images:
                        if img == '__temp_image__':
                            # will handle the temporary image path below
                            continue
                        cmd.extend(["--image", img])
                if "--audio" in arg_candidates and audios:
                    for aud in audios:
                        cmd.extend(["--audio", aud])
            # Add images/audio only when multimodal and using mtmd CLI (handled above)

            # GPU layers
            if n_gpu_layers >= 0 and "--gpu-layers" in arg_candidates:
                cmd.extend(["--gpu-layers", str(n_gpu_layers)])

            # Seed
            if seed != -1 and "--seed" in arg_candidates:
                cmd.extend(["--seed", str(seed)])

            # Jinja/chat template
            if use_jinja and "--jinja" in arg_candidates:
                cmd.append("--jinja")
                if jinja_chat_template and "--chat-template" in arg_candidates:
                    cmd.extend(["--chat-template", jinja_chat_template])

            # System prompt
            if system_prompt and system_prompt.strip() and "--system-prompt" in arg_candidates:
                cmd.extend(["--system-prompt", system_prompt])

            # Stop string
            if stop_string and stop_string.strip() and "--reverse-prompt" in arg_candidates:
                cmd.extend(["--reverse-prompt", stop_string])

            # No display prompt
            if no_display_prompt and "--no-display-prompt" in arg_candidates:
                cmd.append("--no-display-prompt")

            # Debug logging
            debug_command = " ".join(shlex.quote(part) for part in cmd)
            debug_log_path = os.path.join(folder_paths.temp_directory, 'llama_cmd_debug.log')
            
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[CMD] {debug_command}\n")
                    f.write(f"[SELECTED_CLI] {cli_name} at {llamacli_path}\n")
                    f.write(f"[IMAGES] {images}\n")
                    f.write(f"[AUDIOS] {audios}\n")
            except Exception:
                pass

            # Execute command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=int(cli_timeout),
            )

            stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""

            # Check for errors
            if result.returncode != 0:
                # If we have substantial output, return it anyway
                if stdout_text.strip() and len(stdout_text.strip()) > 10:
                    return (stdout_text.strip(),)
                return (f"{cli_name} failed (code {result.returncode}): {stderr_text}",)

            # Return output
            output = stdout_text.strip()
            # Optional minimal prompt stripping: remove exact prompt prefix if echoed back
            if strip_prompt and isinstance(text, str) and text.strip():
                try:
                    plain_prompt = text.strip()
                    if output.startswith(plain_prompt):
                        output = output[len(plain_prompt):].strip()
                except Exception:
                    pass
            if not output:
                return (f"Warning: Empty output from {cli_name}",)

            return (output,)

        except subprocess.TimeoutExpired:
            return (f"Error: Command timed out after {cli_timeout}s",)
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

    def build_command(self,
                      text,
                      gguf_model,
                      temperature,
                      seed,
                      n_gpu_layers,
                      ctx_size,
                      interactive=False,
                      repeat_penalty=1.1,
                      llama_cpp_folder=r"d:\Apps\llama-cuda",
                      image=None,
                      audio=None,
                      media_paths=None,
                      mmproj_model=None,
                      stop_string="",
                      n_predict=-2,
                      use_jinja=False,
                      jinja_chat_template="",
                      system_prompt="",
                      ):
        """Build the command that would be executed (without running it).
        Returns tuple: (cmd, cli_name, llamacli_path, images, audios)
        """
        # Prepare inputs using shared logic
        try:
            gguf_path, mmproj_path, images, audios, use_multimodal = self._prepare_inputs(
                gguf_model, mmproj_model, media_paths, image, audio
            )
        except ValueError as e:
            return ([], f"error: {e}", "", [], [])

        # Determine binary and arg candidates
        cli_name = "llama-mtmd-cli.exe" if use_multimodal else "llama-cli.exe"
        llamacli_path = os.path.join(llama_cpp_folder, cli_name)

        if use_multimodal:
            arg_candidates = {
                "-m", "-p", "--predict", "--temp", "--ctx-size", "--repeat-penalty",
                "--gpu-layers", "--seed", "--jinja", "--chat-template",
                "--system-prompt", "--mmproj", "--image", "--audio",
                "--no-perf", "--threads"
            }
        else:
            arg_candidates = {
                "-m", "-p", "--predict", "--temp", "--ctx-size", "--repeat-penalty",
                "--single-turn", "--gpu-layers", "--seed", "--jinja", "--chat-template",
                "--system-prompt", "--reverse-prompt", "--no-perf", "--threads",
                "--no-display-prompt"
            }

        if n_predict != -2:
            n_predict = min(max(1, n_predict), ctx_size, 8192)

        cmd = [llamacli_path]
        if "-m" in arg_candidates:
            cmd.extend(["-m", gguf_path])
        if "-p" in arg_candidates:
            cmd.extend(["-p", text])
        if "--predict" in arg_candidates:
            cmd.extend(["--predict", str(n_predict)])
        if "--temp" in arg_candidates:
            cmd.extend(["--temp", str(temperature)])
        if "--ctx-size" in arg_candidates:
            cmd.extend(["--ctx-size", str(ctx_size)])
        if "--repeat-penalty" in arg_candidates:
            cmd.extend(["--repeat-penalty", str(repeat_penalty)])
        if not interactive and "--single-turn" in arg_candidates:
            cmd.append("--single-turn")
        if use_multimodal:
            if mmproj_path and "--mmproj" in arg_candidates:
                cmd.extend(["--mmproj", mmproj_path])
            if "--image" in arg_candidates and images:
                for img in images:
                    if img == '__temp_image__':
                        continue
                    cmd.extend(["--image", img])
            if "--audio" in arg_candidates and audios:
                for aud in audios:
                    cmd.extend(["--audio", aud])
        if n_gpu_layers >= 0 and "--gpu-layers" in arg_candidates:
            cmd.extend(["--gpu-layers", str(n_gpu_layers)])
        if seed != -1 and "--seed" in arg_candidates:
            cmd.extend(["--seed", str(seed)])
        if use_jinja and "--jinja" in arg_candidates:
            cmd.append("--jinja")
            if jinja_chat_template and "--chat-template" in arg_candidates:
                cmd.extend(["--chat-template", jinja_chat_template])
        if system_prompt and system_prompt.strip() and "--system-prompt" in arg_candidates:
            cmd.extend(["--system-prompt", system_prompt])
        if stop_string and stop_string.strip() and "--reverse-prompt" in arg_candidates:
            cmd.extend(["--reverse-prompt", stop_string])
        # No display prompt hardcoded to True
        cmd.append("--no-display-prompt")

        return (cmd, cli_name, llamacli_path, images, audios)


NODE_CLASS_MAPPINGS = {
    "ComfyLLama": ComfyLLama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLLama": "ComfyLLama (llama.cpp)",
}