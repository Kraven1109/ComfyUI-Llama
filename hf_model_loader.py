"""
HuggingFace Model Loader for ComfyUI-LLama

Loads Vision-Language models from HuggingFace (like Qwen2-VL, Step3-VL, etc.)
with optional 4-bit/8-bit quantization using bitsandbytes.
"""

import os
import gc
import torch
import folder_paths
from pathlib import Path


class HFModelLoader:
    """Load HuggingFace Vision-Language models with quantization support.
    
    Downloads models to [ComfyUI]/models/LLM/ directory.
    Supports bitsandbytes quantization (4-bit, 8-bit) for reduced VRAM usage.
    """
    
    # Popular VL models for quick selection
    POPULAR_MODELS = [
        "huihui-ai/Huihui-Step3-VL-10B-abliterated",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-2B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "microsoft/Phi-4-multimodal-instruct",
        "Custom (use model_id field)",
    ]
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_preset": (s.POPULAR_MODELS, {
                    "default": s.POPULAR_MODELS[0],
                    "tooltip": "Select a popular VL model or 'Custom' to use model_id field"
                }),
                "quantization": (["none", "4bit", "8bit"], {
                    "default": "4bit",
                    "tooltip": "Quantization mode. 4bit/8bit uses bitsandbytes to reduce VRAM. 'none' loads full precision."
                }),
                "torch_dtype": (["auto", "float16", "bfloat16", "float32"], {
                    "default": "auto",
                    "tooltip": "Data type for model weights. 'auto' = model default, 'float16' for most GPUs, 'bfloat16' for newer GPUs (30xx+)"
                }),
                "device_map": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device placement. 'auto' = automatic, 'cuda' = GPU only, 'cpu' = CPU only"
                }),
                "trust_remote_code": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Trust remote code from HuggingFace. Required for some models (Qwen, etc.)"
                }),
            },
            "optional": {
                "model_id": ("STRING", {
                    "default": "",
                    "tooltip": "Custom HuggingFace model ID. Used when model_preset is 'Custom'."
                }),
                "revision": ("STRING", {
                    "default": "main",
                    "tooltip": "Model revision/branch to use (default: main)"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for gated models (like Llama). Leave empty for public models."
                }),
                "flash_attention": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use advanced attention (Flash Attn 2 or SDPA). SDPA can leverage Sage/Radial attention in ComfyUI."
                }),
                "max_memory_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "tooltip": "Max GPU memory in GB. 0 = use all available. Useful to reserve memory for other tasks."
                }),
            },
        }

    RETURN_TYPES = ("HF_MODEL", "HF_PROCESSOR",)
    RETURN_NAMES = ("model", "processor",)
    FUNCTION = "load_model"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = """Load HuggingFace Vision-Language models.

📦 **Models download to:** [ComfyUI]/models/LLM/

⚡ **Quantization:**
• 4bit: ~4GB VRAM for 7B model (recommended)
• 8bit: ~8GB VRAM for 7B model
• none: Full precision (~14GB for 7B)

� **Attention (if you have Sage/Radial attention):**
• Flash Attention ON = Uses Flash Attn 2 or SDPA
• SDPA automatically leverages Sage/Radial if available
• Faster inference, lower memory

🔧 **Supported Models:**
• Standard: Qwen2-VL, Llava, Phi-4, Llama-Vision
• Custom: Huihui Step3-VL (auto-detected)

💡 **Tips:**
• First load downloads (~15GB, may take 5-10min)
• Use 4bit for most use cases"""

    def load_model(
        self,
        model_preset,
        quantization,
        torch_dtype,
        device_map,
        trust_remote_code,
        model_id="",
        revision="main",
        hf_token="",
        flash_attention=True,
        max_memory_gb=0.0,
    ):
        # Determine model ID
        if model_preset == "Custom (use model_id field)":
            if not model_id:
                raise ValueError("Please provide a model_id when using 'Custom' preset")
            final_model_id = model_id
        else:
            final_model_id = model_preset
        
        print(f"[HFLoader] Loading model: {final_model_id}")
        
        # Set cache directory to ComfyUI models/LLM
        cache_dir = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[HFLoader] Cache directory: {cache_dir}")
        
        # Import transformers
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor, 
                BitsAndBytesConfig
            )
            # Use the new API if available, fallback to deprecated one
            try:
                from transformers import AutoModelForImageTextToText
                VisionModel = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq
                VisionModel = AutoModelForVision2Seq
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers accelerate")
        
        # Prepare kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
            "revision": revision,
        }
        
        processor_kwargs = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
            "revision": revision,
        }
        
        # Add HF token if provided
        if hf_token:
            model_kwargs["token"] = hf_token
            processor_kwargs["token"] = hf_token
        
        # Set dtype (use 'dtype' for newer API, fallback to 'torch_dtype' for compatibility)
        dtype_value = None
        if torch_dtype == "auto":
            dtype_value = "auto"
        elif torch_dtype == "float16":
            dtype_value = torch.float16
        elif torch_dtype == "bfloat16":
            dtype_value = torch.bfloat16
        else:
            dtype_value = torch.float32
        
        # Try new 'dtype' param first, fallback to 'torch_dtype'
        model_kwargs["dtype"] = dtype_value
        
        # Set device map
        model_kwargs["device_map"] = device_map
        
        # Handle max memory
        if max_memory_gb > 0:
            model_kwargs["max_memory"] = {0: f"{max_memory_gb}GB"}
        
        # Configure quantization
        if quantization in ["4bit", "8bit"]:
            try:
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:  # 8bit
                    quant_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                model_kwargs["quantization_config"] = quant_config
                print(f"[HFLoader] Using {quantization} quantization")
            except Exception as e:
                print(f"[HFLoader] Warning: BitsAndBytes quantization failed: {e}")
                print("[HFLoader] Falling back to full precision")
        
        # Attention implementation
        # Try to use advanced attention if available (sage/flash/sdpa)
        if flash_attention:
            # Try Flash Attention 2 first
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("[HFLoader] Using Flash Attention 2")
            except ImportError:
                # Fallback to SDPA (PyTorch scaled dot product attention)
                # This automatically uses sage/radial attention if available in ComfyUI
                try:
                    model_kwargs["attn_implementation"] = "sdpa"
                    print("[HFLoader] Using SDPA (scaled dot product attention)")
                except Exception:
                    print("[HFLoader] Using default attention")
        
        # Clear VRAM before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load processor first (smaller, faster)
        print("[HFLoader] Loading processor...")
        processor = AutoProcessor.from_pretrained(final_model_id, **processor_kwargs)
        
        # Load model - try VisionModel first, fallback to CausalLM for custom models
        print("[HFLoader] Loading model (this may take a while on first run)...")
        
        # For custom models with local code, add to sys.path
        import sys
        model_cache_path = None
        
        # Helper to try loading with dtype fallback
        def try_load_model(model_class, model_id, kwargs):
            """Try loading with 'dtype', fallback to 'torch_dtype' if needed."""
            try:
                return model_class.from_pretrained(model_id, **kwargs)
            except TypeError as te:
                if "'dtype'" in str(te):
                    # Fallback to torch_dtype for older models
                    fallback_kwargs = kwargs.copy()
                    if "dtype" in fallback_kwargs:
                        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                    return model_class.from_pretrained(model_id, **fallback_kwargs)
                raise
        
        try:
            # Try standard VL API (ImageTextToText or Vision2Seq)
            model = try_load_model(VisionModel, final_model_id, model_kwargs)
            print("[HFLoader] Loaded as VL model")
        except (ValueError, KeyError, ImportError, TypeError) as e:
            # Fallback to CausalLM for custom VL models (Huihui, custom architectures)
            error_msg = str(e)
            if ("Unrecognized configuration class" in error_msg or 
                "StepRoboticsConfig" in error_msg or
                "configuration_step_vl" in error_msg or
                "vision_encoder" in error_msg):
                
                print("[HFLoader] Model uses custom architecture, loading as CausalLM...")
                print("[HFLoader] Adding model cache to sys.path for custom imports...")
                
                # Get the model cache directory and add to sys.path
                # This allows importing custom modules like configuration_step_vl
                from huggingface_hub import snapshot_download
                try:
                    model_cache_path = snapshot_download(
                        repo_id=final_model_id,
                        cache_dir=cache_dir,
                        revision=revision,
                        token=hf_token if hf_token else None,
                        ignore_patterns=["*.bin", "*.safetensors"],  # Just get the code files
                    )
                    if model_cache_path and model_cache_path not in sys.path:
                        sys.path.insert(0, model_cache_path)
                        print(f"[HFLoader] Added {model_cache_path} to sys.path")
                except Exception as download_err:
                    print(f"[HFLoader] Warning: Could not add model path to sys.path: {download_err}")
                
                # Now try loading with custom modules available
                model = try_load_model(AutoModelForCausalLM, final_model_id, model_kwargs)
                print("[HFLoader] Loaded as CausalLM with custom config")
            else:
                raise
        
        print(f"[HFLoader] Model loaded successfully!")
        if hasattr(model, 'device'):
            print(f"[HFLoader] Device: {model.device}")
        
        # Return model info dict for the inference node
        model_info = {
            "model": model,
            "model_id": final_model_id,
            "quantization": quantization,
        }
        
        processor_info = {
            "processor": processor,
            "model_id": final_model_id,
        }
        
        return (model_info, processor_info)


class HFModelUnloader:
    """Unload HuggingFace model to free VRAM.
    
    Use this node after inference to release GPU memory.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HF_MODEL",),
                "processor": ("HF_PROCESSOR",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Unload HuggingFace model to free VRAM. Connect after inference is done."

    def unload(self, model, processor):
        print("[HFUnloader] Unloading model...")
        
        if model and "model" in model:
            del model["model"]
        if processor and "processor" in processor:
            del processor["processor"]
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[HFUnloader] Model unloaded, VRAM freed")
        return ()
