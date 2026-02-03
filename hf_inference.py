"""
HuggingFace Vision-Language Inference for ComfyUI-LLama

Runs inference using HuggingFace VL models (Qwen2-VL, Step3-VL, Phi-4, etc.)
Smart caching: Keep model in VRAM across runs, only unload when changing models.
"""

import os
import gc
import torch
import numpy as np
from PIL import Image
import folder_paths

# Global model cache for smart persistence
_HF_MODEL_CACHE = {
    "model": None,
    "processor": None,
    "model_id": None,
}


def _get_cached_model(model_info, processor_info):
    """Get model from cache or from input dict. Handles ComfyUI dict reuse."""
    global _HF_MODEL_CACHE
    
    model_id = model_info.get("model_id", "unknown")
    
    # Check if we have a valid model in cache
    if (_HF_MODEL_CACHE["model"] is not None and 
        _HF_MODEL_CACHE["model_id"] == model_id):
        print(f"[HFInference] Reusing cached model: {model_id}")
        return _HF_MODEL_CACHE["model"], _HF_MODEL_CACHE["processor"]
    
    # Check if model is in input dict (fresh from loader)
    if "model" in model_info and model_info["model"] is not None:
        hf_model = model_info["model"]
        hf_processor = processor_info.get("processor")
        
        # Cache it for reuse
        _HF_MODEL_CACHE["model"] = hf_model
        _HF_MODEL_CACHE["processor"] = hf_processor
        _HF_MODEL_CACHE["model_id"] = model_id
        print(f"[HFInference] Cached model: {model_id}")
        
        return hf_model, hf_processor
    
    # No valid model found
    return None, None


def _unload_cached_model():
    """Unload model from global cache to free VRAM."""
    global _HF_MODEL_CACHE
    
    if _HF_MODEL_CACHE["model"] is not None:
        model_id = _HF_MODEL_CACHE["model_id"]
        print(f"[HFInference] Unloading cached model: {model_id}")
        
        del _HF_MODEL_CACHE["model"]
        del _HF_MODEL_CACHE["processor"]
        _HF_MODEL_CACHE["model"] = None
        _HF_MODEL_CACHE["processor"] = None
        _HF_MODEL_CACHE["model_id"] = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[HFInference] Model unloaded, VRAM freed")
        return True
    return False


class HFVLInference:
    """Run inference with HuggingFace Vision-Language models.
    
    Supports multimodal input (images + text) for vision-language models.
    Uses smart caching: model stays in VRAM across runs until explicitly unloaded.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HF_MODEL", {
                    "tooltip": "HuggingFace model from HFModelLoader"
                }),
                "processor": ("HF_PROCESSOR", {
                    "tooltip": "Processor from HFModelLoader"
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "User prompt for the model"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image input (ComfyUI IMAGE tensor)"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "System prompt to set model behavior"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. 0 = deterministic, higher = more creative"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold (recommended: 0.95 for VL models)"
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Min-p sampling. Scales based on top token probability. More intelligent than top_p."
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 0,
                    "tooltip": "Top-k sampling. 0 = disabled. Recommended: 20 for VL models."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeating tokens. 1.0 = no penalty. Keep at 1.0 for JSON generation!"
                }),
                "do_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use sampling. False = greedy decoding (deterministic, recommended for VL tasks)"
                }),
                "strip_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Strip thinking/reasoning tags from output. Some models include <think>...</think> reasoning."
                }),
                "text_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Additional text input from other nodes"
                }),
                "auto_unload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-unload model after inference to free VRAM. Recommended for one-shot workflows."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = """Vision-Language inference with HuggingFace models.

🔗 **Inputs:**
• model/processor: From HFModelLoader
• prompt: Your text prompt
• image: Optional image for VL models

⚡ **Generation params:**
• temperature: Creativity (0=deterministic)
• top_p/top_k: Sampling diversity
• max_new_tokens: Output length limit

💡 **Tips:**
• For image description: Connect image + describe prompt
• For text-only: Leave image disconnected
• Use do_sample=False for deterministic output"""

    def _tensor_to_pil(self, image_tensor):
        """Convert ComfyUI IMAGE tensor to PIL Image."""
        # Handle batch dimension
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]  # Take first image
        
        # Convert from 0-1 float to 0-255 uint8
        img_np = (255. * image_tensor.cpu().numpy()).astype(np.uint8)
        return Image.fromarray(img_np)

    def _strip_thinking_tags(self, text):
        """Strip thinking/reasoning tags from model output.
        
        Many "thinking" models (Qwen3, Step3-VL, etc.) output internal reasoning 
        in <think>...</think> blocks. This method extracts only the final answer.
        
        Based on chat template logic:
        - If </think> is in content, split and take what's after </think>
        - Remove any remaining <think> or </think> tags
        """
        import re
        
        result = text
        
        # Method 1: If </think> exists, take content after it (most reliable)
        if '</think>' in result:
            # Split on </think> and take the last part
            parts = result.split('</think>')
            result = parts[-1].strip()
        
        # Method 2: Remove any remaining <think> blocks
        result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)
        
        # Remove standalone opening/closing tags
        result = re.sub(r'</?think>\s*', '', result)
        
        return result.strip()

    def generate(
        self,
        model,
        processor,
        prompt,
        image=None,
        system_prompt="You are a helpful assistant.",
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        min_p=0.05,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=False,
        strip_thinking=True,
        text_input=None,
        auto_unload=True,
    ):
        # Get model from cache or input (handles ComfyUI dict reuse)
        hf_model, hf_processor = _get_cached_model(model, processor)
        
        if hf_model is None or hf_processor is None:
            return ("Error: Invalid model. Please connect HFModelLoader output or re-run the loader node.",)
        
        model_id = model.get("model_id", _HF_MODEL_CACHE.get("model_id", "unknown"))
        
        # Combine prompts
        final_prompt = prompt
        if text_input:
            if final_prompt:
                final_prompt = f"{final_prompt}\n{text_input}"
            else:
                final_prompt = text_input
        
        try:
            print(f"[HFInference] Running inference with {model_id}")
            
            # Check if this is a Step3-VL or Huihui model (needs special handling)
            is_step3_model = "step3" in model_id.lower() or "huihui" in model_id.lower()
            
            # Prepare image if provided
            pil_image = None
            temp_image_path = None
            if image is not None:
                pil_image = self._tensor_to_pil(image)
                print(f"[HFInference] Image size: {pil_image.size}")
                
                # For Step3-VL models, save to temp file (they expect file paths)
                if is_step3_model:
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_image_path = temp_file.name
                    pil_image.save(temp_image_path)
                    temp_file.close()
                    print(f"[HFInference] Saved temp image for Step3-VL: {temp_image_path}")
            
            # Prepare messages in chat format
            messages = []
            
            # Note: Step3-VL doesn't use system prompt in the same way
            if system_prompt and not is_step3_model:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Prepare user message content
            user_content = []
            
            # Add image if provided
            if image is not None:
                if is_step3_model and temp_image_path:
                    # Step3-VL expects file path
                    user_content.append({
                        "type": "image",
                        "image": temp_image_path,
                    })
                else:
                    # Other models can use PIL directly
                    user_content.append({
                        "type": "image",
                        "image": pil_image,
                    })
            
            # Add text
            user_content.append({
                "type": "text",
                "text": final_prompt,
            })
            
            messages.append({
                "role": "user",
                "content": user_content,
            })
            
            # Apply chat template
            print("[HFInference] Applying chat template...")
            
            # For models that use apply_chat_template (Qwen, Llava, Huihui, etc.)
            if hasattr(hf_processor, 'apply_chat_template'):
                try:
                    # Try tokenize=True first (for models like Huihui/Step3-VL)
                    try:
                        inputs = hf_processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        print("[HFInference] Using direct tokenization (tokenize=True)")
                    except (TypeError, AttributeError):
                        # Fallback to tokenize=False for models that don't support it
                        text = hf_processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        # Process with images
                        if image is not None:
                            inputs = hf_processor(
                                text=[text],
                                images=[self._tensor_to_pil(image)],
                                return_tensors="pt",
                                padding=True,
                            )
                        else:
                            inputs = hf_processor(
                                text=[text],
                                return_tensors="pt",
                                padding=True,
                            )
                        print("[HFInference] Using two-step processing (tokenize=False)")
                        
                except Exception as e:
                    print(f"[HFInference] apply_chat_template failed: {e}, using fallback")
                    # Fallback to simple text processing
                    if image is not None:
                        inputs = hf_processor(
                            text=final_prompt,
                            images=[self._tensor_to_pil(image)],
                            return_tensors="pt",
                        )
                    else:
                        inputs = hf_processor(
                            text=final_prompt,
                            return_tensors="pt",
                        )
            else:
                # Fallback for models without chat template (custom models)
                print("[HFInference] Using direct text processing (no chat template)")
                if image is not None:
                    # Try with image parameter
                    try:
                        inputs = hf_processor(
                            text=final_prompt,
                            images=[self._tensor_to_pil(image)],
                            return_tensors="pt",
                        )
                    except Exception:
                        # Some models might use different parameter names
                        inputs = hf_processor(
                            final_prompt,
                            [self._tensor_to_pil(image)],
                            return_tensors="pt",
                        )
                else:
                    inputs = hf_processor(
                        text=final_prompt,
                        return_tensors="pt",
                    )
            
            # Move to device
            device = hf_model.device if hasattr(hf_model, 'device') else 'cuda'
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generation config - following model card recommendations
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            
            # Set pad and eos token ids
            if hasattr(hf_processor, 'tokenizer'):
                gen_kwargs["pad_token_id"] = hf_processor.tokenizer.pad_token_id
                gen_kwargs["eos_token_id"] = hf_processor.tokenizer.eos_token_id
            
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["min_p"] = min_p
                if top_k > 0:
                    gen_kwargs["top_k"] = top_k
                if repetition_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = repetition_penalty
            
            # Generate
            print("[HFInference] Generating...")
            with torch.inference_mode():
                outputs = hf_model.generate(**inputs, **gen_kwargs)
            
            # Decode - skip input tokens
            input_len = inputs.get("input_ids", inputs.get("input_token_ids", [[]])).shape[1]
            generated_ids = outputs[:, input_len:]
            
            # Decode
            if hasattr(hf_processor, 'batch_decode'):
                result = hf_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            else:
                result = hf_processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
            
            # Strip thinking/reasoning tags if requested
            if strip_thinking:
                result = self._strip_thinking_tags(result)
            
            print(f"[HFInference] Generated {len(result)} characters")
            return (result.strip(),)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"Error during inference: {str(e)}",)
        
        finally:
            # Clean up temp image file if created
            if temp_image_path:
                try:
                    import os
                    os.unlink(temp_image_path)
                    print(f"[HFInference] Cleaned up temp image: {temp_image_path}")
                except Exception:
                    pass
            
            # Auto-unload model to free VRAM (one-shot approach)
            if auto_unload:
                _unload_cached_model()


class HFTextInference:
    """Run text-only inference with HuggingFace models.
    
    Simplified node for text-only generation without image input.
    Uses the same models but skips vision processing for faster text generation.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HF_MODEL", {}),
                "processor": ("HF_PROCESSOR", {}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "text_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Text-only inference with HuggingFace models. Faster than VL node when no image is needed."

    def generate(
        self,
        model,
        processor,
        prompt,
        system_prompt="You are a helpful assistant.",
        max_new_tokens=512,
        temperature=0.7,
        text_input=None,
    ):
        # Reuse HFVLInference logic without image
        vl = HFVLInference()
        return vl.generate(
            model=model,
            processor=processor,
            prompt=prompt,
            image=None,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            text_input=text_input,
        )
