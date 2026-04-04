import os
import folder_paths

CHAT_TEMPLATE_OPTIONS = [
    "auto", "chatml", "llama2", "llama3", "gemma", "phi3", "phi4",
    "mistral-v1", "mistral-v3", "deepseek", "deepseek2", "deepseek3",
    "command-r", "vicuna", "zephyr", "openchat", "falcon3", "exaone3",
]

class GGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        models_dir = folder_paths.models_dir
        gguf_files = []
        
        # Walk through directories, following symlinks to include them in dropdown
        for root, dirs, files in os.walk(models_dir, followlinks=True):
            for file in files:
                if file.endswith('.gguf'):
                    rel_path = os.path.relpath(os.path.join(root, file), models_dir)
                    gguf_files.append(rel_path)
        
        mmproj_options = ["None"] + gguf_files

        if not gguf_files:
            gguf_files = ["(no .gguf files found — check models_dir)"]

        return {
            "required": {
                "gguf_name": (gguf_files, {"default": gguf_files[0]}),
            },
            "optional": {
                "mmproj_name": (mmproj_options, {
                    "default": "None",
                    "tooltip": "Vision projector (mmproj) for multimodal inference. Select 'None' for text-only models.",
                }),
                "gguf_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to GGUF model. Overrides gguf_name when set.",
                }),
                "mmproj_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to mmproj model. Overrides mmproj_name when set.",
                }),
                # === Chat Template (model-specific, belongs with the model) ===
                "use_jinja": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Jinja2 chat template (--jinja). Required for most modern models.",
                }),
                "chat_template": (CHAT_TEMPLATE_OPTIONS, {
                    "default": "auto",
                    "tooltip": "Chat template format. 'auto' = detect from model metadata.",
                }),
                "custom_chat_template": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Custom Jinja2 chat template. Two formats:\n"
                        "• Absolute path to a .jinja file\n"
                        "• Inline Jinja2 template string\n"
                        "Overrides chat_template when set."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_gguf"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = (
        "Load a GGUF model. Optionally select an mmproj for multimodal (vision) inference — "
        "no second loader node needed.\n\n"
        "Chat template is model-specific and can be set here: use custom_chat_template to "
        "point to a .jinja file or provide an inline template."
    )

    def load_gguf(
        self,
        gguf_name,
        mmproj_name="None",
        gguf_path_override="",
        mmproj_path_override="",
        use_jinja=True,
        chat_template="auto",
        custom_chat_template="",
    ):
        # Resolve main model path
        if gguf_path_override and os.path.exists(gguf_path_override):
            gguf_path = gguf_path_override
        else:
            gguf_path = os.path.join(folder_paths.models_dir, gguf_name)

        if not gguf_path or not os.path.exists(gguf_path):
            raise ValueError(f"GGUF file not found: {gguf_path}")

        # Resolve mmproj path (optional)
        mmproj_path = None
        if mmproj_path_override and os.path.exists(mmproj_path_override):
            mmproj_path = mmproj_path_override
        elif mmproj_name and mmproj_name != "None":
            candidate = os.path.join(folder_paths.models_dir, mmproj_name)
            if os.path.exists(candidate):
                mmproj_path = candidate
            else:
                raise ValueError(f"mmproj file not found: {candidate}")

        result = {"path": gguf_path}
        if mmproj_path:
            result["mmproj_path"] = mmproj_path
        # Chat template info travels with the model dict
        result["use_jinja"] = use_jinja
        result["chat_template"] = chat_template
        result["custom_chat_template"] = custom_chat_template.strip()

        return (result,)