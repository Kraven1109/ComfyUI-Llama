class MediaPathsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine"
    CATEGORY = "🦙 ComfyUI-LLama"
    DESCRIPTION = "Combine multiple file paths for multimodal inputs. Connect path inputs dynamically. Outputs a list of paths for image/audio processing."

    def combine(self, **kwargs):
        paths = []
        for key in sorted(kwargs.keys()):
            if key.startswith("path_"):
                value = kwargs[key]
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
        return (paths,)