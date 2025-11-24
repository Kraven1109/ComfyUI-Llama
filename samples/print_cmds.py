import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from llamacpp_inference import ComfyLLama


def main():
    node = ComfyLLama()
    # Use user-provided models (adjust if needed)
    base_dir = Path(__file__).parents[1]
    text_model = Path(r"D:\Apps\ComfyUI_portable\ComfyUI\models\prompt_generator\ggml-model-q8_0__Huihui-Qwen3-VL-8B-Thinking-abliterated.gguf")
    mmproj_model = Path(r"D:\Apps\ComfyUI_portable\ComfyUI\models\prompt_generator\mmproj-model-f16__Huihui-Qwen3-VL-8B-Thinking-abliterated.gguf")
    if not text_model.exists():
        print('Warning: text model not found at', text_model)
    if not mmproj_model.exists():
        print('Warning: mmproj model not found at', mmproj_model)
    # create a small sample image if not present
    sample_image = base_dir / 'sample.png'
    try:
        from PIL import Image
        if not sample_image.exists():
            Image.new('RGB', (64, 64), (128, 128, 128)).save(sample_image)
    except Exception:
        if not sample_image.exists():
            with open(sample_image, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n')

    # Text-only scenario
    cmd_text, cli_text_name, path_text, images_text, aud_text = node.build_command(
        text="Hello world",
        gguf_model=str(text_model),
        temperature=0.7,
        n_predict=8,
        seed=-1,
        n_gpu_layers=-1,
        ctx_size=2048,
        interactive=False,
        repeat_penalty=1.1,
        media=None,
        media_paths=None,
        mmproj_model=None,
        system_prompt="",
        stop_string="",
        llama_cpp_folder=r"D:\Apps\llama-cuda",
        use_jinja=False,
        chat_template="",
        strip_prompt=True,
    )
    print('--- Text-only scenario ---')
    print('CLI Name:', cli_text_name)
    print('Executable:', path_text)
    print('Command:', ' '.join(cmd_text))
    print('Images:', images_text, 'Audios:', aud_text)

    # Multimodal scenario: say we have an image path and mmproj
    cmd_multi, cli_multi_name, path_multi, images_multi, aud_multi = node.build_command(
        text="Describe the image",
        gguf_model=str(text_model),
        temperature=0.5,
        n_predict=8,
        seed=42,
        n_gpu_layers=0,
        ctx_size=2048,
        interactive=False,
        repeat_penalty=1.1,
        media=str(sample_image),
        media_paths=None,
        mmproj_model=str(mmproj_model),
        system_prompt="",
        stop_string="",
        llama_cpp_folder=r"D:\Apps\llama-cuda",
        use_jinja=False,
        chat_template="",
        strip_prompt=True,
    )
    print('\n--- Multimodal scenario ---')
    print('CLI Name:', cli_multi_name)
    print('Executable:', path_multi)
    print('Command:', ' '.join(cmd_multi))
    print('Images:', images_multi, 'Audios:', aud_multi)

    # Multimodal (multiple media) scenario
    cmd_multi2, cli_multi2_name, path_multi2, images_multi2, aud_multi2 = node.build_command(
        text="Describe the images",
        gguf_model=str(text_model),
        temperature=0.5,
        n_predict=8,
        seed=42,
        n_gpu_layers=0,
        ctx_size=2048,
        interactive=False,
        repeat_penalty=1.1,
        media=None,
        media_paths=[str(sample_image), str(sample_image)],
        mmproj_model=str(mmproj_model),
        system_prompt="",
        stop_string="",
        llama_cpp_folder=r"D:\Apps\llama-cuda",
        use_jinja=False,
        chat_template="",
        strip_prompt=True,
    )
    print('\n--- Multimodal multiple media scenario ---')
    print('CLI Name:', cli_multi2_name)
    print('Executable:', path_multi2)
    print('Command:', ' '.join(cmd_multi2))
    print('Images:', images_multi2, 'Audios:', aud_multi2)

    # Optionally run the commands (small tokens & short timeout to avoid long inference)
    execute = True
    cli_timeout = 20
    if execute:
        import subprocess
        print('\nRunning text-only command...')
        try:
            r = subprocess.run(cmd_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=cli_timeout)
            print('Return code:', r.returncode)
            print('Stdout (truncated 1k):')
            print((r.stdout or '')[:1000])
            print('Stderr (truncated 1k):')
            print((r.stderr or '')[:1000])
        except subprocess.TimeoutExpired:
            print(f"text-only command timed out after {cli_timeout}s")
        except Exception as e:
            print('Error running text-only command:', e)

        print('\nRunning multimodal command...')
        try:
            r2 = subprocess.run(cmd_multi, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=cli_timeout)
            print('Return code:', r2.returncode)
            print('Stdout (truncated 1k):')
            print((r2.stdout or '')[:1000])
            print('Stderr (truncated 1k):')
            print((r2.stderr or '')[:1000])
        except subprocess.TimeoutExpired:
            print(f"multimodal command timed out after {cli_timeout}s")
        except Exception as e:
            print('Error running multimodal command:', e)


if __name__ == '__main__':
    main()
