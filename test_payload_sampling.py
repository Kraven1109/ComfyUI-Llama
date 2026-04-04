"""
Test: Can we send sampling params in the API payload alongside --reasoning-budget -1?

Server CMD has NO sampling params (just hardware/model config).
Three payloads tested:
  A) minimal (baseline - known working)
  B) + basic sampling (temp, top_k, top_p, min_p)
  C) + repetition penalties (repeat_penalty, freq, pres, dry)

Run: python test_payload_sampling.py [A|B|C]
Default: C (full payload with all sampling params)
"""
import sys, io, base64, time, subprocess, os, signal, requests

VARIANT = sys.argv[1].upper() if len(sys.argv) > 1 else "C"

SERVER_EXE = "/DATA1/quang_dev/llama.cpp_kraven/build-linux-cuda/bin/llama-server"
MODEL      = "/DATA1/Apps/ComfyUI_portable/ComfyUI/models/LLM/models/qwen3.5-VL-9b-opus46d-abl.Q6_K.gguf"
MMPROJ     = "/DATA1/Apps/ComfyUI_portable/ComfyUI/models/LLM/models/qwen3.5-VL-9b-mmproj.BF16.gguf"
TMPL       = "/DATA2/llm/models/chat_template_qwen35.jinja"
PORT       = 8102

# Server CMD: hardware/model only — no sampling params at all
cmd = [
    SERVER_EXE, "-m", MODEL,
    "--port", str(PORT), "--ctx-size", "131072",
    "--n-gpu-layers", "-1", "--flash-attn", "on",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--batch-size", "2048", "--ubatch-size", "512",
    "--jinja", "--mmproj", MMPROJ,
    "--reasoning-budget", "-1",
    "--chat-template-file", TMPL,
]
print(f"=== Variant {VARIANT} ===")
print("CMD:", " ".join(cmd))

t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True)
print("Server starting...", flush=True)

server_ready = False
for _ in range(300):
    if proc.poll() is not None:
        print("Server crashed!"); sys.exit(1)
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=1)
        if r.status_code == 200:
            print(f"Server ready in {time.time()-t0:.1f}s"); server_ready = True; break
    except Exception:
        pass
    time.sleep(1)

if not server_ready:
    print("Timeout"); proc.kill(); sys.exit(1)

from PIL import Image
img = Image.open("/DATA1/Apps/ComfyUI_portable/ComfyUI/input/videoframe_1348.png").convert("RGB")
buf = io.BytesIO(); img.save(buf, "JPEG", quality=90); buf.seek(0)
b64 = "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()

PROMPT = ("you are a well known film director\n"
          "1. analyze the image in ultra details\n"
          "2. add a guy approaching from her left side, and pat her head\n"
          "3. adapt all the info to a positive prompt to make video with motion/action\n"
          "4. give me only the final prompt for the scene")

# Base payload (always present)
payload = {
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": b64}},
        {"type": "text", "text": PROMPT},
    ]}],
    "n_predict": -1,
    "truncate_prompt_tokens": int(131072 * 0.75),
    "stop": [],
}

if VARIANT in ("B", "C"):
    # Add basic sampling params
    payload.update({
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
    })
    print("+ basic sampling (temp, top_k, top_p, min_p)")

if VARIANT == "C":
    # Add repetition penalties
    payload.update({
        "repeat_penalty": 1.1,
        "repeat_last_n": 256,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": -1,
    })
    print("+ repetition penalties (repeat_penalty, freq, pres, dry)")

print("Sending prompt...", flush=True)
t1 = time.time()
resp = requests.post(f"http://localhost:{PORT}/chat/completions", json=payload, timeout=600)
elapsed = time.time() - t1

print(f"Status: {resp.status_code}  Time: {elapsed:.1f}s")
if resp.status_code == 200:
    msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or ""
    print(f"content_len={len(content)}  reasoning_len={len(reasoning)}")
    if content:
        print("\n=== CONTENT (first 500 chars) ===")
        print(content[:500])
    else:
        print("\n[CONTENT EMPTY]")
        if reasoning:
            print(f"[Reasoning: {len(reasoning)} chars]")
            print(reasoning[:200])
else:
    print(resp.text[:500])

os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=10)
print("Server stopped.")
