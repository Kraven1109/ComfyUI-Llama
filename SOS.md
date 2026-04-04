got prompt
ComfyLLama: Using chat template file: /DATA2/llm/models/chat_template_qwen35.jinja
ComfyLLama: Starting One-Shot Server on port 8080...
ComfyLLama: Command: /DATA1/quang_dev/llama.cpp_kraven/build-linux-cuda/bin/llama-server -m /DATA1/Apps/ComfyUI_portable/ComfyUI/models/LLM/models/qwen3.5-VL-9b-opus46d-abl.Q6_K.gguf --port 8080 --ctx-size 90000 --n-gpu-layers -1 --threads -1 --flash-attn auto --temp 0.6 --top-k 40 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.3 --repeat-last-n 256 --frequency-penalty 0.1 --presence-penalty 0.1 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n -1 --jinja --chat-template-file /DATA2/llm/models/chat_template_qwen35.jinja --mmproj /DATA1/Apps/ComfyUI_portable/ComfyUI/models/LLM/models/qwen3.5-VL-9b-mmproj.BF16.gguf
ComfyLLama: Waiting up to 300 seconds for model to load...
ComfyLLama: Model loaded successfully!
ComfyLLama: Sending prompt...
FETCH ComfyRegistry Data: 100/136
FETCH ComfyRegistry Data: 105/136
ComfyLLama: Killing server to release VRAM...
ComfyLLama: Server shutdown complete.
Prompt executed in 22.44 seconds

prompt: you are a well known film director
1. analyze the image in ultra details (image path /DATA1/Apps/ComfyUI_portable/ComfyUI/input/videoframe_1348.png)
2. add a to the scence, he is approaching from her left side, and pat her head
3. adapt all the info to a positive prompt to make video with motion/ action 
4. give me only the final prompt for the scene

Here is my analysis and the resulting prompt for your video scene.

### 1. Ultra-Detailed Analysis
*   **Subject:** A young woman with a heart-shaped face, fair skin, and expressive brown eyes. She has shoulder-length, wavy hair with a light brown or dark blonde tone.
*   **Attire:** She is wearing a crisp white button-down shirt. The shirt is unbuttoned significantly, creating a deep V-neckline. The fabric appears to be a high-quality cotton or linen blend.
*   **Setting:** A bright, airy indoor space. The background is soft and out of focus (bokeh), revealing hints of a grey sofa and a tripod floor lamp with a white shade.
*   **Lighting:** The lighting is soft and diffused, likely natural window light hitting her face from the front-left, highlighting her cheekbones and eyes.
*   **Mood:** Intimate, relaxed, and inviting.

### 2. Director's Vision & Action
To bring this to life, I want to capture a moment of gentle affection. The camera will stay relatively static to focus on the subtle movement of her hair and the hand interaction.

*   **The Action:** A hand (presumably a man's) will enter the frame from the **left side**. It will glide gently through her hair, patting her head and neck, causing her hair to sway slightly. She will maintain a soft, inviting smile.

### 3. The Final Prompt
Here is the optimized prompt ready for your video generation tool:

**Cinematic medium shot of a beautiful young woman with wavy light brown hair wearing an unbuttoned white shirt. A hand enters from the left side of the frame and gently brushes her hair back and pats her head while she smiles warmly at the camera. Soft natural lighting, shallow depth of field with a blurred living room background, high resolution, realistic texture.**


The user wants me to act as a film director and create a prompt for a video based on the provided image.

**1. Analyze the Image (Ultra Details):**
*   **Subject:** A young woman with shoulder-length, wavy light brown/dark blonde hair.
*   **Face:** Fair skin, subtle makeup, soft brown eyes, smiling gently, looking slightly off-camera (or just at the lens).
*   **Clothing:** A white button-down shirt. It's unbuttoned low, revealing cleavage. The fabric looks crisp but soft. Sleeves are rolled up slightly.
*   **Pose:** Seated or standing still, facing forward.
*   **Background:** Indoors, blurred (shallow depth of field). Looks like a living room. There's a grey sofa and a floor lamp with a white shade visible in the background. The lighting is soft, natural, coming from the front/left.
*   **Mood:** Intimate, calm, inviting, warm.

**2. Add an Action:**
*   **Action:** A man approaches from her left side. He is out of frame mostly but his hand is visible. He gently pats/brushes her hair back with his hand.
*   **Interaction:** This adds a dynamic element. It suggests affection or greeting.

**3. Adapt to a Positive Prompt (for Video Generation):**
*   **Keywords:** 4k, high resolution, realistic, cinematic lighting, soft focus background.
*   **Subject Descriptions:** Beautiful Asian woman, wavy brown hair, white button-up shirt, open collar.
*   **Action Descriptions:** A hand enters from the left frame, gently caressing/patting her hair. She smiles warmly. Slight head turn.
*   **Lighting/Atmosphere:** Natural daylight, soft shadows, warm tones, bokeh background.

**4. Draft the Final Prompt:**
*   *Draft 1:* A video of a beautiful woman in a white shirt. A hand comes from the left and touches her hair. She smiles.
*   *Refining for "Ultra Detail":* Need to specify the lighting, the texture of the hair, the movement.
*   *Final Polish:* "Cinematic shot of a beautiful young woman with wavy light brown hair wearing an unbuttoned white shirt. A male hand enters from the left side of the frame and gently brushes her hair back while she smiles softly at the camera. Shallow depth of field with a blurred living room background."

Let's make it even more descriptive for a video generator like Sora or Runway.

*   "Medium shot, low angle slightly. A stunning woman with shoulder-length wavy hair and a white button-down shirt unbuttoned at the top. A hand enters from the left foreground, gently patting her hair and neck. She looks happy and relaxed. Soft natural lighting, bright, airy aesthetic."

Let's combine the analysis and the final prompt into the requested format.

**Plan:**
1.  Write the analysis section.
2.  Write the step-by-step thought process (simulating the "director" persona).
3.  Provide the final prompt.