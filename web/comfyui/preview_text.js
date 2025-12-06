import { app } from "../../../../scripts/app.js";

console.log("ComfyLLama Preview Text extension loaded");

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

app.registerExtension({
    name: "Comfy.ComfyLLama.PreviewText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ComfyLLamaPreviewText") {
            console.log("Registering ComfyLLamaPreviewText extension...");

            nodeType.prototype.onExecuted = function(message) {
                if (message.text && message.text.length > 0) {
                    const text = message.text[0];
                    const title = message.title ? message.title[0] : "Output";

                    // Create or update text preview widget
                    if (!this.textWidget) {
                        const container = document.createElement("div");
                        container.className = "llama_text_preview";
                        container.style.cssText = `
                            width: 100%;
                            max-height: 300px;
                            overflow-y: auto;
                            padding: 8px;
                            background: rgba(0, 0, 0, 0.3);
                            border-radius: 4px;
                            font-family: monospace;
                            font-size: 12px;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            color: #e0e0e0;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                        `;

                        this.textWidget = this.addDOMWidget("text_preview", "preview", container, {
                            serialize: false,
                            hideOnZoom: false,
                        });
                        this.textWidget.container = container;
                    }

                    // Update content
                    this.textWidget.container.innerHTML = "";
                    
                    // Add title
                    const titleEl = document.createElement("div");
                    titleEl.style.cssText = `
                        font-weight: bold;
                        margin-bottom: 6px;
                        color: #90caf9;
                        font-size: 11px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    `;
                    titleEl.textContent = title;
                    this.textWidget.container.appendChild(titleEl);

                    // Add text content
                    const textEl = document.createElement("div");
                    textEl.textContent = text;
                    this.textWidget.container.appendChild(textEl);

                    // Add copy button
                    const copyBtn = document.createElement("button");
                    copyBtn.textContent = "📋 Copy";
                    copyBtn.style.cssText = `
                        margin-top: 8px;
                        padding: 4px 8px;
                        font-size: 11px;
                        cursor: pointer;
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 3px;
                        color: #e0e0e0;
                    `;
                    copyBtn.onclick = () => {
                        navigator.clipboard.writeText(text).then(() => {
                            copyBtn.textContent = "✓ Copied!";
                            setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1500);
                        });
                    };
                    this.textWidget.container.appendChild(copyBtn);

                    fitHeight(this);
                }
            };
        }
    }
});
