import { app } from "../../../../scripts/app.js";

console.log("ComfyLLama Preview Text extension loaded");

app.registerExtension({
    name: "Comfy.ComfyLLama.PreviewText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ComfyLLamaPreviewText") {
            console.log("Registering ComfyLLamaPreviewText extension...");

            // Helper to create/update the preview UI
            function createPreviewUI(node, text, title) {
                if (!node.textWidget) {
                    const container = document.createElement("div");
                    container.className = "llama_text_preview";
                    container.style.cssText = `
                        width: calc(100% - 20px);
                        max-height: 100%;
                        overflow: hidden;
                        display: flex;
                        flex-direction: column;
                        padding: 8px;
                        margin: 0 10px;
                        background: rgba(0, 0, 0, 0.3);
                        border-radius: 4px;
                        font-family: monospace;
                        font-size: 12px;
                        color: #e0e0e0;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-sizing: border-box;
                    `;

                    node.textWidget = node.addDOMWidget("text_preview", "preview", container, {
                        serialize: false,
                        hideOnZoom: false,
                    });
                    node.textWidget.container = container;
                }

                // Store text/title for persistence
                node._previewText = text;
                node._previewTitle = title;

                // Update content
                node.textWidget.container.innerHTML = "";
                
                // Add title
                const titleEl = document.createElement("div");
                titleEl.style.cssText = `
                    font-weight: bold;
                    margin-bottom: 6px;
                    color: #90caf9;
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    flex-shrink: 0;
                `;
                titleEl.textContent = title;
                node.textWidget.container.appendChild(titleEl);

                // Add scrollable text content wrapper
                const textWrapper = document.createElement("div");
                textWrapper.style.cssText = `
                    flex: 1;
                    overflow-y: auto;
                    overflow-x: hidden;
                    min-height: 0;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                `;
                
                // Add text content
                const textEl = document.createElement("div");
                textEl.textContent = text;
                textWrapper.appendChild(textEl);
                node.textWidget.container.appendChild(textWrapper);

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
                    flex-shrink: 0;
                `;
                copyBtn.onclick = () => {
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.textContent = "✓ Copied!";
                        setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1500);
                    });
                };
                node.textWidget.container.appendChild(copyBtn);

                // Request a redraw
                node.setDirtyCanvas(true, true);
            }

            // Handle execution results
            nodeType.prototype.onExecuted = function(message) {
                if (message.text && message.text.length > 0) {
                    const text = message.text[0];
                    const title = message.title ? message.title[0] : "Output";
                    createPreviewUI(this, text, title);
                }
            };

            // Save state for persistence
            const origOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (origOnSerialize) {
                    origOnSerialize.call(this, o);
                }
                // Save preview text and title
                if (this._previewText !== undefined) {
                    o._previewText = this._previewText;
                    o._previewTitle = this._previewTitle || "Output";
                }
            };

            // Restore state on load
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (origOnConfigure) {
                    origOnConfigure.call(this, o);
                }
                // Restore preview text if saved
                if (o._previewText) {
                    // Delay to ensure node is fully initialized
                    setTimeout(() => {
                        createPreviewUI(this, o._previewText, o._previewTitle || "Output");
                    }, 100);
                }
            };
        }
    }
});
