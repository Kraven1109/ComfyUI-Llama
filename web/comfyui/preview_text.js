import { app } from "../../../../scripts/app.js";

console.log("ComfyLLama Preview Text extension loaded");

// Lightweight markdown renderer (no external dependencies)
function renderMarkdown(text) {
    if (!text) return "";

    // Escape HTML to prevent XSS, then selectively re-allow our own tags
    function escHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    let html = escHtml(text);

    // Fenced code blocks (``` ... ```) — must come first
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        const langLabel = lang ? `<span class="llm-code-lang">${escHtml(lang)}</span>` : "";
        return `<pre class="llm-code-block">${langLabel}<code>${code}</code></pre>`;
    });

    // Inline code (`code`)
    html = html.replace(/`([^`\n]+)`/g, '<code class="llm-inline-code">$1</code>');

    // Headers (process line by line via split + rejoin)
    html = html.replace(/^######\s+(.+)$/gm, '<h6 class="llm-h">$1</h6>');
    html = html.replace(/^#####\s+(.+)$/gm, '<h5 class="llm-h">$1</h5>');
    html = html.replace(/^####\s+(.+)$/gm, '<h4 class="llm-h">$1</h4>');
    html = html.replace(/^###\s+(.+)$/gm, '<h3 class="llm-h">$1</h3>');
    html = html.replace(/^##\s+(.+)$/gm, '<h2 class="llm-h">$1</h2>');
    html = html.replace(/^#\s+(.+)$/gm, '<h1 class="llm-h">$1</h1>');

    // Horizontal rules
    html = html.replace(/^---+$/gm, '<hr class="llm-hr"/>');

    // Bold + italic (***text***)
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    // Bold (**text**)
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic (*text*)
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Bold (__text__)
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic (_text_)
    html = html.replace(/_([^_\n]+)_/g, '<em>$1</em>');

    // Unordered list items (- item or * item)
    html = html.replace(/^[\*\-]\s+(.+)$/gm, '<li class="llm-li">$1</li>');
    // Wrap consecutive <li> items in <ul>
    html = html.replace(/(<li class="llm-li">.*<\/li>\n?)+/g, m => `<ul class="llm-ul">${m}</ul>`);

    // Ordered list items (1. item)
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li class="llm-li">$1</li>');
    // Wrap consecutive (numbered) <li> items in <ol>
    html = html.replace(/(<li class="llm-li">.*<\/li>\n?)+/g, m => {
        if (m.includes('<ul')) return m; // already wrapped
        return `<ol class="llm-ol">${m}</ol>`;
    });

    // Blockquotes
    html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote class="llm-bq">$1</blockquote>');

    // Double newlines → paragraph breaks
    html = html.replace(/\n\n+/g, '</p><p class="llm-p">');

    // Single newlines (not inside block elements) → <br>
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not starting with a block element
    if (!html.startsWith('<h') && !html.startsWith('<pre') && !html.startsWith('<ul') &&
        !html.startsWith('<ol') && !html.startsWith('<blockquote') && !html.startsWith('<hr')) {
        html = `<p class="llm-p">${html}</p>`;
    }

    return html;
}

const PREVIEW_STYLES = `
.llm-preview-content {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: #e0e0e0;
}
.llm-preview-content .llm-p {
    margin: 0.4em 0;
}
.llm-preview-content .llm-h {
    color: #90caf9;
    margin: 0.6em 0 0.3em;
    font-weight: 600;
    line-height: 1.3;
}
.llm-preview-content h1.llm-h { font-size: 1.3em; }
.llm-preview-content h2.llm-h { font-size: 1.2em; }
.llm-preview-content h3.llm-h { font-size: 1.1em; }
.llm-preview-content h4.llm-h,
.llm-preview-content h5.llm-h,
.llm-preview-content h6.llm-h { font-size: 1em; }
.llm-preview-content .llm-hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.2);
    margin: 0.5em 0;
}
.llm-preview-content .llm-ul,
.llm-preview-content .llm-ol {
    padding-left: 1.4em;
    margin: 0.3em 0;
}
.llm-preview-content .llm-li {
    margin: 0.15em 0;
}
.llm-preview-content .llm-bq {
    border-left: 3px solid #90caf9;
    margin: 0.4em 0;
    padding: 0.2em 0.6em;
    color: #b0bec5;
    font-style: italic;
}
.llm-preview-content .llm-code-block {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 4px;
    padding: 8px 10px;
    margin: 0.4em 0;
    overflow-x: auto;
    white-space: pre;
    font-family: "SFMono-Regular", Consolas, "Courier New", monospace;
    font-size: 11.5px;
    line-height: 1.5;
    position: relative;
}
.llm-preview-content .llm-code-lang {
    position: absolute;
    top: 4px;
    right: 8px;
    font-size: 10px;
    color: #78909c;
    font-family: sans-serif;
}
.llm-preview-content .llm-inline-code {
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    padding: 1px 4px;
    font-family: "SFMono-Regular", Consolas, "Courier New", monospace;
    font-size: 11.5px;
    color: #f48fb1;
}
.llm-preview-content strong { color: #fff; }
.llm-preview-content em { color: #cfd8dc; }
`;

// Inject styles once
let _stylesInjected = false;
function injectStyles() {
    if (_stylesInjected) return;
    const style = document.createElement("style");
    style.textContent = PREVIEW_STYLES;
    document.head.appendChild(style);
    _stylesInjected = true;
}

app.registerExtension({
    name: "Comfy.ComfyLLama.PreviewText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ComfyLLamaPreviewText") {
            console.log("Registering ComfyLLamaPreviewText extension...");
            injectStyles();

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
                
                // Header row: title + toggle
                const headerRow = document.createElement("div");
                headerRow.style.cssText = `
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 6px;
                    flex-shrink: 0;
                `;

                const titleEl = document.createElement("div");
                titleEl.style.cssText = `
                    font-weight: bold;
                    color: #90caf9;
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                `;
                titleEl.textContent = title;
                headerRow.appendChild(titleEl);

                // Markdown toggle button
                let isMarkdown = true;
                const mdToggle = document.createElement("button");
                mdToggle.textContent = "MD";
                mdToggle.title = "Toggle: Markdown / Raw";
                mdToggle.style.cssText = `
                    padding: 1px 6px;
                    font-size: 10px;
                    cursor: pointer;
                    background: rgba(144, 202, 249, 0.15);
                    border: 1px solid rgba(144, 202, 249, 0.4);
                    border-radius: 3px;
                    color: #90caf9;
                    font-family: monospace;
                `;
                headerRow.appendChild(mdToggle);
                node.textWidget.container.appendChild(headerRow);

                // Scrollable content wrapper
                const textWrapper = document.createElement("div");
                textWrapper.style.cssText = `
                    flex: 1;
                    overflow-y: auto;
                    overflow-x: hidden;
                    min-height: 0;
                `;
                
                // Markdown rendered content div
                const mdEl = document.createElement("div");
                mdEl.className = "llm-preview-content";
                mdEl.innerHTML = renderMarkdown(text);
                textWrapper.appendChild(mdEl);

                // Raw text div (hidden by default)
                const rawEl = document.createElement("div");
                rawEl.style.cssText = `
                    display: none;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-family: "SFMono-Regular", Consolas, "Courier New", monospace;
                    font-size: 12px;
                    color: #e0e0e0;
                    line-height: 1.5;
                `;
                rawEl.textContent = text;
                textWrapper.appendChild(rawEl);

                node.textWidget.container.appendChild(textWrapper);

                // Toggle handler
                mdToggle.onclick = () => {
                    isMarkdown = !isMarkdown;
                    if (isMarkdown) {
                        mdEl.style.display = "";
                        rawEl.style.display = "none";
                        mdToggle.textContent = "MD";
                        mdToggle.style.background = "rgba(144, 202, 249, 0.15)";
                        mdToggle.style.color = "#90caf9";
                    } else {
                        mdEl.style.display = "none";
                        rawEl.style.display = "";
                        mdToggle.textContent = "RAW";
                        mdToggle.style.background = "rgba(255,255,255,0.08)";
                        mdToggle.style.color = "#b0bec5";
                    }
                };

                // Footer: char count + copy button
                const footer = document.createElement("div");
                footer.style.cssText = `
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-top: 6px;
                    flex-shrink: 0;
                `;

                const charCount = document.createElement("span");
                charCount.style.cssText = `
                    font-size: 10px;
                    color: #78909c;
                    font-family: monospace;
                `;
                charCount.textContent = `${text.length} chars`;
                footer.appendChild(charCount);

                const copyBtn = document.createElement("button");
                copyBtn.textContent = "📋 Copy";
                copyBtn.style.cssText = `
                    padding: 3px 8px;
                    font-size: 11px;
                    cursor: pointer;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 3px;
                    color: #e0e0e0;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                `;
                copyBtn.onclick = () => {
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.textContent = "✓ Copied!";
                        setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1500);
                    });
                };
                footer.appendChild(copyBtn);
                node.textWidget.container.appendChild(footer);

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
