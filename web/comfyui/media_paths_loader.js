import { app } from "../../../../scripts/app.js";

console.log("MediaPathsLoader extension file loaded");

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
        };
    } else {
        object[property] = callback;
    }
}

app.registerExtension({
    name: "Comfy.ComfyLLama.MediaPathsLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("beforeRegisterNodeDef called for:", nodeData.name); // Removed noisy log
        if (nodeData.name === "MediaPathsLoader") {
            console.log("Registering MediaPathsLoader extension...");

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                console.log("MediaPathsLoader node created, adding widgets...");
            });

            // Capture widgets_values during configure so we can create matching widgets
            // when the node is actually created. This avoids ordering issues when ComfyUI
            // restores widget values from a saved workflow.
            chainCallback(nodeType.prototype, "configure", function (info) {
                try {
                    if (info && info.widgets_values) {
                        // store for onNodeCreated to consume
                        this._mediaPathsLoader_pending_widgets = info.widgets_values;
                    }
                } catch (e) {
                    console.error("MediaPathsLoader: error handling configure:", e);
                }
            });

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                // Define the add callback
                const addCallback = () => {
                    const pathWidgets = this.widgets.filter(w => w.name && w.name.startsWith("path_"));
                    const nextIndex = pathWidgets.length + 1;
                    if (nextIndex <= 8) {
                        addPathWidget.call(this, nextIndex, "");
                    }
                };

                // Add '+' button at the top
                const addButton = this.addWidget("button", "+", "", addCallback);
                addButton.serialize = false;

                // helper: create a single inline custom widget containing text + Browse + Remove
                function addPathWidget(index, initialValue) {
                    const name = `path_${index}`;
                    const widget = this.addWidget("custom", name, initialValue || "", null);
                    widget.serialize = true;
                    // ensure widget knows its node (some environments may not set this immediately)
                    widget.node = this;
                    widget.name = name;
                    widget.value = initialValue || "";

                    // sizes for parts
                    const margin = 12;
                    const removeW = 20;

                    widget.computeSize = function (width) {
                        return [width, LiteGraph.NODE_WIDGET_HEIGHT];
                    };

                    widget.draw = function (ctx, node, width, y, height) {
                        this.last_y = y;
                        const totalW = width - margin * 2;
                        // merged area: text + implicit browse behaviour
                        const textW = totalW - (removeW + 4);
                        const textX = margin;
                        const textY = y + 2;
                        const textH = height - 4;

                        // text area background — use roundRect when available, fallback to rect
                        ctx.save();
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        if (typeof ctx.roundRect === "function") {
                            ctx.beginPath();
                            ctx.roundRect(textX, textY, textW, textH, [textH * 0.5]);
                            ctx.fill();
                        } else {
                            ctx.fillRect(textX, textY, textW, textH);
                        }

                        // text: show placeholder (path_index) when empty
                        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                        ctx.textBaseline = "middle";
                        ctx.textAlign = "left";
                        // Resolve the display text from the widget value robustly.
                        let textVal = name;
                        if (typeof this.value === "string") {
                            textVal = this.value.length ? this.value : name;
                        } else if (this.value && typeof this.value === "object") {
                            textVal = this.value.name || this.value.file || this.value.path || (Array.isArray(this.value) ? this.value.join(", ") : "");
                            if (!textVal || textVal === "[object Object]") {
                                // Try to find any string property on the object
                                for (const k of Object.keys(this.value)) {
                                    const v = this.value[k];
                                    if (typeof v === "string" && v.length) {
                                        textVal = v;
                                        break;
                                    }
                                }
                            }
                            if (!textVal) textVal = name;
                        }

                        // Clip to text area so overflowing text won't draw over the widget background
                        const clipX = textX + 8;
                        const clipW = Math.max(0, textW - 16);
                        if (clipW > 8) {
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(clipX, textY, clipW, textH);
                            ctx.clip();

                            // Ellipsize if needed (guard measureText)
                            let drawText = String(textVal);
                            try {
                                if (ctx.measureText && ctx.measureText(drawText).width > clipW) {
                                    while (drawText.length > 0 && ctx.measureText(drawText + "…").width > clipW) {
                                        drawText = drawText.slice(0, -1);
                                    }
                                    drawText = drawText + "…";
                                }
                            } catch (e) {
                                // measureText may fail in some contexts; fall back to raw string
                            }
                            ctx.fillText(drawText, textX + 8, textY + textH / 2);
                            ctx.restore();
                        } else {
                            // Not enough room to clip — draw a short placeholder
                            ctx.fillText(name, textX + 8, textY + textH / 2);
                        }

                        // Remove (×) button (smaller) — use roundRect when available
                        const removeX = textX + textW + 4;
                        ctx.fillStyle = "#2b2b2b";
                        if (typeof ctx.roundRect === "function") {
                            ctx.beginPath();
                            ctx.roundRect(removeX, textY, removeW, textH, [4]);
                            ctx.fill();
                        } else {
                            ctx.fillRect(removeX, textY, removeW, textH);
                        }
                        ctx.fillStyle = "#ff8888";
                        ctx.textAlign = "center";
                        ctx.fillText("×", removeX + removeW / 2, textY + textH / 2);

                        ctx.restore();
                    };

                    widget.mouse = function (event, pos, node) {
                        if (!this.last_y) this.last_y = 0;
                        const localX = pos[0];
                        const localY = pos[1] - this.last_y;
                        const margin = 12;
                        const removeW = 20;
                        const totalW = node.size[0] - margin * 2;
                        const textW = totalW - (removeW + 4);
                        const textX = margin;
                        const textY = 2;
                        const textH = LiteGraph.NODE_WIDGET_HEIGHT - 4;

                        const removeX = textX + textW + 4;

                        if (event.type === "pointerdown") {
                            // click remove
                            if (localX >= removeX && localX <= removeX + removeW && localY >= textY && localY <= textY + textH) {
                                const pathWidgets = this.node.widgets.filter(w => w.name && w.name.startsWith("path_"));
                                if (pathWidgets.length > 2) {
                                    const idx = this.node.widgets.indexOf(this);
                                    if (idx !== -1) {
                                        this.node.widgets.splice(idx, 1);
                                        // Recompute node height only, preserve current width
                                        const curWidth = Array.isArray(this.node.size)
                                            ? this.node.size[0]
                                            : (typeof this.node.size === "number" ? this.node.size : undefined);
                                        const computed = this.node.computeSize(curWidth || undefined);
                                        if (computed && computed[1] != null) {
                                            this.node.size = [curWidth || (computed[0] || 0), computed[1]];
                                        }
                                        this.node.setDirtyCanvas(true, true);
                                    }
                                }
                                return true;
                            }

                            // click text area -> browse (unless shift-click to edit text)
                            if (localX >= textX && localX <= textX + textW && localY >= textY && localY <= textY + textH) {
                                if (event.shiftKey) {
                                    // open text prompt to edit value
                                    app.canvas.prompt("Path", this.value || "", (v) => {
                                        this.value = v;
                                        this.node.setDirtyCanvas(true, true);
                                    }, event);
                                    return true;
                                }
                                // browse
                                const input = document.createElement("input");
                                input.type = "file";
                                input.style.display = "none";
                                document.body.appendChild(input);
                                input.onchange = () => {
                                    if (input.files.length) {
                                        this.value = input.files[0].webkitRelativePath || input.files[0].name;
                                    }
                                    input.remove();
                                    this.node.setDirtyCanvas(true, true);
                                };
                                input.click();
                                return true;
                            }
                        }
                        return false;
                    };
                }

                // If we have pending widgets from a saved workflow, recreate them now.
                if (this._mediaPathsLoader_pending_widgets && Array.isArray(this._mediaPathsLoader_pending_widgets) && this._mediaPathsLoader_pending_widgets.length) {
                    for (let i = 0; i < this._mediaPathsLoader_pending_widgets.length && i < 8; i++) {
                        const v = this._mediaPathsLoader_pending_widgets[i];
                        // the saved widget value might be a string or an object; pass it through
                        addPathWidget.call(this, i + 1, typeof v === 'string' ? v : v);
                    }
                    // cleanup
                    delete this._mediaPathsLoader_pending_widgets;
                } else {
                    // Create initial 2 path widgets
                    for (let i = 1; i <= 2; i++) {
                        addPathWidget.call(this, i, "");
                    }
                }

                console.log("MediaPathsLoader widgets setup complete");

                // Refresh the node display while preserving user-resized width
                const curWidth = Array.isArray(this.size) ? this.size[0] : (typeof this.size === "number" ? this.size : 0);
                const curHeight = Array.isArray(this.size) ? this.size[1] : LiteGraph.NODE_WIDGET_HEIGHT || 0;
                const computed = this.computeSize(curWidth || undefined);
                this.size = [Math.max(curWidth || 0, computed[0] || 0), computed[1] || curHeight];
                this.setDirtyCanvas(true, true);
            });
        }
    },
});
