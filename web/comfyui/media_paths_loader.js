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
        if (nodeData.name === "MediaPathsLoader") {
            console.log("Registering MediaPathsLoader extension...");

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                console.log("MediaPathsLoader node created, adding widgets...");
            });

            chainCallback(nodeType.prototype, "configure", function (info) {
                try {
                    if (info && info.widgets_values) {
                        this._mediaPathsLoader_pending_widgets = info.widgets_values;
                    }
                } catch (e) {
                    console.error("MediaPathsLoader: error handling configure:", e);
                }
            });

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                const addCallback = () => {
                    const pathWidgets = this.widgets.filter(w => w.name && w.name.startsWith("path_"));
                    const nextIndex = pathWidgets.length + 1;
                    if (nextIndex <= 8) {
                        addPathWidget.call(this, nextIndex, "");
                    }
                };

                const addButton = this.addWidget("button", "+", "", addCallback);
                addButton.serialize = false;

                function addPathWidget(index, initialValue) {
                    const name = `path_${index}`;
                    const widget = this.addWidget("custom", name, initialValue || "", null);
                    widget.serialize = true;
                    widget.node = this;
                    widget.name = name;
                    widget.value = initialValue || "";

                    // Constants
                    const margin = 12;
                    const removeW = 20;

                    // Ensure the widget takes up the full width of the node
                    widget.computeSize = function (width) {
                        return [width, LiteGraph.NODE_WIDGET_HEIGHT];
                    };

                    widget.draw = function (ctx, node, width, y, height) {
                        this.last_y = y;
                        
                        // --- VISUALS ---
                        // Calculate positions anchored to the RIGHT side
                        // This ensures the X button is always in the same spot regardless of text length
                        const removeX = width - margin - removeW;
                        const textW = removeX - margin - 4; 
                        const textX = margin;
                        const textY = y + 2;
                        const textH = height - 4;

                        // 1. Draw Text Background
                        ctx.save();
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        if (typeof ctx.roundRect === "function") {
                            ctx.beginPath();
                            ctx.roundRect(textX, textY, textW, textH, [textH * 0.5]);
                            ctx.fill();
                        } else {
                            ctx.fillRect(textX, textY, textW, textH);
                        }

                        // 2. Draw Text
                        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                        ctx.textBaseline = "middle";
                        ctx.textAlign = "left";
                        
                        let textVal = name;
                        if (typeof this.value === "string") {
                            textVal = this.value.length ? this.value : name;
                        } else if (this.value && typeof this.value === "object") {
                            textVal = this.value.name || this.value.file || this.value.path || name;
                        }

                        // Clip Text
                        const clipX = textX + 8;
                        const clipW = Math.max(0, textW - 16);
                        
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(clipX, textY, clipW, textH);
                        ctx.clip();
                        
                        // Simple text drawing
                        ctx.fillText(String(textVal), textX + 8, textY + textH / 2);
                        ctx.restore();

                        // 3. Draw Remove Button
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
                        
                        const x = pos[0];
                        const y = pos[1] - this.last_y;
                        const widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT;
                        
                        // Use current node width to find button position (Right Anchor)
                        const nodeWidth = node.size[0];
                        const removeX = nodeWidth - margin - removeW;
                        const textMaxX = removeX - 4;

                        if (event.type === "pointerdown") {
                            
                            // --- INTERACTION 1: REMOVE ---
                            // We removed the strict 'y' check to make clicking easier
                            if (x >= removeX && x <= (removeX + removeW) && y >= 0 && y <= widgetHeight) {
                                const pathWidgets = this.node.widgets.filter(w => w.name && w.name.startsWith("path_"));
                                if (pathWidgets.length > 2) {
                                    const idx = this.node.widgets.indexOf(this);
                                    if (idx !== -1) {
                                        // 1. Remove Widget
                                        this.node.widgets.splice(idx, 1);
                                        
                                        // 2. Re-index Names (path_3 -> path_2)
                                        const remaining = this.node.widgets.filter(w => w.name && w.name.startsWith("path_"));
                                        remaining.forEach((w, i) => {
                                            w.name = `path_${i + 1}`;
                                        });

                                        // 3. DO NOT RESIZE
                                        // We deleted the code that calls computeSize and sets this.node.size
                                        // The node will keep its current Width and Height.
                                        
                                        this.node.setDirtyCanvas(true, true);
                                    }
                                }
                                return true;
                            }

                            // --- INTERACTION 2: BROWSE/EDIT ---
                            if (x >= margin && x <= textMaxX && y >= 0 && y <= widgetHeight) {
                                if (event.shiftKey) {
                                    app.canvas.prompt("Path", this.value || "", (v) => {
                                        this.value = v;
                                        this.node.setDirtyCanvas(true, true);
                                    }, event);
                                    return true;
                                }
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

                // Initialization Logic
                if (this._mediaPathsLoader_pending_widgets && Array.isArray(this._mediaPathsLoader_pending_widgets) && this._mediaPathsLoader_pending_widgets.length) {
                    for (let i = 0; i < this._mediaPathsLoader_pending_widgets.length && i < 8; i++) {
                        const v = this._mediaPathsLoader_pending_widgets[i];
                        addPathWidget.call(this, i + 1, typeof v === 'string' ? v : v);
                    }
                    delete this._mediaPathsLoader_pending_widgets;
                } else {
                    for (let i = 1; i <= 2; i++) {
                        addPathWidget.call(this, i, "");
                    }
                }

                console.log("MediaPathsLoader widgets setup complete");

                // Initial size setup is still okay to ensure it starts clean
                const curWidth = Array.isArray(this.size) ? this.size[0] : (typeof this.size === "number" ? this.size : 0);
                const curHeight = Array.isArray(this.size) ? this.size[1] : LiteGraph.NODE_WIDGET_HEIGHT || 0;
                const computed = this.computeSize(curWidth || undefined);
                this.size = [Math.max(curWidth || 0, computed[0] || 0), computed[1] || curHeight];
                this.setDirtyCanvas(true, true);
            });
        }
    },
});