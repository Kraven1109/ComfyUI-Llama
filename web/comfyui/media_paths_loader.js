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
        console.log("beforeRegisterNodeDef called for:", nodeData.name);
        if (nodeData.name === "MediaPathsLoader") {
            console.log("Registering MediaPathsLoader extension...");

            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                console.log("MediaPathsLoader node created, adding widgets...");

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

                    // sizes for parts
                    const margin = 12;
                    const removeW = 20;
                    const browseW = 56;

                    widget.computeSize = function (width) {
                        return [width, LiteGraph.NODE_WIDGET_HEIGHT];
                    };

                    widget.draw = function (ctx, node, width, y, height) {
                        this.last_y = y;
                        const totalW = width - margin * 2;
                        const textW = totalW - (browseW + removeW + 8);
                        const textX = margin;
                        const textY = y + 2;
                        const textH = height - 4;

                        // text area background
                        ctx.save();
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        ctx.roundRect(textX, textY, textW, textH, [textH * 0.5]);
                        ctx.fill();

                        // text
                        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                        ctx.textBaseline = "middle";
                        ctx.textAlign = "left";
                        const textVal = this.value || "";
                        ctx.fillText(textVal, textX + 8, textY + textH / 2);

                        // Browse button
                        const browseX = textX + textW + 4;
                        ctx.fillStyle = "#2b2b2b";
                        ctx.roundRect(browseX, textY, browseW, textH, [4]);
                        ctx.fill();
                        ctx.fillStyle = "#ddd";
                        ctx.textAlign = "center";
                        ctx.fillText("Browse", browseX + browseW / 2, textY + textH / 2);

                        // Remove (×) button (smaller)
                        const removeX = browseX + browseW + 4;
                        ctx.fillStyle = "#2b2b2b";
                        ctx.roundRect(removeX, textY, removeW, textH, [4]);
                        ctx.fill();
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
                        const browseW = 56;
                        const totalW = node.size[0] - margin * 2;
                        const textW = totalW - (browseW + removeW + 8);
                        const textX = margin;
                        const textY = 2;
                        const textH = LiteGraph.NODE_WIDGET_HEIGHT - 4;

                        // click inside browse
                        const browseX = textX + textW + 4;
                        const removeX = browseX + browseW + 4;

                        if (event.type === "pointerdown") {
                            if (localX >= browseX && localX <= browseX + browseW && localY >= textY && localY <= textY + textH) {
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
                            // remove
                            if (localX >= removeX && localX <= removeX + removeW && localY >= textY && localY <= textY + textH) {
                                const pathWidgets = this.node.widgets.filter(w => w.name && w.name.startsWith("path_"));
                                if (pathWidgets.length > 2) {
                                    const idx = this.node.widgets.indexOf(this);
                                    if (idx !== -1) {
                                        this.node.widgets.splice(idx, 1);
                                        this.node.setSize(this.node.computeSize());
                                    }
                                }
                                return true;
                            }
                            // click text -> prompt edit
                            if (localX >= textX && localX <= textX + textW && localY >= textY && localY <= textY + textH) {
                                app.canvas.prompt("Path", this.value || "", (v) => {
                                    this.value = v;
                                    this.node.setDirtyCanvas(true, true);
                                }, event);
                                return true;
                            }
                        }
                        return false;
                    };
                }

                // Create initial 2 path widgets
                for (let i = 1; i <= 2; i++) {
                    addPathWidget.call(this, i, "");
                }

                console.log("MediaPathsLoader widgets setup complete");

                // Refresh the node display
                this.setSize(this.computeSize());
            });
        }
    },
});
