import { app } from "../../../../../scripts/app.js";

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
                        const textWidget = this.addWidget("text", `path_${nextIndex}`, "", () => {});
                        console.log(`Added new text widget path_${nextIndex}`);

                        // Add browse button for new widget
                        const browseButton = this.addWidget("button", "Browse", "", () => {
                            const input = document.createElement("input");
                            input.type = "file";
                            input.style.display = "none";
                            document.body.appendChild(input);
                            input.onchange = () => {
                                if (input.files.length) {
                                    textWidget.value = input.files[0].webkitRelativePath || input.files[0].name;
                                    console.log(`Set path_${nextIndex} to: ${textWidget.value}`);
                                }
                                input.remove();
                            };
                            input.click();
                        });
                        browseButton.serialize = false;

                        // Add remove button for this new widget
                        const removeButton = this.addWidget("button", "x", "", () => {
                            const idx = this.widgets.indexOf(textWidget);
                            if (idx !== -1) {
                                this.widgets.splice(idx, 3); // text, browse, remove buttons
                                console.log(`Removed widget path_${nextIndex}`);
                            }
                        });
                        removeButton.serialize = false;

                        // Move the new widgets to after the + button
                        const startIndex = this.widgets.length - 3;
                        const widgetsToMove = this.widgets.splice(startIndex, 3);
                        const insertIndex = 1; // after the + button at index 0
                        this.widgets.splice(insertIndex, 0, ...widgetsToMove);
                    }
                };

                // Add '+' button at the top
                const addButton = this.addWidget("button", "+", "", addCallback);
                addButton.serialize = false;

                // Create initial 2 path widgets
                for (let i = 1; i <= 2; i++) {
                    const textWidget = this.addWidget("text", `path_${i}`, "", () => {});
                    console.log(`Added text widget path_${i}`);

                    // Add browse button
                    const browseButton = this.addWidget("button", "Browse", "", () => {
                        const input = document.createElement("input");
                        input.type = "file";
                        input.style.display = "none";
                        document.body.appendChild(input);
                        input.onchange = () => {
                            if (input.files.length) {
                                textWidget.value = input.files[0].webkitRelativePath || input.files[0].name;
                                console.log(`Set path_${i} to: ${textWidget.value}`);
                            }
                            input.remove();
                        };
                        input.click();
                    });
                    browseButton.serialize = false;

                    // Add remove button
                    const removeButton = this.addWidget("button", "x", "", () => {
                        const pathWidgets = this.widgets.filter(w => w.name && w.name.startsWith("path_"));
                        if (pathWidgets.length > 2) { // Keep at least 2
                            const idx = this.widgets.indexOf(textWidget);
                            if (idx !== -1) {
                                this.widgets.splice(idx, 3); // text, browse, remove buttons
                                console.log(`Removed widget path_${i}`);
                            }
                        }
                    });
                    removeButton.serialize = false;
                }

                console.log("MediaPathsLoader widgets setup complete");

                // Refresh the node display
                this.setSize(this.computeSize());
            });
        }
    },
});
