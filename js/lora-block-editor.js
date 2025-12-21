import {app} from "../../../scripts/app.js";

const url = new URL('./', import.meta.url).href;
let sdxl_unet = null;
fetch(url + "sdxl_unet.svg")
    .then(response => response.text())
    .then(svg => {
        sdxl_unet = svg;
    });

let sd15_unet = null;
fetch(url + "sd15_unet.svg")
    .then(response => response.text())
    .then(svg => {
        sd15_unet = svg;
    });

let dit = null;
fetch(url + "dit.svg")
    .then(response => response.text())
    .then(svg => {
        dit = svg;
    });




app.registerExtension({
    name: "LoRABlockEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const style = document.createElement('style');
        style.textContent = `
          .noselect {
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
          }`;
        document.head.appendChild(style);

        function set_svg_color(block, scale) {
            const c = Math.round(128 + (255 - 128) * scale);
            const fillColor = `rgb(${c}, ${c}, ${c})`;
            const rect = block.querySelector("rect");
            if (rect) rect.style.fill = fillColor;

            // Find a text element with ".sf" in its name to update scale
            const sfLabel = block.querySelector("[id$='.sf']");
            if (sfLabel) {
                const tspan = sfLabel.querySelector("tspan");
                if (tspan) {
                    tspan.textContent = `SF: ${scale.toFixed(2)}`;
                }
            }
        }

        function load_network(mode, div_unet) {
            if (mode === "sdxl") {
                div_unet.innerHTML = sdxl_unet;
            } else if (mode === "sd15") {
                div_unet.innerHTML = sd15_unet;
            } else if (mode === "dit") {
                div_unet.innerHTML = dit;
            }
        }

        function wire_block(block, settings, widget) {
            block.style.cursor = "ns-resize";  // Visual cue for up/down drag

            // Create a floating label element for scale display
            const scaleLabel = document.createElement("div");
            scaleLabel.style.position = "fixed";
            scaleLabel.style.pointerEvents = "none";
            scaleLabel.style.background = "black";
            scaleLabel.style.color = "white";
            scaleLabel.style.padding = "2px 6px";
            scaleLabel.style.borderRadius = "4px";
            scaleLabel.style.fontSize = "12px";
            scaleLabel.style.zIndex = "1000";
            scaleLabel.style.display = "none";
            document.body.appendChild(scaleLabel);

            let isDragging = false;
            let startY = 0;
            block.addEventListener("mousedown", (e) => {
                isDragging = true;
                startY = e.clientY;

                const onMouseMove = (e) => {
                    if (!isDragging) return;

                    e.preventDefault();
                    document.body.classList.add("noselect");

                    const dy = startY - e.clientY;  // positive = move up
                    let scale = settings.blockScales[block.id] + dy / 100;  // adjust sensitivity
                    scale = Math.min(1, Math.max(0, scale));  // clamp between 0 and 1
                    settings.blockScales[block.id] = scale;

                    widget.options.setValue(JSON.stringify(settings));

                    // Interpolate between grey (128) and white (255)
                    set_svg_color(block, scale);

                    startY = e.clientY;  // update for next movement

                    // Update label
                    scaleLabel.style.display = "block";
                    scaleLabel.style.left = `${e.clientX + 10}px`;
                    scaleLabel.style.top = `${e.clientY + 10}px`;
                    scaleLabel.textContent = "Block weight " + scale.toFixed(2);
                };

                const onMouseUpOrLeave = () => {
                    isDragging = false;
                    scaleLabel.style.display = "none";
                    document.body.classList.remove("noselect");
                    window.removeEventListener("mousemove", onMouseMove);
                    window.removeEventListener("mouseup", onMouseUpOrLeave);
                    window.removeEventListener("mouseleave", onMouseUpOrLeave);
                };

                window.addEventListener("mousemove", onMouseMove);
                window.addEventListener("mouseup", onMouseUpOrLeave);
                window.addEventListener("mouseleave", onMouseUpOrLeave);
            });
        }

        if (nodeData?.name === "PM LoRA Modifier") {
            let settings = {"mode": "sdxl", "blockScales": {}};

            nodeType.prototype.onNodeCreated = function () {
                const node = this;

                // Container for both widgets
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.height = "100%";    // fill node height

                // Add a dropdown where the use can select the model type
                const modelTypeSelect = document.createElement("select");
                modelTypeSelect.style.margin = "4px";
                modelTypeSelect.style.width = "100%";
                modelTypeSelect.innerHTML = `
                    <option value="sdxl">SDXL</option>
                    <option value="sd15">SD 1.5</option>
                    <option value="dit">DiT (40 layers)</option>
                `;

                // SVG container - flex grow
                const div_unet = document.createElement("div");
                div_unet.id = "sdxl_unet";
                div_unet.style.overflow = "hidden";
                div_unet.style.flexGrow = "1";      // fill available vertical space

                // Add model type select and SVG container to the main container
                container.appendChild(modelTypeSelect);
                container.appendChild(div_unet);

                // Add an event listener to the model type select and update the SVG
                modelTypeSelect.addEventListener("change", (e) => {
                    const selectedModel = e.target.value;
                    load_network(selectedModel, div_unet);
                    // Reset block scales for the new SVG
                    // Therefore we need to find out all blocks in the new SVG
                    let allBlocks;
                    if (selectedModel === "dit") {
                        // For DiT, select layer groups
                        allBlocks = div_unet.querySelectorAll("[id*='layers_group.']");
                    } else {
                        // For SD/SDXL, select blocks
                        allBlocks = div_unet.querySelectorAll("[id*='blocks.']");
                    }
                    const blocks = Array.from(allBlocks)
                        .filter(el => !el.id.includes("rect") && !el.id.includes("grid") && !el.id.includes(".sf"));
                    settings = {"mode": selectedModel, blockScales: {}}
                    blocks.forEach(block => {
                        const blockId = block.id;
                        settings.blockScales[blockId] = 1;
                        // Set initial color
                        set_svg_color(block, settings.blockScales[blockId]);
                        // Create event listeners for the new blocks
                        wire_block(block, settings, newWidget);
                    });
                    // Update the widget with the new block scales
                    newWidget.options.setValue(JSON.stringify(settings));
                });


                // Add entire container as one DOM widget
                const newWidget = node.addDOMWidget("text", "blocks_store", container, {
                    blocksStoreWidget: node.widgets.find(w => w.name === "blocks_store"),
                    getValue() {
                        return JSON.stringify(settings);
                    },
                    setValue(v) {
                        let parsed = {};
                        try {
                            parsed = JSON.parse(v || "{}");
                        } catch (e) {
                            console.warn("Failed to parse blocks_store JSON:", v);
                        }
                        settings = parsed;

                        // Apply the parsed block scales to the SVG blocks
                        if (!settings.blockScales) {
                            settings.blockScales = {};
                        }
                        for (const [blockId, scale] of Object.entries(settings.blockScales)) {
                            const el = div_unet.querySelector(`[id="${blockId}"]`);
                            if (el)
                                set_svg_color(el, scale);
                            else
                                console.warn(`Block with ID ${blockId} not found in SVG.`);
                        }

                        // Update the original text widget if needed (optional)
                        if (this.blocksStoreWidget) {
                            this.blocksStoreWidget.value = JSON.stringify(parsed, null, 2);
                        }
                    }
                });


                setTimeout(() => {
                    if (settings.mode) {
                        modelTypeSelect.value = settings.mode;
                        load_network(settings.mode, div_unet);
                    }
                     // If blockScales is empty, initialize it with the new SVG blocks
                    if (!settings.blockScales || Object.keys(settings.blockScales).length === 0) {
                        modelTypeSelect.dispatchEvent(new Event("change"));
                    }else {
                        // If blockScales already has values, apply them to the existing blocks
                        let allBlocks;
                        if (settings.mode === "dit") {
                            allBlocks = div_unet.querySelectorAll("[id*='layers_group.']");
                        } else {
                            allBlocks = div_unet.querySelectorAll("[id*='blocks.']");
                        }
                        const blocks = Array.from(allBlocks)
                            .filter(el => !el.id.includes("rect") && !el.id.includes("grid") && !el.id.includes(".sf"));

                        blocks.forEach(block => {
                            let blockId = block.id;
                            set_svg_color(block, settings.blockScales[blockId]);
                            // Create event listeners for the new blocks
                            wire_block(block, settings, newWidget);
                        });
                    }
                }, 100);
            };
        }
        if (nodeData?.name === "PM LoRA Block Display") {
            nodeType.prototype.onNodeCreated = function () {
                const node = this;

                // Container for both widgets
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.height = "100%";    // fill node height
                container.innerHTML = sdxl_unet;
                container.id = "sdxl_unet";

                const svgDisplayWidget = node.addDOMWidget("text", "svg_display", container, {
                    getValue: () => "",
                    setValue: (val) => {
                        console.log("Setting SVG value:", val);
                    },
                });

                // add onExecuted hook to update the widget when the node is executed
                node.onExecuted = function(output_data, context, graph_canvas) {
                    console.log("Node executed, output_data is:", output_data);
                };
            };
        }
    }
});
