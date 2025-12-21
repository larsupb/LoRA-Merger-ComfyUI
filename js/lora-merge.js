import {app} from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.LoRAMerger",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        if (nodeData.name === 'PM LoRA Merger' || nodeData.name === 'PM LoRA SVD Merger') {
            nodeType.prototype.onConnectionsChange = function (type, index, connected) {
                if (type !== 1) return;

                this.inputs.forEach((input, i) => input.name = `lora${i + 1}`);

                if (connected && this.inputs[this.inputs.length - 1].link !== null) {
                    this.addInput(`lora${this.inputs.length + 1}`, this.inputs[0].type);
                } else {
                    if (this.inputs.length > 1 && this.inputs[this.inputs.length - 2].link == null)
                        this.removeInput(this.inputs.length - 1);
                }
            }
        }

        if (nodeData.name === 'PM LoRA Stacker') {
            // Add initialization to ensure proper state on load
            nodeType.prototype.onNodeCreated = function() {
                this.ensureExactlyOneFreeLoRASlot = () => {
                    const first_lora_idx = 1; // model, lora1, ..., layer_filter
                    const last_input_idx = this.inputs.length - 1;
                    const layer_filter_idx = last_input_idx;

                    // Find all LoRA inputs (between first_lora_idx and layer_filter)
                    let lora_inputs = [];
                    for (let i = first_lora_idx; i < layer_filter_idx; i++) {
                        lora_inputs.push(i);
                    }

                    // Count connected and disconnected LoRA slots
                    let connected_count = 0;
                    let disconnected_count = 0;

                    for (let idx of lora_inputs) {
                        if (this.inputs[idx].link !== null && this.inputs[idx].link !== undefined) {
                            connected_count++;
                        } else {
                            disconnected_count++;
                        }
                    }

                    // Ensure exactly one free slot
                    if (disconnected_count === 0) {
                        // Need to add a slot - insert before layer_filter
                        const new_lora_num = lora_inputs.length + 1;
                        const lora_type = this.inputs[first_lora_idx].type;
                        this.addInput(`lora${new_lora_num}`, lora_type);

                        // Move layer_filter to the end
                        let temp = this.inputs[this.inputs.length - 1];
                        this.inputs[this.inputs.length - 1] = this.inputs[this.inputs.length - 2];
                        this.inputs[this.inputs.length - 2] = temp;
                    } else if (disconnected_count > 1) {
                        // Too many free slots - remove extras from the end (but keep layer_filter)
                        let to_remove = disconnected_count - 1;
                        for (let i = layer_filter_idx - 1; i >= first_lora_idx && to_remove > 0; i--) {
                            if (this.inputs[i].link === null || this.inputs[i].link === undefined) {
                                this.removeInput(i);
                                to_remove--;
                            }
                        }
                    }

                    // Renumber all LoRA inputs sequentially
                    let lora_num = 1;
                    for (let i = first_lora_idx; i < this.inputs.length - 1; i++) {
                        this.inputs[i].name = `lora${lora_num}`;
                        lora_num++;
                    }

                    this.computeSize();
                };
            };

            nodeType.prototype.onConnectionsChange = function (type, index, connected) {
                // Check if the event type is 1 (input)
                if (type !== 1) return;

                const first_lora_idx = 1;
                const last_input_idx = this.inputs.length - 1;

                // Ignore changes to model, clip, or layer_filter
                if (index < first_lora_idx || index === last_input_idx) return;

                // Ensure the layer_filter is always at the end
                if (this.inputs[index].name === "layer_filter" && index !== last_input_idx) {
                    // Move the layer_filter input to the end
                    let temp_type = this.inputs[index].type;
                    let temp_widget = this.inputs[index].widget;
                    this.removeInput(index);
                    this.addInput(`layer_filter`, temp_type);
                    this.inputs[last_input_idx].widget = temp_widget;
                    this.computeSize();
                    return;
                }

                // Use the helper function to maintain exactly one free slot
                if (this.ensureExactlyOneFreeLoRASlot) {
                    this.ensureExactlyOneFreeLoRASlot();
                }
            };

            // Call initialization on graph load to fix state after reload
            const origOnGraphConfigured = nodeType.prototype.onGraphConfigured;
            nodeType.prototype.onGraphConfigured = function() {
                if (origOnGraphConfigured) {
                    origOnGraphConfigured.apply(this, arguments);
                }
                // Ensure proper state after loading from workflow
                if (this.ensureExactlyOneFreeLoRASlot) {
                    setTimeout(() => {
                        this.ensureExactlyOneFreeLoRASlot();
                    }, 100);
                }
            };
        }
    },
});