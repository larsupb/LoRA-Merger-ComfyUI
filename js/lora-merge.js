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
            nodeType.prototype.onConnectionsChange = function (type, index, connected) {
                // Check if the event type is 1 (input)
                if (type !== 1) return;

                let first_lora_idx = 2;
                let last_lora_idx = this.inputs.length - 2;
                let lora_slot_count = last_lora_idx - first_lora_idx + 1;
                if (index < first_lora_idx) return;

                if (index !== this.inputs.length - 1) {
                    if (this.inputs[index].name === "layer_filter") {
                        // Move the non-LoRA input to the end
                        let temp_type = this.inputs[index].type
                        let temp_widget = this.inputs[index].widget;
                        this.removeInput(index);
                        this.addInput(`layer_filter`, temp_type);
                        this.inputs[this.inputs.length - 1].widget = temp_widget;
                        this.computeSize(); // update visual layout
                    }
                    this.inputs[index].name = `lora${index - first_lora_idx + 1}`;
                }

                // Add new input if the second last one is connected
                if (connected) {
                    // Count number of disconnected LoRA slots
                    let disconnected_slots = 0;
                    for (let i = first_lora_idx; i < last_lora_idx; i++) {
                        if (this.inputs[i].link === null) {
                            disconnected_slots++;
                        }
                    }
                    // If there are no disconnected slots, add a new one
                    if (disconnected_slots > 0) return;
                    // Add the new input (before the last one)
                    this.addInput(`lora${lora_slot_count + 1}`, this.inputs[first_lora_idx].type);
                    // Switch the last input to the new one
                    let temp = this.inputs[this.inputs.length - 1]
                    this.inputs[this.inputs.length - 1] = this.inputs[this.inputs.length - 2];
                    this.inputs[this.inputs.length - 2] = temp;

                    this.computeSize(); // update visual layout
                }
                else if (!connected) {
                    if (index === last_lora_idx) {
                        // Do nothing, because we want to keep one disconnected lora slot
                    }
                    // Disconnect the index input, rearrange the inputs and remove the last one
                    else {
                        // Rearrange inputs
                        for (let i = index; i < last_lora_idx; i++) {
                            this.inputs[i] = this.inputs[i + 1];
                        }
                        this.removeInput(last_lora_idx);
                        for (let i = 0; i < lora_slot_count - 1; i++) {
                            this.inputs[first_lora_idx + i].name = `lora${i + 1}`;
                        }
                    }
                }
            }
        }
    },
});