import {app} from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.LoRAMerger",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === 'PM LoRA Merger' || nodeData.name === 'PM LoRA SVD Merger') {
            nodeType.prototype.onConnectionsChange = function (type, index, connected) {
                if (type !== 1 ) return;

                const loraInputs = this.inputs.filter(input => input.name.startsWith('lora'));

                loraInputs.forEach((input, i) => input.name = `lora${i + 1}`);

                if (connected && loraInputs.at(-1).link !== null) {
                    this.addInput(`lora${loraInputs.length + 1}`, loraInputs[0].type);
                } else {
                    if (loraInputs.length > 1 && loraInputs.at(-2).link == null) {
                        const lastLoraInput = loraInputs.at(-1);
                        this.removeInput(this.inputs.indexOf(lastLoraInput));
                    }
                }
            }
        }
    },
});
