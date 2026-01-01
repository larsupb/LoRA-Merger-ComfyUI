import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Default data structure for a LoRA widget
const DEFAULT_LORA_WIDGET_DATA = {
    on: true,
    lora: null,
    strength: 1.0,
};

// Utility function to get list of available LoRAs
async function getLorasList() {
    try {
        const response = await fetch('/object_info');
        const objectInfo = await response.json();

        // Find LoRA loader nodes and extract lora_name options
        for (const nodeType in objectInfo) {
            const nodeInfo = objectInfo[nodeType];
            if (nodeInfo.input && nodeInfo.input.required && nodeInfo.input.required.lora_name) {
                const loraOptions = nodeInfo.input.required.lora_name[0];
                if (Array.isArray(loraOptions)) {
                    return loraOptions;
                }
            }
        }

        return [];
    } catch (error) {
        console.error("Error fetching LoRAs list:", error);
        return [];
    }
}

class PMLoraStackerWidget {
    constructor(name, node) {
        this.name = name;
        this.node = node;
        this.type = "pm_lora_widget";
        this.value = { ...DEFAULT_LORA_WIDGET_DATA };
        this.options = { serialize: true };

        // For mouse interaction
        this.lastY = 0;
        this.height = 30;
    }

    draw(ctx, node, widgetWidth, y, widgetHeight) {
        const margin = 10;
        const innerMargin = 3;

        this.lastY = y;

        // Background
        ctx.fillStyle = this.value.on ? "#2a2a2a" : "#1a1a1a";
        ctx.fillRect(margin, y, widgetWidth - margin * 2, widgetHeight);

        // Border
        ctx.strokeStyle = "#4a4a4a";
        ctx.lineWidth = 1;
        ctx.strokeRect(margin, y, widgetWidth - margin * 2, widgetHeight);

        let posX = margin + innerMargin;
        const midY = y + widgetHeight / 2;

        // Toggle checkbox
        const checkboxSize = 16;
        ctx.fillStyle = this.value.on ? "#4a9eff" : "#666";
        ctx.fillRect(posX, midY - checkboxSize / 2, checkboxSize, checkboxSize);

        if (this.value.on) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(posX + 3, midY);
            ctx.lineTo(posX + 6, midY + 4);
            ctx.lineTo(posX + 12, midY - 4);
            ctx.stroke();
        }

        posX += checkboxSize + innerMargin * 2;
        const loraNameStartX = posX;

        // Calculate available space for LoRA name
        const strengthWidth = 80;
        const reservedRightSpace = strengthWidth + margin + innerMargin * 3;

        // LoRA name - use all available space
        const loraNameWidth = Math.max(100, widgetWidth - loraNameStartX - reservedRightSpace);

        ctx.fillStyle = this.value.on ? "#ddd" : "#666";
        ctx.font = "12px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        const loraName = this.value.lora || "None";
        const displayName = this.truncateText(ctx, loraName, loraNameWidth);
        ctx.fillText(displayName, posX, midY);

        posX = widgetWidth - margin - innerMargin;

        // Strength control
        posX -= strengthWidth;
        this.drawStrengthControl(ctx, posX, midY, strengthWidth, this.value.strength, "strength");

        // Store the actual LoRA name width for mouse hit detection
        this.loraNameWidth = loraNameWidth;

        return widgetHeight;
    }

    drawStrengthControl(ctx, x, y, width, value, type) {
        const btnWidth = 18;
        const textWidth = width - btnWidth * 2 - 4;

        // Decrement button
        ctx.fillStyle = "#555";
        ctx.fillRect(x, y - 8, btnWidth, 16);
        ctx.fillStyle = "#ddd";
        ctx.font = "bold 12px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("-", x + btnWidth / 2, y);

        // Value display
        ctx.fillStyle = "#333";
        ctx.fillRect(x + btnWidth + 2, y - 8, textWidth, 16);
        ctx.fillStyle = "#ddd";
        ctx.font = "11px monospace";
        ctx.fillText(value.toFixed(2), x + btnWidth + 2 + textWidth / 2, y);

        // Increment button
        ctx.fillStyle = "#555";
        ctx.fillRect(x + btnWidth + textWidth + 4, y - 8, btnWidth, 16);
        ctx.fillStyle = "#ddd";
        ctx.font = "bold 12px monospace";
        ctx.fillText("+", x + btnWidth + textWidth + 4 + btnWidth / 2, y);

        // Store hit areas for this control
        if (!this.hitAreas) this.hitAreas = {};
        this.hitAreas[type] = {
            dec: { x: x, y: y - 8, width: btnWidth, height: 16 },
            inc: { x: x + btnWidth + textWidth + 4, y: y - 8, width: btnWidth, height: 16 },
            val: { x: x + btnWidth + 2, y: y - 8, width: textWidth, height: 16 }
        };
    }

    truncateText(ctx, text, maxWidth) {
        const metrics = ctx.measureText(text);
        if (metrics.width <= maxWidth) {
            return text;
        }

        let truncated = text;
        while (ctx.measureText(truncated + "...").width > maxWidth && truncated.length > 0) {
            truncated = truncated.slice(0, -1);
        }
        return truncated + "...";
    }

    mouse(event, pos, node) {
        const widgetWidth = node.size[0];
        const margin = 10;
        const innerMargin = 3;
        const checkboxSize = 16;
        const checkboxX = margin + innerMargin;
        const checkboxY = this.lastY + this.height / 2 - checkboxSize / 2;

        const localX = pos[0];
        const localY = pos[1];

        if (event.type === "pointerdown" && event.button === 0) {
            // Check checkbox
            if (localX >= checkboxX && localX <= checkboxX + checkboxSize &&
                localY >= checkboxY && localY <= checkboxY + checkboxSize) {
                this.value.on = !this.value.on;
                return true;
            }

            // Check strength control
            if (this.hitAreas && this.hitAreas.strength) {
                const result = this.checkStrengthHit(localX, localY, "strength");
                if (result) return true;
            }

            // Check LoRA name area (for opening selector)
            const loraNameX = checkboxX + checkboxSize + innerMargin * 2;
            const loraNameWidth = this.loraNameWidth || 280;
            if (localX >= loraNameX && localX <= loraNameX + loraNameWidth &&
                localY >= this.lastY && localY <= this.lastY + this.height) {
                this.openLoraSelector(node, event);
                return true;
            }
        }

        return false;
    }

    checkStrengthHit(x, y, type) {
        const areas = this.hitAreas[type];
        if (!areas) return false;

        // Check decrement button
        if (x >= areas.dec.x && x <= areas.dec.x + areas.dec.width &&
            y >= areas.dec.y && y <= areas.dec.y + areas.dec.height) {
            this.adjustStrength(type, -0.05);
            return true;
        }

        // Check increment button
        if (x >= areas.inc.x && x <= areas.inc.x + areas.inc.width &&
            y >= areas.inc.y && y <= areas.inc.y + areas.inc.height) {
            this.adjustStrength(type, 0.05);
            return true;
        }

        // Check value area (for direct input)
        if (x >= areas.val.x && x <= areas.val.x + areas.val.width &&
            y >= areas.val.y && y <= areas.val.y + areas.val.height) {
            this.promptStrength(type);
            return true;
        }

        return false;
    }

    adjustStrength(type, delta) {
        this.value.strength = Math.round((this.value.strength + delta) * 100) / 100;
        app.graph.setDirtyCanvas(true);
    }

    promptStrength(type) {
        const currentValue = this.value.strength;

        const newValue = prompt(`Enter strength:`, currentValue);
        if (newValue !== null) {
            const numValue = parseFloat(newValue);
            if (!isNaN(numValue)) {
                this.value.strength = Math.round(numValue * 100) / 100;
                app.graph.setDirtyCanvas(true);
            }
        }
    }

    async openLoraSelector(node, event) {
        const loras = await getLorasList();

        // Build simple array of LoRA names (enables built-in search)
        const loraOptions = ["None", ...loras];

        // Use LiteGraph's context menu with array of strings (enables search)
        new LiteGraph.ContextMenu(loraOptions, {
            event: event,
            title: "Select LoRA",
            scale: Math.max(1.0, app.canvas.ds?.scale || 1.0),
            className: "dark",
            callback: (value) => {
                if (value && value.content) {
                    // Handle object format (shouldn't happen with string array)
                    this.value.lora = value.content === "None" ? null : value.content;
                } else {
                    // Handle string format (expected)
                    this.value.lora = value === "None" ? null : value;
                }
                app.graph.setDirtyCanvas(true);
            }
        });
    }

    serializeValue() {
        return { ...this.value };
    }
}

// Register the extension
app.registerExtension({
    name: "Comfy.PMPowerLoraStacker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PM LoRA Power Stacker") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                this.serialize_widgets = true;
                this.loraWidgetsCounter = 0;

                // Add layer filter widget (architecture-agnostic presets)
                this.addWidget(
                    "combo",
                    "layer_filter",
                    "full",
                    (value) => {},
                    { values: ["full", "attn-only", "mlp-only", "attn-mlp"] }
                );

                // Add divider
                this.addWidget("button", "━━━━━━━━━━━━━━━", null, () => {});

                // Add "Add Lora" button
                this.addWidget("button", "➕ Add Lora", null, () => {
                    this.addLoraWidget();
                });

                this.setSize(this.computeSize());
            };

            nodeType.prototype.addLoraWidget = function() {
                this.loraWidgetsCounter++;
                const widget = new PMLoraStackerWidget(`lora_${this.loraWidgetsCounter}`, this);

                // Insert before the "Add Lora" button (last widget)
                const addButton = this.widgets[this.widgets.length - 1];
                this.widgets.splice(this.widgets.length - 1, 0, widget);

                // Only update height to fit widgets, preserve user's width
                const computedSize = this.computeSize();
                this.setSize([
                    Math.max(this.size[0], computedSize[0]),  // Keep user's width if larger
                    Math.max(this.size[1], computedSize[1])   // Expand height if needed
                ]);
                app.graph.setDirtyCanvas(true);

                return widget;
            };

            nodeType.prototype.removeLoraWidget = function(widget) {
                const index = this.widgets.indexOf(widget);
                if (index !== -1) {
                    this.widgets.splice(index, 1);

                    // Only update height to fit remaining widgets, preserve user's width
                    const computedSize = this.computeSize();
                    this.setSize([
                        Math.max(this.size[0], computedSize[0]),  // Keep user's width if larger
                        Math.max(computedSize[1], 100)            // Shrink height but maintain minimum
                    ]);
                    app.graph.setDirtyCanvas(true);
                }
            };

            // Override getSlotInPosition to detect mouse over widget areas
            nodeType.prototype.getSlotInPosition = function(canvasX, canvasY) {
                // Call parent implementation first
                const slot = LGraphNode.prototype.getSlotInPosition.call(this, canvasX, canvasY);

                if (!slot) {
                    // Check if mouse is over a widget
                    let lastWidget = null;
                    for (const widget of this.widgets) {
                        if (widget instanceof PMLoraStackerWidget) {
                            if (!widget.lastY) continue;

                            // Check if mouse Y is past this widget
                            if (canvasY > this.pos[1] + widget.lastY) {
                                lastWidget = widget;
                                continue;
                            }
                            break;
                        }
                    }

                    // If we found a LoRA widget under the mouse, return it as a virtual slot
                    if (lastWidget && lastWidget instanceof PMLoraStackerWidget) {
                        // Check if mouse is actually within the widget bounds
                        const widgetBottom = this.pos[1] + lastWidget.lastY + lastWidget.height;
                        if (canvasY <= widgetBottom) {
                            return { widget: lastWidget, output: { type: "PM_LORA_WIDGET" } };
                        }
                    }
                }

                return slot;
            };

            // Override getSlotMenuOptions to show context menu for widgets
            nodeType.prototype.getSlotMenuOptions = function(slot) {
                // Check if this is a LoRA widget slot
                if (slot?.widget && slot.widget instanceof PMLoraStackerWidget) {
                    const widget = slot.widget;
                    const index = this.widgets.indexOf(widget);

                    // Find all LoRA widgets to determine if we can move up/down
                    const loraWidgets = this.widgets.filter(w => w instanceof PMLoraStackerWidget);
                    const loraIndex = loraWidgets.indexOf(widget);

                    const canMoveUp = loraIndex > 0;
                    const canMoveDown = loraIndex < loraWidgets.length - 1;

                    const menuItems = [
                        {
                            content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
                            callback: () => {
                                widget.value.on = !widget.value.on;
                                app.graph.setDirtyCanvas(true);
                            }
                        },
                        null, // Separator
                        {
                            content: "⬆️ Move Up",
                            disabled: !canMoveUp,
                            callback: () => {
                                if (canMoveUp) {
                                    const prevLoraWidget = loraWidgets[loraIndex - 1];
                                    const prevIndex = this.widgets.indexOf(prevLoraWidget);
                                    // Swap positions
                                    this.widgets[index] = prevLoraWidget;
                                    this.widgets[prevIndex] = widget;
                                    app.graph.setDirtyCanvas(true);
                                }
                            }
                        },
                        {
                            content: "⬇️ Move Down",
                            disabled: !canMoveDown,
                            callback: () => {
                                if (canMoveDown) {
                                    const nextLoraWidget = loraWidgets[loraIndex + 1];
                                    const nextIndex = this.widgets.indexOf(nextLoraWidget);
                                    // Swap positions
                                    this.widgets[index] = nextLoraWidget;
                                    this.widgets[nextIndex] = widget;
                                    app.graph.setDirtyCanvas(true);
                                }
                            }
                        },
                        null, // Separator
                        {
                            content: "🗑️ Remove",
                            callback: () => {
                                this.removeLoraWidget(widget);
                            }
                        }
                    ];

                    new LiteGraph.ContextMenu(menuItems, {
                        title: "LoRA Widget",
                        event: event
                    });

                    // Return undefined to prevent default menu
                    return undefined;
                }

                // For non-widget slots, call parent implementation if it exists
                return LGraphNode.prototype.getSlotMenuOptions ?
                    LGraphNode.prototype.getSlotMenuOptions.call(this, slot) : null;
            };

            // Override onConfigure to restore widgets from saved data
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                // Save the user's size before onConfigure potentially changes it
                const savedSize = this.size ? [...this.size] : null;

                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }

                // Restore LoRA widgets from widgets_values
                if (info.widgets_values) {
                    // Clear existing LoRA widgets (keep layer_filter, divider, and add button)
                    const nonLoraWidgets = this.widgets.filter(w => !(w instanceof PMLoraStackerWidget));
                    this.widgets = nonLoraWidgets;

                    // Restore LoRA widgets
                    info.widgets_values.forEach((widgetValue, index) => {
                        // Skip non-LoRA widget values (first value is layer_filter)
                        if (index === 0) return; // layer_filter

                        if (widgetValue && typeof widgetValue === 'object' && widgetValue.lora !== undefined) {
                            const widget = this.addLoraWidget();
                            widget.value = { ...widgetValue };
                        }
                    });
                }

                // Restore the user's custom size after widgets are added
                if (savedSize) {
                    const computedSize = this.computeSize();
                    this.size = [
                        Math.max(savedSize[0], computedSize[0]),  // Use saved width if larger
                        Math.max(savedSize[1], computedSize[1])   // Use saved height if larger
                    ];
                }
            };

            // Serialize widgets properly
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) {
                    onSerialize.apply(this, arguments);
                }

                o.widgets_values = [];

                this.widgets.forEach(w => {
                    if (w instanceof PMLoraStackerWidget) {
                        o.widgets_values.push(w.serializeValue());
                    } else if (w.name === "layer_filter") {
                        o.widgets_values.push(w.value);
                    }
                });
            };
        }
    }
});
