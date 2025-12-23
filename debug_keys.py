#!/usr/bin/env python3
"""
Debug script to analyze LoRA layer keys from a Wan 2.2 model.
Run this to see what layer keys are actually present in your LoRA.
"""

import sys
import safetensors.torch

def analyze_lora_keys(lora_path):
    """Load and analyze the layer keys in a LoRA file."""
    print(f"\nAnalyzing LoRA: {lora_path}\n")

    try:
        # Load the LoRA
        if lora_path.endswith('.safetensors'):
            lora_dict = safetensors.torch.load_file(lora_path)
        else:
            import torch
            lora_dict = torch.load(lora_path, map_location='cpu')

        keys = list(lora_dict.keys())
        print(f"Total keys: {len(keys)}\n")

        # Analyze key patterns
        print("Sample keys (first 20):")
        for key in keys[:20]:
            print(f"  {key}")

        print("\n" + "="*80)

        # Extract unique layer components
        components = set()
        for key in keys:
            parts = key.split('.')
            for part in parts:
                if part not in ['lora_up', 'lora_down', 'alpha', 'weight', 'bias']:
                    components.add(part)

        print("\nUnique layer components found:")
        for comp in sorted(components):
            print(f"  {comp}")

        print("\n" + "="*80)

        # Check for common patterns
        patterns = {
            'attention': ['attn', 'attention', 'self_attn', 'cross_attn'],
            'mlp/feedforward': ['mlp', 'ff', 'feed_forward', 'ffn', 'fc'],
            'projection': ['proj', 'projection'],
            'norm': ['norm', 'ln', 'layer_norm'],
            'embedding': ['emb', 'embedding', 'token'],
        }

        print("\nPattern matching analysis:")
        for category, pattern_list in patterns.items():
            matching_keys = []
            for key in keys:
                if any(pattern in key.lower() for pattern in pattern_list):
                    matching_keys.append(key)

            if matching_keys:
                print(f"\n{category.upper()} ({len(matching_keys)} keys):")
                for key in matching_keys[:5]:
                    print(f"  {key}")
                if len(matching_keys) > 5:
                    print(f"  ... and {len(matching_keys) - 5} more")

        print("\n" + "="*80)

        # Try to identify architecture
        print("\nArchitecture detection:")
        key_str = ' '.join(keys).lower()

        if 'double_blocks' in key_str or 'single_blocks' in key_str:
            print("  -> Likely a Flux/DiT architecture (double_blocks/single_blocks)")
        elif 'joint_blocks' in key_str:
            print("  -> Likely a SD3/MMDiT architecture (joint_blocks)")
        elif 'input_blocks' in key_str or 'output_blocks' in key_str:
            print("  -> Likely a SD1.5/SDXL architecture (UNet)")
        else:
            print("  -> Unknown architecture")

    except Exception as e:
        print(f"Error loading LoRA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_keys.py <path_to_lora.safetensors>")
        print("\nExample:")
        print("  python debug_keys.py /path/to/your/wan2.2_lora.safetensors")
        sys.exit(1)

    analyze_lora_keys(sys.argv[1])
