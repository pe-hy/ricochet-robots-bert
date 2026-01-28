#!/usr/bin/env python3
"""Convert JSON dataset to JSONL format for efficient lazy loading."""

import json
import sys
from pathlib import Path

def convert_json_to_jsonl(input_path: str, output_path: str):
    """
    Convert JSON dataset to JSONL format.

    Input format: {"metadata": {...}, "examples": [{...}, {...}, ...]}
    Output format:
        Line 1: {"metadata": {...}}
        Line 2: {...}  # First example
        Line 3: {...}  # Second example
        ...
    """
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data['examples'])} examples")
    print(f"Writing to {output_path}...")

    with open(output_path, 'w') as f:
        # Write metadata as first line
        json.dump(data.get('metadata', {}), f)
        f.write('\n')

        # Write each example as a separate line
        for i, example in enumerate(data['examples']):
            json.dump(example, f)
            f.write('\n')

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} examples...")

    print(f"Done! Wrote {len(data['examples'])} examples to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_jsonl.py <input.json> <output.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_json_to_jsonl(input_path, output_path)
