#!/usr/bin/env python3
"""
Convert JSON dataset to JSONL format for memory-efficient loading.
Usage: python convert_to_jsonl.py input.json output.jsonl
"""
import json
import sys
from pathlib import Path


def convert_to_jsonl(input_path: str, output_path: str):
    """Convert JSON dataset to JSONL format"""
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    metadata = data.get('metadata', {'num_examples': len(examples)})
    print(f"Found {len(examples)} examples")

    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        # Write metadata as first line
        f.write(json.dumps({'metadata': metadata}) + '\n')

        # Write examples
        for i, example in enumerate(examples):
            f.write(json.dumps(example) + '\n')
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(examples)} examples")

    print(f"Done! Created {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_jsonl.py input.json output.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    convert_to_jsonl(input_path, output_path)
