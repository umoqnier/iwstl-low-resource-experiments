#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

def process_text(text) -> str:
    """Lowercase and remove punctuation from text."""
    punctuation = r'[.,;:!?¿¡"\'()\[\]{}<>]'
    return re.sub(punctuation, "", text.lower())


def process_json(input_file: str) -> None:
    """Process JSON file to filter by duration and modify text/pnc fields."""
    skiped_lines = [] 
    output_file = Path(input_file).with_stem(f"{Path(input_file).stem}_processed")

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # Process text: lowercase and remove punctuation
                record["text"] = process_text(record["text"])

                # Change pnc to "No"
                record["pnc"] = "No"

                # Skip records with duration > 15
                if record["duration"] > 15:
                    skiped_lines.append(record)
                    continue

                # Write modified record
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")

            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue
        
        # Save skiped lines to a separate file
        with open(Path(input_file).with_stem(f"{Path(input_file).stem}_skiped_lines"), "w", encoding="utf-8") as skiped_file:
            json.dump(skiped_lines, skiped_file, ensure_ascii=False, indent=2)

    print(f"Processing complete. Output saved to: {output_file}. Skiped lines={len(skiped_lines)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_json_file>")
        sys.exit(1)

    process_json(sys.argv[1])
