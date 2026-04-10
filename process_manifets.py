#!/usr/bin/env python3
import json
import re
from pathlib import Path
import click


def process_text(text) -> str:
    """Lowercase and remove punctuation from text."""
    punctuation = r'[.,;:!?¿¡"\'()\[\]{}<>]'
    return re.sub(punctuation, "", text.lower())


def write_jsonl_file(lines: list[dict], output_path: Path):
    """Write a list of dictionaries to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as outfile:
        for line in lines:
            json_record = json.dumps(line, ensure_ascii=False)
            outfile.write(json_record + "\n")


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--filter-long/--no-filter-long",
    is_flag=True,
    default=False,
    help="Filter out audio files longer than 15 seconds",
)
@click.option(
    "--max-duration",
    type=int,
    default=15,
    help="Maximum duration in seconds for audio files to be included",
)
def process_json(input_file: str, filter_long: bool, max_duration: int) -> None:
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
                if filter_long and record["duration"] > max_duration:
                    skiped_lines.append(record)
                    continue

                # Write modified record
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")

            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                continue

        # Save skiped lines to a separate file
        if filter_long and skiped_lines:
            write_jsonl_file(
                skiped_lines,
                Path(input_file).with_stem(f"{Path(input_file).stem}_skiped_lines"),
            )

    print(
        f"Processing complete. Output saved to: {output_file}. Skiped lines={len(skiped_lines)}"
    )


if __name__ == "__main__":
    process_json()
