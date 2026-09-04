import os
from pathlib import Path

import click
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from data.processors import MapugungunProcessor, NahuatlProcessor, QuechuaProcessor
from utils.configs import MAPUCHE_ID, NAHUATL_PATH, QUECHUA_PATH
from utils.logging_utils import get_logger

# Initialize Rich console and logger
console = Console()
logger = get_logger(__name__)

PROCESSOR_MAP = {
    "azz": {
        "class": NahuatlProcessor,
        "name": "Nahuatl",
        "out_manifests_subdir": "nahuatl",
        "kwargs": {"data_dir": NAHUATL_PATH},
    },
    "map": {
        "class": MapugungunProcessor,
        "name": "Mapugungun",
        "out_manifests_subdir": "mapugungun",
        "kwargs": {"dataset_id": MAPUCHE_ID, "text_column": "spa"},
    },
    "que": {
        "class": QuechuaProcessor,
        "name": "Quechua",
        "out_manifests_subdir": "quechua",
        "kwargs": {"data_dir": QUECHUA_PATH, "split_mapping": {}},  # Add mapping here
    },
}


@click.command()
@click.option(
    "--out",
    "-o",
    default="manifests",
    type=click.Path(),
    help="Output directory for manifests and audio chunks.",
)
@click.option(
    "--task",
    "-t",
    type=click.Choice(["asr", "ast", "both"], case_sensitive=False),
    default="ast",
    help="Task type: 'asr' (transcriptions only), 'ast' (translations), or 'both'.",
)
@click.option(
    "--max-examples",
    "-m",
    type=int,
    default=10,
    help="Limit the number of examples per split for debugging.",
)
@click.option(
    "--language-mode",
    type=click.Choice(["map", "que", "azz", "multi"]),
    default="azz",
    help="Language mode: 'azz' (Nahuatl), 'map', 'que', or 'multi' (all).",
)
@click.option(
    "--streaming", is_flag=True, help="Enable streaming for HF datasets (prototyping)."
)
def build(out, task, max_examples, language_mode, streaming):
    """
    Build datasets for Canary experiments.
    """
    # Dynamic Panel Title based on language mode
    title = (
        "Multilingual Dataset Builder"
        if language_mode == "multi"
        else f"{PROCESSOR_MAP[language_mode]['name']} Dataset Builder"
    )

    console.print(
        Panel(
            f"[bold blue]{title}[/bold blue]\n[grey]Mode: {language_mode} | Task: {task} | Output: {out}",
            expand=False,
        )
    )
    try:
        # 1. Determine which processors to run
        selected_langs = (
            list(PROCESSOR_MAP.keys()) if language_mode == "multi" else [language_mode]
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            for lang_code in selected_langs:
                config = PROCESSOR_MAP[lang_code]

                manifest_paths = {
                    "train": out / Path(config["name"]) / Path("train_manifest.json"),
                    "validation": out
                    / Path(config["name"])
                    / Path("val_manifest.json"),
                    "test": out / Path(config["name"]) / Path("test_manifest.json"),
                }

                processor_args = {
                    "name": config["name"],
                    "out_dir": Path(out) / Path(config["out_manifests_subdir"]),
                    "max_examples": max_examples,
                    **config["kwargs"],
                }

                # 2. Inject streaming flag only if the processor is Mapugungun (or supports streaming)
                if config["class"] == MapugungunProcessor:
                    processor_args["streaming"] = streaming

                console.print(f"[yellow]⚙️ Processing {config['name']}...[/yellow]")
                processor = config["class"](**processor_args)

                progress_task = progress.add_task(
                    description=f"Building {config['name']}...", total=None
                )
                lang_entries = processor.process(task=task)

                logger.info(f"Writing manifest for {config['name']}")

                for split, path in manifest_paths.items():
                    write_manifest(path, lang_entries[split])
                    console.print(
                        f"\n[bold green]✅ Success![/bold green] Manifests written to [bold]{path}[/bold]"
                    )

                display_summary(lang_entries)

                progress.remove_task(progress_task)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error occurred during build:[/bold red] {e}")
        logger.exception("Build failed")
        raise click.Abort()


def display_summary(entries):
    """Displays a Rich table with the stats of the built dataset."""
    table = Table(
        title="Dataset Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Split", style="dim", width=12)
    table.add_column("Samples", justify="right")
    table.add_column("Duration (Hrs)", justify="right")

    for split in ["train", "validation", "test"]:
        samples = entries[split]
        total_hours = sum(s["duration"] for s in samples) / 3600.0
        table.add_row(split, str(len(samples)), f"{total_hours:.2f}")

    console.print(table)


if __name__ == "__main__":
    build()
