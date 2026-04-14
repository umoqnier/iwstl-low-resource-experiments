import os
import re
import click

# Force PyTorch to allow full object loading (Must be before torch imports)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import random
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

OUTPUT_PATH_FMT = "krusty_crab.st.constrained.{LABEL}.arn-spa.txt"


def preprocess_text(text: str) -> str:
    """Removes punctuation and converts to lowercase."""
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()


def run_predictions(
    model, file_paths: list, batch_size: int, source_lang: str, target_lang: str
) -> list:
    """Runs inference on the provided audio files."""
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.to(torch.bfloat16)

    preds = model.transcribe(
        file_paths,
        pnc="no",
        task="ast",
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=batch_size,
    )
    return [preprocess_text(p.text) for p in preds]


def get_file_paths(manifest_path: str, max_samples: int = None):
    """Loads audio paths from the manifest."""
    data = read_manifest(manifest_path)
    if max_samples is not None:
        data = data[:max_samples]

    return [x["audio_filepath"] for x in data]


@click.command()
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="Path to the JSON manifest for evaluation.",
)
@click.option(
    "--adapter-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to the trained adapter .pt file. Leave empty to test base model.",
)
@click.option(
    "--base-model", default="nvidia/canary-1b-v2", help="HuggingFace model name."
)
@click.option(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size for inference. Reduce if you hit OOM.",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Maximum number of samples to evaluate (useful for quick tests).",
)
@click.option(
    "--label",
    default="primary",
    type=click.Choice(["primary", "contrastive1", "contrastive2"]),
    help="Label for the evaluation.",
)
@click.option(
    "--print-examples",
    default=5,
    type=int,
    help="Number of random prediction examples to print.",
)
def main(
    manifest, adapter_path, base_model, batch_size, max_samples, label, print_examples
):
    """ """
    click.echo(f"Loading base model: {base_model}")
    model = nemo_asr.models.ASRModel.from_pretrained(base_model)

    # Load adapters only if a path is provided
    if adapter_path:
        click.echo("Configuring architecture for adapters...")
        model.replace_adapter_compatible_modules()
        click.echo(f"Loading adapters from: {adapter_path}...")
        model.load_adapters(adapter_path)
    else:
        click.echo("No adapter path provided. Evaluating base model only.")

    model.eval()

    # Load Data
    click.echo(f"Loading tests paths from: {manifest}...")
    file_paths = get_file_paths(manifest, max_samples=max_samples)

    # Transcribe
    click.echo(
        f"Running predictions on {len(file_paths)} samples (Batch Size: {batch_size})..."
    )
    preds = run_predictions(model, file_paths, batch_size, "en", "es")

    # Writing results
    output_path = OUTPUT_PATH_FMT.format(LABEL=label)
    click.echo(f"Writing results to: {output_path}...")
    with open(output_path, "w") as f:
        f.write("\n".join(preds))
    click.echo("Done")

    # Random Examples
    if print_examples > 0 and len(preds) > 0:
        click.echo(f"🔍 Printing {min(print_examples, len(preds))} random examples:\n")
        random.shuffle(preds)

        for i, pred in enumerate(preds[:print_examples]):
            click.echo(f"--- Example {i + 1} ---")
            click.echo(f"Pred: {pred}\n")

if __name__ == "__main__":
    main()