import json
import os
import random
import re

import click

# Force PyTorch to allow full object loading (Must be before torch imports)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from bert_score import score as bert_score_fn
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from torchmetrics.text import CHRFScore, SacreBLEUScore


def preprocess_text(text: str) -> str:
    """Removes punctuation and converts to lowercase."""
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()


def get_eval_data(manifest_path: str, max_samples: int = None):
    """Loads audio paths and ground truth texts from the manifest."""
    data = read_manifest(manifest_path)
    if max_samples is not None:
        data = data[:max_samples]

    file_paths = [x["audio_filepath"] for x in data]
    gt_texts = [x["text"] for x in data]
    return file_paths, gt_texts


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


def calculate_bertscore(
    predictions: list,
    gt_texts: list,
    model_type: str = "xlm-roberta-large",
    device: str | None = None,
    batch_size: int = 32,
) -> dict:
    """Calculates BERTScore (Precision, Recall, F1).

    Defaults to xlm-roberta-large which is multilingual — good fit for
    Spanish (es) targets. Pass device='cpu' to avoid OOM if Canary is on GPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score_fn(
        cands=predictions,
        refs=gt_texts,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def eval_model(predictions: list, gt_texts: list) -> dict:
    """Calculates WER and SacreBLEU scores."""
    wer = word_error_rate(predictions, gt_texts)

    sacrebleu = SacreBLEUScore(n_gram=4)
    # Torchmetrics SacreBLEU expects target to be a sequence of sequences of strings
    formatted_gt = [[gt] for gt in gt_texts]
    sacrebleu.update(predictions, formatted_gt)
    bleu = sacrebleu.compute()
    # Extract tensor value if it's a tensor
    bleu_val = bleu.item() if hasattr(bleu, "item") else bleu

    # chrF2++
    chrf2pp = CHRFScore(n_char_order=6, n_word_order=2, beta=2)
    chrf2pp.update(predictions, formatted_gt)
    chrf2pp_val = chrf2pp.compute().item()

    # Bertscore
    bert_scores = calculate_bertscore(predictions, gt_texts)

    return {"wer": wer, "bleu": bleu_val, "chrf2++": chrf2pp_val, **bert_scores}


def run_evaluation(model, file_paths, gt_texts, batch_size):
    """Runs predictions and calculates metrics for a given model."""
    preds = run_predictions(model, file_paths, batch_size, "en", "es")
    metrics = eval_model(preds, gt_texts)
    return metrics, preds


def save_results_json(
    results: dict,
    output_path: str | Path,
    config: dict | None = None,
) -> Path:
    """Saves evaluation results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metrics": results,
        "config": config or {},
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path


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
    "--print-examples",
    default=5,
    type=int,
    help="Number of random prediction examples to print.",
)
@click.option(
    "--output-json",
    type=click.Path(),
    default=None,
    help="Optional path to save the metrics as JSON (e.g. results/run_01.json).",
)
@click.option(
    "--baseline",
    is_flag=True,
    default=False,
    help="Indicate if this run is for calculating baseline metrics from vanilla model.",
)
def main(
    manifest,
    adapter_path,
    base_model,
    batch_size,
    max_samples,
    print_examples,
    output_json,
    baseline,
):
    """Evaluates a Canary AST model with or without adapters."""

    click.echo(f"Loading base model: {base_model}...")
    model = nemo_asr.models.ASRModel.from_pretrained(base_model)
    model.eval()

    # Load Data
    click.echo(f"Loading evaluation data from: {manifest}...")
    file_paths, gt_texts = get_eval_data(manifest, max_samples=max_samples)

    baseline_metrics = None
    baseline_preds = None
    adapted_metrics = None
    adapted_preds = None

    # 1. Baseline Evaluation
    if baseline:
        click.echo("\n--- Phase 1: Baseline Evaluation (Vanilla Model) ---")
        baseline_metrics, baseline_preds = run_evaluation(
            model, file_paths, gt_texts, batch_size
        )
        click.echo("Baseline metrics calculated.")

    # 2. Adaptation / Main Evaluation
    click.echo("\n--- Phase 2: Main Evaluation ---")
    if adapter_path and not (baseline and not adapter_path):
        # If we are in baseline mode, we only load adapters if adapter_path is actually provided
        # If we are NOT in baseline mode, we load adapters if provided
        click.echo("Configuring architecture for adapters...")
        model.replace_adapter_compatible_modules()
        click.echo(f"Loading adapters from: {adapter_path}...")
        model.load_adapters(adapter_path)
    elif not baseline:
        click.echo("No adapter path provided. Evaluating base model only.")
    elif baseline and not adapter_path:
        click.echo(
            "Baseline mode active but no adapter path provided. Skipping adapted evaluation."
        )

    adapted_metrics, adapted_preds = run_evaluation(
        model, file_paths, gt_texts, batch_size
    )
    click.echo("Main evaluation metrics calculated.")

    # If baseline was requested but no adapter provided, adapted is same as baseline
    if baseline and not adapter_path:
        adapted_metrics = baseline_metrics
        adapted_preds = baseline_preds

    # Save JSON results
    if output_json:
        output_path = Path(output_json)

        if baseline:
            # Save baseline
            baseline_config = {
                "base_model": base_model,
                "adapter_path": None,
                "baseline": True,
                "manifest": manifest,
                "batch_size": batch_size,
                "max_samples": max_samples,
                "source_lang": "en",
                "target_lang": "es",
                "num_samples": len(baseline_preds) if baseline_preds else 0,
            }
            baseline_file = output_path.with_name(
                f"{output_path.stem}_baseline{output_path.suffix}"
            )
            save_results_json(baseline_metrics, baseline_file, config=baseline_config)
            click.echo(f"💾 Baseline results saved to: {baseline_file}")

            # Save adapted
            adapted_config = {
                "base_model": base_model,
                "adapter_path": adapter_path,
                "baseline": False,
                "manifest": manifest,
                "batch_size": batch_size,
                "max_samples": max_samples,
                "source_lang": "en",
                "target_lang": "es",
                "num_samples": len(adapted_preds) if adapted_preds else 0,
            }
            adapted_file = output_path.with_name(
                f"{output_path.stem}_adapted{output_path.suffix}"
            )
            save_results_json(adapted_metrics, adapted_file, config=adapted_config)
            click.echo(f"💾 Adapted results saved to: {adapted_file}")
        else:
            # Single run
            config = {
                "base_model": base_model,
                "adapter_path": adapter_path,
                "baseline": False,
                "manifest": manifest,
                "batch_size": batch_size,
                "max_samples": max_samples,
                "source_lang": "en",
                "target_lang": "es",
                "num_samples": len(adapted_preds),
            }
            written = save_results_json(adapted_metrics, output_json, config=config)
            click.echo(f"💾 Results saved to: {written}")

    # Print Results
    def print_metrics(label, metrics):
        if metrics is None:
            return
        click.echo(f"\n--- {label} RESULTS ---")
        click.echo(f" WER (Word Error Rate): {metrics['wer']:.4f}")
        click.echo(f" SacreBLEU Score:       {metrics['bleu']:.4f}")
        click.echo(f" chrF2++ Score:         {metrics['chrf2++']:.4f}")
        click.echo(f" BERTScore F1:          {metrics['bertscore_f1']:.4f}")
        click.echo(
            f" BERTScore P / R:       {metrics['bertscore_precision']:.4f} / {metrics['bertscore_recall']:.4f}"
        )

    print_metrics("BASELINE", baseline_metrics)
    print_metrics("ADAPTED", adapted_metrics)
    click.echo("\n" + "=" * 40)

    # Random Examples (for adapted model)
    if print_examples > 0 and adapted_preds and len(adapted_preds) > 0:
        click.echo(
            f"🔍 Printing {min(print_examples, len(adapted_preds))} random examples (Adapted Model):\n"
        )
        examples = list(zip(gt_texts, adapted_preds))
        random.shuffle(examples)
        for i, (gt, pred) in enumerate(examples[:print_examples]):
            click.echo(f"--- Example {i + 1} ---")
            click.echo(f"GT:   {gt}")
            click.echo(f"Pred: {pred}\n")

    click.echo("=" * 40 + "\n")


if __name__ == "__main__":
    main()
