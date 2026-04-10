import itertools
import logging
import os

import click
import datasets
import lightning.pytorch as L
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
import tqdm
import yaml
from datasets import load_dataset
from huggingface_hub import login
from huggingface_hub import utils as hf_utils
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.asr.parts.utils.manifest_utils import (
    write_manifest,
)
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.prompt_fn import (
    get_prompt_format_fn,
)
from nemo.collections.common.parts import LinearAdapterConfig
from nemo.collections.common.prompts import PromptFormatter
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.data import Dataset

from custom_aumentation import augment

# Set up rich logging
console = Console()

file_handler = logging.FileHandler("training_logs.log")
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console), file_handler],
)
logger = logging.getLogger(__name__)

# 1. Disable the datasets download progress bars
datasets.disable_progress_bars()

# 2. Mute datasets info/warning logs (only show critical errors)
datasets.utils.logging.set_verbosity_error()

# 3. Mute huggingface_hub info logs (like connection messages)
hf_utils.logging.set_verbosity_error()

# Mute noisy third-party networking loggers
noisy_loggers = ["httpx", "httpcore", "urllib3", "fsspec", "filelock"]
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

class MyCanaryPromptedAudioToTextLhotseDataset(Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``Cut`` a single MonoCut or MixedCut
    * a ``PromptFormatter`` Prepend and append control tokens to the token sequence

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """

    def __init__(self, tokenizer: "TokenizerSpec", prompt: PromptFormatter):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.prompt_format_fn = get_prompt_format_fn(
            Cut, self.prompt
        )  # Use the default canary prompt function

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)
        audio_np = audio.numpy()
        
        augmented = [augment(samples=sample, sample_rate=16000) for sample in audio_np]
        
        audio = torch.from_numpy(np.stack(augmented))
        answers = []
        prompts = []
        prompts_with_answers = []

        for cut in cuts:
            prompted_answers = self.prompt_format_fn(cut, self.prompt)
            answers.append(prompted_answers["answer_ids"])
            prompts.append(prompted_answers["context_ids"])
            prompts_with_answers.append(prompted_answers["input_ids"])

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(
            prompts_with_answers
        )
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            cuts=cuts.drop_in_memory_data(),
        )

    def _collate_tokens(
        self, tokens: list[list[int] | torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens



class CanaryMultilingualDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        prompt_formatter,
        map_dataset_name: str = "mengct00/Mapudungun_iwslt26",
        que_data_dir: str = "./data/que_data/",
        out_data_dir: str = "./combined_data/",
        batch_size: int = 8,
        streaming: bool = False,
        max_examples: int | None = None,  # Applied per language
        language_mode: str = "multi",  # Accepts: "map", "que", "multi"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter

        self.map_dataset_name = map_dataset_name
        self.que_data_dir = que_data_dir
        self.out_data_dir = out_data_dir
        self.batch_size = batch_size
        self.streaming = streaming
        self.max_examples = max_examples

        if language_mode not in ["map", "que", "multi"]:
            raise ValueError("language_mode must be 'map', 'que', or 'multi'")
        self.language_mode = language_mode

        # AST manifests paths
        self.train_manifest = os.path.join(
            out_data_dir, f"train_{self.language_mode}_manifest.json"
        )
        self.val_manifest = os.path.join(
            out_data_dir, f"val_{self.language_mode}_manifest.json"
        )
        self.test_manifest = os.path.join(
            out_data_dir, f"test_{self.language_mode}_manifest.json"
        )

    def setup(self, stage=None):
        pass

    def _setup_dataloader(self, config):
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1

        return get_lhotse_dataloader_from_config(
            OmegaConf.create(config),
            global_rank=rank,
            world_size=world_size,
            dataset=MyCanaryPromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer, prompt=self.prompt_formatter
            ),
        )

    def train_dataloader(self):
        return self._setup_dataloader(
            {
                "manifest_filepath": self.train_manifest,
                "num_buckets": 30,
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": True,
                "persistent_workers": True,
                "pin_memory": True,  # speeds up CPU-to-GPU transfer
            }
        )

    def val_dataloader(self):
        return self._setup_dataloader(
            {
                "manifest_filepath": self.val_manifest,
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": False,
                "persistent_workers": True,
                "pin_memory": True,
            }
        )

    def test_dataloader(self):
        # Optional: handle cases where test split might not exist
        if not os.path.exists(self.test_manifest):
            logger.info(
                "Warning: No test manifest found, returning val_dataloader for test."
            )
            return self.val_dataloader()

        return self._setup_dataloader(
            {
                "manifest_filepath": self.test_manifest,
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": False,
            }
        )

    def _process_mapudungun(self, split: str) -> list:
        logger.info(f"Processing Mapudungun split: {split}...")
        dataset = load_dataset(
            self.map_dataset_name, split=split, streaming=self.streaming
        )
        wav_dir = os.path.join(self.out_data_dir, "map_wavs")
        os.makedirs(wav_dir, exist_ok=True)

        if self.max_examples is not None:
            dataset = (
                dataset.take(self.max_examples)
                if self.streaming
                else dataset.select(range(min(len(dataset), self.max_examples)))
            )

        entries = []
        try:
            total = self.max_examples if self.streaming else len(dataset)
        except TypeError:
            total = self.max_examples

        for idx, item in tqdm.tqdm(
            enumerate(dataset), total=total, desc=f"Mapudungun {split}"
        ):
            audio_info = item.get("audio")
            if not audio_info:
                continue

            filepath = os.path.join(wav_dir, f"map_{split}_{idx}.wav")
            # Only write if it doesn't exist to speed up re-runs
            if not os.path.exists(filepath):
                sf.write(filepath, audio_info["array"], audio_info["sampling_rate"])
            duration = len(audio_info["array"]) / audio_info["sampling_rate"]

            text = item.get("spa", "")
            if not text:
                for k, v in item.items():
                    if (
                        isinstance(k, str)
                        and (k.endswith("_es") or k.endswith("_spa"))
                        and isinstance(v, str)
                    ):
                        text = v
                        break

            if text:
                entries.append(
                    {
                        "audio_filepath": os.path.abspath(filepath),
                        "duration": duration,
                        "text": text,
                        "pnc": "no",
                        "source_lang": "en",  # Placeholder for Mapudungun
                        "target_lang": "es",
                    }
                )
        return entries

    def _process_quechua(self, split: str) -> list:
        logger.info(f"Processing Quechua split: {split}...")

        # Determine which subdirectories map to this split
        subdirs = []
        if split == "train":
            subdirs = [
                "que_spa_unconstrained/train",
                "que_spa_synthetic_translation/train",
            ]
        elif split == "validation":
            subdirs = ["que_spa_unconstrained/valid"]
        else:
            return []  # No test split defined locally for Quechua

        entries = []
        for subdir in subdirs:
            split_dir = os.path.join(self.que_data_dir, subdir)
            if not os.path.exists(split_dir):
                logger.info(
                    f"Warning: Quechua directory {split_dir} not found. Skipping."
                )
                continue

            # Figure out local split name ('train' or 'valid') for the .yaml/.spa files
            local_split = os.path.basename(subdir)

            yaml_path = os.path.join(split_dir, "txt", f"{local_split}.yaml")
            text_path = os.path.join(split_dir, "txt", f"{local_split}.spa")
            wav_dir = os.path.join(split_dir, "wav")

            if not os.path.exists(yaml_path) or not os.path.exists(text_path):
                logger.info(f"Warning: Missing metadata/text files in {split_dir}")
                continue

            with open(yaml_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            with open(text_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f.readlines()]

            for i, meta in tqdm.tqdm(
                enumerate(metadata), total=len(metadata), desc=f"Quechua {subdir}"
            ):
                # Enforce total max_examples across ALL subdirectories for Quechua
                if self.max_examples and len(entries) >= self.max_examples:
                    break

                wav_file = meta["wav"]
                duration = meta["duration"]
                text = texts[i]
                audio_path = os.path.abspath(os.path.join(wav_dir, wav_file))

                if os.path.exists(audio_path):
                    entries.append(
                        {
                            "audio_filepath": audio_path,
                            "duration": duration,
                            "text": text,
                            "pnc": "no",
                            "source_lang": "en",  # Placeholder
                            "target_lang": "es",
                        }
                    )
                else:
                    logger.info(f"Warning: Audio file not found: {audio_path}")

            # Break outer loop if limit is reached
            if self.max_examples and len(entries) >= self.max_examples:
                break

        return entries

    def prepare_data(self):
        if not os.path.exists(self.out_data_dir):
            os.makedirs(self.out_data_dir)

        if os.path.exists(self.train_manifest) and os.path.exists(self.val_manifest):
            logger.info(
                f"Manifests for mode '{self.language_mode}' found. Skipping data preparation."
            )
            return

        # TODO: Add test when available
        splits = ["train", "validation"]

        for split in splits:
            logger.info(f"\n--- Preparing {split} split ---")

            map_entries = []
            que_entries = []

            # Extract based on language_mode flag
            if self.language_mode in ["map", "multi"]:
                map_entries = self._process_mapudungun(split)
            if self.language_mode in ["que", "multi"]:
                que_entries = self._process_quechua(split)

            # Interleave entries: 1 Mapudungun, 1 Quechua, 1 Mapudungun...
            combined_entries = []
            for map_entry, que_entry in itertools.zip_longest(map_entries, que_entries):
                if map_entry is not None:
                    combined_entries.append(map_entry)
                if que_entry is not None:
                    combined_entries.append(que_entry)

            if combined_entries:
                # Set path based on split name mapping
                if split == "train":
                    path = self.train_manifest
                elif split == "validation":
                    path = self.val_manifest
                else:
                    path = self.test_manifest

                write_manifest(path, combined_entries)
                logger.info(
                    f"Created manifest with {len(combined_entries)} total examples ({len(map_entries)} MAP, {len(que_entries)} QUE): {path}"
                )
            else:
                logger.info(
                    f"No data found for split {split}. Skipping manifest creation."
                )

        logger.info("Data preparation complete.")


@click.command()
@click.option(
    "--language-mode",
    type=click.Choice(["map", "que", "multi"], case_sensitive=False),
    default="map",
    help="Language mode: 'map' for Mapudungun, 'que' for Quechua, 'multi' for both",
)
@click.option(
    "--model",
    type=click.Choice(
        ["nvidia/canary-1b-flash", "nvidia/canary-1b-v2"], case_sensitive=False
    ),
    default=None,
    help="Model to use. If not specified, will auto-select based on CUDA availability",
)
@click.option(
    "--devices",
    type=int,
    default=1,
    help="Number of GPU devices to be used. Default = 1",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for training",
)
@click.option(
    "--max-examples",
    type=int,
    default=None,
    help="Maximum number of examples to use per language (None for all)",
)
@click.option(
    "--max-epochs",
    type=int,
    default=50,
    help="Maximum training epochs",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    help="Use streaming mode for dataset loading",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./models",
    help="Directory to save adapter models",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="./combined_data",
    help="Directory for combined data manifests",
)
@click.option(
    "--que-data-dir",
    type=click.Path(),
    default="./data/que_data/",
    help="Directory for Quechua data",
)
@click.option(
    "--map-dataset-id",
    type=str,
    default="mengct00/Mapudungun_iwslt26",
    help="HF mapudungun dataset id",
)
def main(
    language_mode,
    model,
    devices,
    batch_size,
    max_examples,
    max_epochs,
    streaming,
    output_dir,
    data_dir,
    que_data_dir,
    map_dataset_id,
):
    """Train Canary model adapters for Mapudungun and/or Quechua languages."""
    logger.info("Starting Canary model adapter training")

    # Model selection
    if model is None:
        logger.info(f"Is CUDA available? {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            model = "nvidia/canary-1b-flash"
            logger.warning("CUDA not available, using flash model")
        else:
            model = "nvidia/canary-1b-v2"
            logger.info("CUDA available, using v2 model")

    logger.info(f"Using model: {model}")
    logger.info(f"Language mode: {language_mode}")
    logger.info(f"Batch size: {batch_size}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs detected: {gpu_count}. Using {devices}")

    for i in range(gpu_count):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model
    logger.info("Loading model...")
    model = nemo_asr.models.ASRModel.from_pretrained(model)

    # Habilitando adapters
    logger.info("Configuring adapters...")
    model.replace_adapter_compatible_modules()
    input_dim = model.cfg.encoder.d_model
    adapter_dim = 8
    enc_adapter_cfg = LinearAdapterConfig(in_features=input_dim, dim=adapter_dim)

    # queremos solo habilitar el encoder
    model.add_adapter(name="encoder:enc", cfg=enc_adapter_cfg)
    model.freeze()
    model.unfreeze_enabled_adapters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({(trainable_params / total_params) * 100:.3f}%)"
    )

    # HF login
    logger.info("Logging into HuggingFace Hub...")
    login()

    # Setup data module
    logger.info("Setting up data module...")
    data_loader = CanaryMultilingualDataModule(
        tokenizer=model.tokenizer,
        prompt_formatter=model.prompt,
        map_dataset_name=map_dataset_id,
        que_data_dir=que_data_dir,
        out_data_dir=data_dir,
        streaming=streaming,
        max_examples=max_examples,
        batch_size=batch_size,
        language_mode=language_mode,
    )

    # Prepare data
    logger.info("Preparing data...")
    data_loader.prepare_data()

    # Setup optimization
    logger.info("Configuring optimization...")
    model.cfg.optim.lr = 3e-4
    model.cfg.optim.sched.warmup_steps = 25

    # TODO: Dropout

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate output model name based on parameters
    model_name_parts = [
        "canary",
        language_mode,
        f"bs{batch_size}",
        f"epoch{max_epochs}",
    ]
    if max_examples:
        model_name_parts.append(f"max{max_examples}")
    adapter_model_name = "_".join(model_name_parts) + ".pt"
    adapter_model_path = os.path.join(output_dir, adapter_model_name)

    logger.info(f"Training will save adapter to: {adapter_model_path}")

    # Setup Checkpointing
    logger.info("Configuring checkpoints...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="canary_adapter_{epoch:02d}",
        every_n_epochs=3,
        save_top_k=10,  # Keeps only the 10 most recent/best checkpoints
        monitor="step",  # Monitors the training step to determine the "top 10"
        mode="max",
        save_last=True,  # Always saves a 'last.ckpt' to easily resume from crashes
    )

    # 1. Setup Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # The metric PyTorch Lightning logs during validation
        min_delta=0.00,  # Minimum change to qualify as an improvement
        patience=5,  # How many validation checks to wait before stopping
        verbose=True,
        mode="min",  # We want the loss to minimize
    )

    # This will create a directory called 'tb_logs' in your project root
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs", 
        name=f"canary_mapudungun_bs{batch_size}_epochs{max_epochs}"
    )
    # 2. Setup Learning Rate Monitor (Optional but very useful)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = L.Trainer(
        devices=devices,
        accelerator="gpu",
        strategy="ddp",
        max_epochs=max_epochs,
        # Drastically reduces memory by using 16-bit floats for activations
        precision="bf16-mixed",  # Mixed Precision
        # If we cut batch_size to 4, accumulating 4 batches gives an effective batch of 16
        accumulate_grad_batches=4,
        logger=tb_logger,
        enable_checkpointing=True,  # CRITICAL: Must be True to save checkpoints
        check_val_every_n_epoch=2,
        use_distributed_sampler=False,  # Keeps Lhotse distributed sampling happy
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    # Detect if a previous checkpoint exists
    last_ckpt_path = os.path.join(output_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        logger.info(
            f"Found existing checkpoint. Resuming training from: {last_ckpt_path}"
        )
        resume_ckpt = last_ckpt_path
    else:
        logger.info("No existing checkpoint found. Starting training from scratch.")
        resume_ckpt = None

    # Train (Pass the resume_ckpt path here)
    logger.info("Starting training...")
    trainer.fit(model, data_loader, ckpt_path=resume_ckpt)

    # Save final adapter
    logger.info(f"Saving final adapter to {adapter_model_path}...")
    model.save_adapters(adapter_model_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()