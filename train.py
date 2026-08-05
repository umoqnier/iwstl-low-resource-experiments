import os

import click
import lightning.pytorch as L
import nemo.collections.asr as nemo_asr
import torch
from huggingface_hub import login as hf_login
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from nemo.collections.common.parts import LinearAdapterConfig

from data.datamodule import CanaryMultilingualDataModule
from data.processors import MapugungunProcessor, NahuatlProcessor, QuechuaProcessor
from utils.configs import (
    CANARY_FLASH_MODEL_ID,
    CANARY_MODEL_ID,
    MANIFESTS_PATH,
    MAPUCHE_ID,
    MODELS_PATH,
    NAHUATL_PATH,
    QUECHUA_PATH,
)
from utils.logging_utils import setup_logging

logger = setup_logging()


@click.command()
@click.option(
    "--language-mode",
    type=click.Choice(["map", "que", "azz", "multi"]),
    default="azz",
)
@click.option(
    "--model-base", type=click.Choice([CANARY_FLASH_MODEL_ID, CANARY_MODEL_ID])
)
@click.option("--devices", type=int, default=1)
@click.option("--batch-size", type=int, default=8)
@click.option("--max-examples", type=int, default=None)
@click.option("--max-epochs", type=int, default=50)
@click.option("--streaming/--no-streaming", default=True)
@click.option("--models-dir", type=click.Path(), default=MODELS_PATH)
@click.option("--manifests-dir", type=click.Path(), default=MANIFESTS_PATH)
@click.option("--que-dataset-dir", type=click.Path(), default=QUECHUA_PATH)
@click.option("--map-dataset-id", type=str, default=MAPUCHE_ID)
@click.option("--azz-dataset-dir", type=click.Path(), default=NAHUATL_PATH)
def main(
    language_mode,
    model_base,
    devices,
    batch_size,
    max_examples,
    max_epochs,
    streaming,
    models_dir,
    manifests_dir,
    que_dataset_dir,
    map_dataset_id,
    azz_dataset_dir,
):
    hf_login()
    max_examples = 10

    processors = []
    if language_mode in ["map", "multi"]:
        processors.append(
            MapugungunProcessor(
                "map", manifests_dir, map_dataset_id, "spa", max_examples
            )
        )
    if language_mode in ["que", "multi"]:
        processors.append(
            QuechuaProcessor(
                "que",
                manifests_dir,
                que_dataset_dir,
                {
                    "train": [
                        "que_spa_unconstrained/train",
                        "que_spa_synthetic_translation/train",
                    ],
                    "validation": ["que_spa_unconstrained/valid"],
                },
                max_examples,
            )
        )
    if language_mode in ["azz", "multi"]:
        processors.append(
            NahuatlProcessor("azz", manifests_dir, azz_dataset_dir, max_examples)
        )

    if model_base is None:
        model_base = (
            CANARY_MODEL_ID if torch.cuda.is_available() else CANARY_FLASH_MODEL_ID
        )

    model = nemo_asr.models.ASRModel.from_pretrained(model_base)
    model.replace_adapter_compatible_modules()
    model.add_adapter(
        name="encoder:enc",
        cfg=LinearAdapterConfig(in_features=model.cfg.encoder.d_model, dim=8),
    )
    model.freeze()
    model.unfreeze_enabled_adapters()

    data_loader = CanaryMultilingualDataModule(
        tokenizer=model.tokenizer,
        prompt_formatter=model.prompt,
        processors=processors,
        out_data_dir=manifests_dir,
        streaming=streaming,
        batch_size=batch_size,
    )
    data_loader.prepare_data()

    model.cfg.optim.lr = 3e-4
    model.cfg.optim.sched.warmup_steps = 25
    os.makedirs(models_dir, exist_ok=True)

    trainer = L.Trainer(
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        max_epochs=max_epochs,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        logger=TensorBoardLogger(
            save_dir="logs/tb_logs", name=f"canary_{language_mode}"
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=models_dir,
                filename="canary_adapter_{epoch:02d}",
                every_n_epochs=3,
                save_top_k=10,
                monitor="step",
                mode="max",
                save_last=True,
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        use_distributed_sampler=False,
    )

    last_ckpt = os.path.join(models_dir, "last.ckpt")
    trainer.fit(
        model,
        data_loader,
        ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None,
    )
    model.save_adapters(os.path.join(models_dir, f"canary_{language_mode}_final.pt"))


if __name__ == "__main__":
    main()
