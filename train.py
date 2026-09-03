import os

import click
import lightning.pytorch as L
import nemo.collections.asr as nemo_asr

# Cap PyTorch's CUDA caching allocator so DDP can't hoard system RAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
from utils.logging_utils import get_logger, setup_logging


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
@click.option("--adapter-enc-dim", type=int, default=32)
@click.option("--adapter-dec-dim", type=int, default=32)
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
    adapter_enc_dim,
    adapter_dec_dim,
    max_examples,
    max_epochs,
    streaming,
    models_dir,
    manifests_dir,
    que_dataset_dir,
    map_dataset_id,
    azz_dataset_dir,
):
    setup_logging(
        log_file=f"logs/training_logs_{language_mode}.log",
        level=os.environ.get("LOG_LEVEL", "INFO"),
    )
    logger = get_logger(__name__)

    hf_login()

    n_gpus = torch.cuda.device_count()
    logger.info(
        "Env: PYTORCH_CUDA_ALLOC_CONF=%r, visible_gpus=%d, devices_arg=%d",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        n_gpus,
        devices,
    )
    if n_gpus > 0:
        for i in range(n_gpus):
            free, total = torch.cuda.mem_get_info(i)
            logger.info(
                "GPU %d: %s, free=%.2f GiB / total=%.2f GiB",
                i,
                torch.cuda.get_device_name(i),
                free / 2**30,
                total / 2**30,
            )

    processors = []
    if language_mode in ["map", "multi"]:
        processors.append(
            MapugungunProcessor(
                "map", manifests_dir, map_dataset_id, "spa", max_examples
            )
        )
    if language_mode in ["que", "multi"]:
        logger.warning("TODO: implement this :(")
        return
    if language_mode in ["azz", "multi"]:
        processors.append(
            NahuatlProcessor("azz", manifests_dir, azz_dataset_dir, max_examples)
        )

    if model_base is None:
        model_base = (
            CANARY_MODEL_ID if torch.cuda.is_available() else CANARY_FLASH_MODEL_ID
        )

    logger.info("Setting up torch float32 matmul precision to medium")
    torch.set_float32_matmul_precision("medium")
    logger.info("Loading base model: %s", model_base)
    model = nemo_asr.models.ASRModel.from_pretrained(model_base)

    # Enable adapters
    model.replace_adapter_compatible_modules()

    # Training acustic processing
    model.add_adapter(
        name="transf_encoder:enc",
        cfg=LinearAdapterConfig(
            in_features=model.cfg.encoder.d_model, dim=adapter_enc_dim
        ),
    )

    # Adapting text generation (due to code-switching)
    model.add_adapter(
        name="tranf_decoder:dec",
        cfg=LinearAdapterConfig(
            in_features=model.cfg.encoder.d_model, dim=adapter_dec_dim
        ),
    )
    model.freeze()
    model.unfreeze_enabled_adapters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model params: trainable=%d (%.2f M), total=%d (%.2f M)",
        trainable,
        trainable / 1e6,
        total,
        total / 1e6,
    )

    data_loader = CanaryMultilingualDataModule(
        tokenizer=model.tokenizer,
        prompt_formatter=model.prompt,
        processors=processors,
        out_data_dir=manifests_dir,
        streaming=streaming,
        batch_size=batch_size,
        lang_mode=language_mode,
    )
    data_loader.prepare_data()

    model.cfg.optim.lr = 3e-4
    model.cfg.optim.sched.warmup_steps = 25
    os.makedirs(models_dir, exist_ok=True)

    strategy = "ddp" if devices > 1 else "auto"
    logger.info(
        "Trainer: devices=%d, strategy=%s, precision=bf16-mixed, "
        "accumulate_grad_batches=4, gradient_clip_val=1.0",
        devices,
        strategy,
    )

    trainer = L.Trainer(
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        max_epochs=max_epochs,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        logger=TensorBoardLogger(
            save_dir=f"logs/{lang_mode}", name=f"canary_{language_mode}"
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=models_dir,
                filename="canary_adapter_azz_{epoch:02d}",
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

    if n_gpus > 1 and data_loader.num_workers >= 4:
        logger.warning(
            "num_workers=%d * world_size=%d = %d worker procs. "
            "If you see OOM at val, lower num_workers to 2.",
            data_loader.num_workers,
            n_gpus,
            data_loader.num_workers * n_gpus,
        )

    last_ckpt = os.path.join(models_dir, "last.ckpt")
    trainer.fit(
        model,
        data_loader,
        ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None,
    )
    final_model_path = os.path.join(
        models_dir,
        f"canary_{language_mode}_enc_adap_{adapter_enc_dim}_dec_adap_{adapter_dec_dim}_final.pt",
    )
    logger.info(f"FINISHED TRAINING. Saving final model at {final_model_path}")
    model.save_adapters(final_model_path)


if __name__ == "__main__":
    main()
