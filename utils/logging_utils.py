import logging
import os
from logging.handlers import RotatingFileHandler

from huggingface_hub import utils as hf_utils

import datasets

_CONFIGURED = False


def setup_logging(
    log_file: str | os.PathLike | None = None,
    level: str | int = "INFO",
    rank: int | None = None,
):
    """Configure the root logger exactly once"""
    global _CONFIGURED
    root = logging.getLogger()

    # Reset any pre-existing handlers on the root so our config takes
    # effect even if NeMo/Lightning already called basicConfig.
    for h in list(root.handlers):
        root.removeHandler(h)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler: plain text, no ANSI. Goes to stderr so it survives
    # stdout redirection without polluting the data stream.
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    if log_file is not None:
        log_path = os.fspath(log_file)
        if rank is not None:
            base, ext = os.path.splitext(log_path)
            log_path = f"{base}_rank{rank}{ext}"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=50 * 1024 * 1024,  # 50 MiB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    # Silence libraries that are genuinely noise.
    datasets.disable_progress_bars()
    datasets.utils.logging.set_verbosity_error()
    hf_utils.logging.set_verbosity_error()

    for name in (
        "httpx",
        "httpcore",
        "urllib3",
        "fsspec",
        "filelock",
        # Lightning chatter
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators.cuda",
        "lightning_fabric",
        # PyTorch DDP
        "torch.distributed.elastic",
        "torch.distributed.ddp_shard",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger; configure logging on first call."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name if name is not None else __name__)
