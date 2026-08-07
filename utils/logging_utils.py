import logging

from huggingface_hub import utils as hf_utils
from rich.console import Console
from rich.logging import RichHandler

import datasets


def setup_logging(log_file: str = "training_logs.log"):
    """Configures rich logging and silences noisy third-party libraries."""
    console = Console()
    file_handler = logging.FileHandler(log_file)
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

    # Silence noisy loggers
    datasets.disable_progress_bars()
    datasets.utils.logging.set_verbosity_error()
    hf_utils.logging.set_verbosity_error()
    for name in ["httpx", "httpcore", "urllib3", "fsspec", "filelock"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)
