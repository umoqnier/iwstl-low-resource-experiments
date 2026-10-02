import os
import sys

import lightning.pytorch as L
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.prompts import PromptFormatter
from omegaconf import OmegaConf

from utils.logging_utils import get_logger

from .dataset import MyCanaryPromptedAudioToTextLhotseDataset, TokenizerSpec

logger = get_logger(__name__)

_LHOTSE_BASE = {
    "sample_rate": 16000,
    "text_field": "text",
    "lang_field": "target_lang",
    "use_bucketing": True,
    "bucket_duration_bins": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    "num_buckets": 7,
    "shuffle_buffer_size": 10000,
    "bucket_buffer_size": 20000,
}


class CanaryMultilingualDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt_formatter: PromptFormatter,
        language_mode: str,
        task: str,
        manifests_dir: str = "./manifests/",
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter
        self.manifests_dir = manifests_dir
        self.language_mode = language_mode
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def manifests(self):
        base = os.path.join(self.manifests_dir, self.language_mode, self.task)
        return {
            "train": os.path.join(base, "train_manifest.json"),
            "validation": os.path.join(base, "val_manifest.json"),
            "test": os.path.join(base, "test_manifest.json"),
        }

    def _setup_dataloader(self, config: dict):
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        return get_lhotse_dataloader_from_config(
            OmegaConf.create(config),
            global_rank=rank,
            world_size=world_size,
            dataset=MyCanaryPromptedAudioToTextLhotseDataset(
                self.tokenizer, self.prompt_formatter
            ),
        )

    def train_dataloader(self):
        cfg = {
            **_LHOTSE_BASE,
            "manifest_filepath": self.manifests["train"],
            "batch_size": self.batch_size,
            "max_duration": 40.0,
            "min_duration": 0.1,
            "num_workers": self.num_workers,
            "shuffle": True,
            "pin_memory": True,
        }
        return self._setup_dataloader(cfg)

    def val_dataloader(self):
        world_size = self.trainer.world_size if self.trainer else 1
        cfg = {
            **_LHOTSE_BASE,
            "manifest_filepath": self.manifests["validation"],
            "batch_size": self.batch_size,
            "max_duration": 15.0,
            "min_duration": 0.1,
            "num_workers": self.num_workers,
            "shuffle": False,
            "pin_memory": False,
        }
        return self._setup_dataloader(cfg)

    def test_dataloader(self):
        cfg = {
            **_LHOTSE_BASE,
            "manifest_filepath": self.manifests["test"],
            "batch_size": self.batch_size,
            "max_duration": 15.0,
            "min_duration": 0.1,
            "num_workers": 1,
            "shuffle": False,
            "pin_memory": False,
        }
        return self._setup_dataloader(cfg)

    def setup(self, stage=None):
        manifests = self.manifests
        for split in ("train", "validation", "test"):
            if not os.path.exists(manifests[split]):
                logger.error(
                    "Manifest not found: %s. Run build_datasets.py first.",
                    manifests[split],
                )
                sys.exit(1)
