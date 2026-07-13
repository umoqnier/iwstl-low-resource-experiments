import itertools
import logging
import os

import lightning.pytorch as L
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.prompts import PromptFormatter
from omegaconf import OmegaConf

from .dataset import MyCanaryPromptedAudioToTextLhotseDataset, TokenizerSpec
from .processors import LanguageProcessor

logger = logging.getLogger(__name__)


class CanaryMultilingualDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt_formatter: PromptFormatter,
        processors: list[LanguageProcessor],
        out_data_dir: str = "./combined_data/",
        batch_size: int = 8,
        streaming: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter
        self.processors = processors
        self.out_data_dir = out_data_dir
        self.batch_size = batch_size
        self.streaming = streaming
        self.manifests = {
            "train": os.path.join(out_data_dir, "train_manifest.json"),
            "validation": os.path.join(out_data_dir, "val_manifest.json"),
            "test": os.path.join(out_data_dir, "test_manifest.json"),
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
        return self._setup_dataloader(
            {
                "manifest_filepath": self.manifests["train"],
                "num_buckets": 30,
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": True,
                "persistent_workers": True,
                "pin_memory": True,
            }
        )

    def val_dataloader(self):
        return self._setup_dataloader(
            {
                "manifest_filepath": self.manifests["validation"],
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": False,
                "persistent_workers": True,
                "pin_memory": True,
            }
        )

    def test_dataloader(self):
        if not os.path.exists(self.manifests["test"]):
            return self.val_dataloader()
        return self._setup_dataloader(
            {
                "manifest_filepath": self.manifests["test"],
                "batch_size": self.batch_size,
                "num_workers": 4,
                "shuffle": False,
            }
        )

    def prepare_data(self):
        os.makedirs(self.out_data_dir, exist_ok=True)
        if os.path.exists(self.manifests["train"]) and os.path.exists(
            self.manifests["validation"]
        ):
            return

        for split in ["train", "validation"]:
            all_lang_entries = [
                p.process(split, self.streaming) for p in self.processors
            ]
            combined_entries = []
            for entries_tuple in itertools.zip_longest(*all_lang_entries):
                for entry in entries_tuple:
                    if entry is not None:
                        combined_entries.append(entry)

            path = (
                self.manifests["train"]
                if split == "train"
                else self.manifests["validation"]
            )
            write_manifest(path, combined_entries)
