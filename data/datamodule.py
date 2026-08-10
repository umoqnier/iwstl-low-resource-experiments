import itertools
import os

import lightning.pytorch as L
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.prompts import PromptFormatter
from omegaconf import OmegaConf

from utils.logging_utils import get_logger

from .dataset import MyCanaryPromptedAudioToTextLhotseDataset, TokenizerSpec
from .processors import LanguageProcessor

logger = get_logger(__name__)

_LHOTSE_BASE = {
    "sample_rate": 16000,
    "text_field": "text",  # matches the "text" key your processors write
    "lang_field": "target_lang",
    "use_bucketing": True,
    "bucket_duration_bins": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    "num_buckets": 7,  # one per bin above
    "shuffle_buffer_size": 10000,
    "bucket_buffer_size": 20000,
}


class CanaryMultilingualDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt_formatter: PromptFormatter,
        processors: list[LanguageProcessor],
        lang_mode: str,
        out_data_dir: str = "./combined_data/",
        batch_size: int = 8,
        streaming: bool = False,
        num_workers: int = 2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter
        self.processors = processors
        self.out_data_dir = out_data_dir
        self.batch_size = batch_size
        self.streaming = streaming
        self.num_workers = num_workers
        self.manifests = {
            "train": os.path.join(out_data_dir, "train_manifest.json"),
            "validation": os.path.join(out_data_dir, "val_manifest.json"),
            "test": os.path.join(out_data_dir, "test_manifest.json"),
        }
        self.lang_mode = lang_mode

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
            "max_duration": 40.0,  # cap total seconds per batch
            "min_duration": 0.1,
            "num_workers": self.num_workers,
            "shuffle": True,
            "pin_memory": True,
        }
        return self._setup_dataloader(cfg)

    def val_dataloader(self):
        cfg = {
            **_LHOTSE_BASE,
            "manifest_filepath": self.manifests["validation"],
            "batch_size": self.batch_size,
            "max_duration": 40.0,
            "min_duration": 0.1,
            "num_workers": self.num_workers,
            "shuffle": False,
            "pin_memory": True,
        }
        return self._setup_dataloader(cfg)

    def test_dataloader(self):
        cfg = {
            **_LHOTSE_BASE,
            "manifest_filepath": self.manifests["validation"],
            "batch_size": self.batch_size,
            "max_duration": 40.0,
            "min_duration": 0.1,
            "num_workers": self.num_workers,
            "shuffle": False,
            "pin_memory": True,
        }
        # if not os.path.exists(self.manifests["test"]):
        #    return self._setup_dataloader(cfg)
        return self._setup_dataloader(cfg)

    def prepare_data(self):
        os.makedirs(self.out_data_dir, exist_ok=True)
        if (
            os.path.exists(self.manifests["train"])
            and os.path.exists(self.manifests["validation"])
            and os.path.exists(self.manifests["test"])
        ):
            logger.warning("Manifests already exists. Skipping creation")
            return

        if self.lang_mode == "multi":
            all_lang_entries = [p.process() for p in self.processors]
            entries: list[dict] = []
            for entries_tuple in itertools.zip_longest(*all_lang_entries):
                for entry in entries_tuple:
                    if entry is not None:
                        entries.append(entry)
        else:
            # We only have one processor
            entries = self.processors[0].process()
        for split in entries:
            write_manifest(self.manifests[split], entries[split])
