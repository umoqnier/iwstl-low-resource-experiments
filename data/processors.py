import os
import random
import re
import string
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import tqdm
import yaml

from datasets import load_dataset
from utils.configs import (
    MAPUCHE_DATASET_PATH,
    NAHUATL_AUDIOS_PATH,
    NAHUATL_TRANSCRIPTIONS_PATH,
    NAHUATL_TRANSLATIONS_PATH,
    SPLITS_RATIOS,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text

    Make it lowercase, remove punctuation, replace whitespace with a single whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + "¿¡"))
    return re.sub(r"\s+", " ", text).strip()


class LanguageProcessor(ABC):
    def __init__(self, name: str, out_dir: str, max_examples: int | None = None):
        self.name = name
        self.out_dir = out_dir
        self.max_examples = max_examples

    @abstractmethod
    def process(self) -> dict:
        pass

    @abstractmethod
    def make_splits(self, segments) -> dict:
        pass


class MapugungunProcessor(LanguageProcessor):
    def __init__(
        self,
        name: str,
        out_dir: str,
        dataset_id: str,
        streaming: bool,
        text_column: str,
        max_examples: int | None = None,
    ):
        super().__init__(name, out_dir, max_examples)
        self.dataset_id = dataset_id
        self.text_column = text_column
        self.streaming = streaming
        self.splits = ["train", "validation", "test"]

    def make_splits(self, segments):
        return segments

    def process(self, task="ast"):
        logger.info(f"Processing {self.name} from HF")

        dataset_dict = load_dataset(self.dataset_id, streaming=self.streaming)
        entries = {"train": [], "validation": [], "test": []}
        for split, dataset in dataset_dict.items():
            logger.info(f"Split = {split}")

            wav_chunks_dir = MAPUCHE_DATASET_PATH / Path(split)
            if not wav_chunks_dir.exists():
                os.makedirs(wav_chunks_dir, exist_ok=True)

            if self.max_examples is not None:
                if self.streaming:
                    dataset = dataset.take(self.max_examples)
                else:
                    dataset = dataset.select(
                        range(min(len(dataset), self.max_examples))
                    )
            else:
                # TODO: Implement this
                pass

            for idx, item in enumerate(dataset):
                audio_info = item.get("audio")
                if not audio_info:
                    continue

                filepath = wav_chunks_dir / Path(f"{self.name}_{idx}.wav")
                if not filepath.exists():
                    sf.write(filepath, audio_info["array"], audio_info["sampling_rate"])

                duration = len(audio_info["array"]) / audio_info["sampling_rate"]
                spanish_text = item.get(self.text_column, "")
                arn_text = item.get("arn", "")
                arn_clean_text = item.get("arn_clean", "")
                if spanish_text:
                    # TODO: Base on task "text" field need to change from spanish to mapuche
                    entries[split].append(
                        {
                            "arn": arn_text,
                            "arn_clean": arn_clean_text,
                            "spa": spanish_text,
                            "text": normalize_text(spanish_text),
                            "duration": duration,
                            "pnc": "no",
                            "source_lang": "en",
                            "target_lang": "es",
                            "audio_filepath": os.path.abspath(filepath),
                        }
                    )
        return entries


class QuechuaProcessor(LanguageProcessor):
    def __init__(
        self,
        name: str,
        out_dir: str,
        data_dir: str,
        split_mapping: dict[str, list[str]],
        max_examples: int | None = None,
    ):
        super().__init__(name, out_dir, max_examples)
        self.data_dir = data_dir
        self.split_mapping = split_mapping

    def process(self):
        logger.info(f"Processing {self.name} split: {split} from local files...")
        subdirs = self.split_mapping.get(split, [])
        entries = []
        for subdir in subdirs:
            split_dir = os.path.join(self.data_dir, subdir)
            if not os.path.exists(split_dir):
                continue
            local_split = os.path.basename(subdir)
            yaml_path = os.path.join(split_dir, "txt", f"{local_split}.yaml")
            text_path = os.path.join(split_dir, "txt", f"{local_split}.spa")
            wav_dir = os.path.join(split_dir, "wav")
            if not (os.path.exists(yaml_path) and os.path.exists(text_path)):
                continue
            with open(yaml_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
            with open(text_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f.readlines()]
            for i, meta in tqdm.tqdm(enumerate(metadata), desc=f"{self.name} {subdir}"):
                if self.max_examples and len(entries) >= self.max_examples:
                    break
                audio_path = os.path.abspath(os.path.join(wav_dir, meta["wav"]))
                if os.path.exists(audio_path):
                    entries.append(
                        {
                            "audio_filepath": audio_path,
                            "duration": meta["duration"],
                            "text": texts[i],
                            "pnc": "no",
                            "source_lang": "en",
                            "target_lang": self.name,
                        }
                    )
        return entries


class NahuatlProcessor(LanguageProcessor):
    def __init__(
        self, name: str, out_dir: str, data_dir: str, max_examples: int | None = None
    ):
        super().__init__(name, out_dir, max_examples)
        self.data_dir = Path(data_dir)
        self.translations_path = NAHUATL_TRANSLATIONS_PATH
        self.transcription_path = NAHUATL_TRANSCRIPTIONS_PATH
        self.nahuatl_audios = NAHUATL_AUDIOS_PATH
        self.chunks_dir = self.data_dir / Path("audio_chunks")
        self.seed = 42
        self.entries = []

    def _load_segments(self) -> list[dict[str, Any]]:
        """Parse every EAF file and return a flat list"""
        eaf_files = set()
        eaf_dirs = [f.name for f in self.translations_path.iterdir() if f.is_dir()]
        for eaf_dir in eaf_dirs:
            eaf_files.update((self.translations_path / Path(eaf_dir)).glob("*.eaf"))
        logger.info(f"Found {len(eaf_files)} EAF files")

        segments = []
        for eaf_path in tqdm.tqdm(sorted(eaf_files), desc=f"{self.name} parsing EAFs"):
            try:
                segments.extend(EAFParser(eaf_path).get_segments())
            except Exception as e:
                logger.error(f"Error processing {eaf_path}: {e}")

        logger.info(f"Collected {len(segments)} segments in total")
        return segments

    def cut_audio_chunk(self, audio_path: Path, segment: dict) -> Path | None:
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        segment_id = segment["start_ts"] + "_" + segment["end_ts"]
        chunk_filename = f"{audio_path.stem}_{segment_id}.wav"
        chunk_path = self.chunks_dir / chunk_filename
        if chunk_path.exists():
            logger.warning(f"chunk path {chunk_path} already exists. Skipping")
            return chunk_path

        start_chunk = segment["start"]
        end_chunk = segment["end"]

        # Cut audio and save as chunk
        audio_info = sf.info(audio_path)
        samplerate = audio_info.samplerate
        start_sample = int(start_chunk * samplerate)
        end_sample = int(end_chunk * samplerate)
        num_frames = end_sample - start_sample

        if num_frames <= 0:
            logger.warning(
                f"Invalid chunk duration for {segment_id}: {num_frames} frames"
            )
            raise Exception(f"Invalid chunk duration {segment_id}: {num_frames} frames")

        # Read only the specific chunk from the file
        chunk_data = sf.read(audio_path, start=start_sample, frames=num_frames)[0]

        # Convert to mono if stereo/multi-channel by averaging channels
        if len(chunk_data.shape) > 1:
            chunk_data = chunk_data.mean(axis=1)

        logger.info(f"Saving chunk {segment_id} for {audio_path.stem}")
        sf.write(chunk_path, chunk_data, samplerate)
        return chunk_path

    def make_splits(self, segments):
        if not segments:
            logger.warning("No segments to split; returning empty splits")
            return {"train": [], "validation": [], "test": []}

        by_audio = defaultdict(list)
        for seg in segments:
            by_audio[seg["audio_file"]].append(seg)

        audio_files = sorted(by_audio.keys())
        rng = random.Random(self.seed)
        rng.shuffle(audio_files)

        n = len(audio_files)
        cut_train_full = int(
            SPLITS_RATIOS["train"] * n
        )  # boundary between train_full and test
        cut_dev = int(
            (1 - SPLITS_RATIOS["validation"]) * cut_train_full
        )  # boundary between train and dev within train_full

        train_files = audio_files[:cut_dev]
        dev_files = audio_files[cut_dev:cut_train_full]
        test_files = audio_files[cut_train_full:]

        assert n == len(train_files) + len(dev_files) + len(test_files)

        splits = {
            "train": [seg for f in train_files for seg in by_audio[f]],
            "validation": [seg for f in dev_files for seg in by_audio[f]],
            "test": [seg for f in test_files for seg in by_audio[f]],
        }

        for name, segs in splits.items():
            files = {"train": train_files, "validation": dev_files, "test": test_files}[
                name
            ]
            hours = sum(s["duration"] for s in segs) / 3600.0
            logger.info(
                f"{self.name} {name}: {len(files)} files, "
                f"{len(segs)} segments, {hours:.2f} hrs "
                f"({len(files) / n:.1%} of files)"
            )
        all_files = set(train_files) | set(dev_files) | set(test_files)
        assert len(all_files) == n, "Audio file leakage across splits!"
        return splits

    def process(self) -> dict:
        logger.info(f"Processing {self.name} from EAF files...")
        segments = self._load_segments()

        splits = self.make_splits(segments)

        entries = {name: [] for name in splits}

        for split, segments in splits.items():
            for seg in segments:
                entries_count = len(entries[split])
                if self.max_examples and entries_count >= self.max_examples:
                    break

                audio_path = self.nahuatl_audios / seg["audio_file"]
                if not audio_path.exists():
                    logger.warning(f"NAHUATL AUDIO PATH {audio_path} NOT FOUND")
                    break

                try:
                    chunk_path = self.cut_audio_chunk(audio_path, seg)
                except Exception as e:
                    logger.error(f"Error chunking {seg['audio_file']}: {e}")
                    continue

                entries[split].append(
                    {
                        "audio_filepath": str(chunk_path.absolute()),
                        "duration": seg["duration"],
                        "text": normalize_text(seg["translation"]),
                        "transcription": seg["transcription"],
                        "pnc": "No",
                        "source_lang": "en",
                        "target_lang": "es",
                    }
                )
            hours = sum(e["duration"] for e in entries[split]) / 3600.0
            logger.info(
                f"{self.name} {split}: {len(entries[split])} segments, {hours:.2f} hrs"
            )
        return entries


class EAFParser:
    def __init__(self, eaf_path):
        self.eaf_path = Path(eaf_path)
        self.tree = ET.parse(eaf_path)
        self.root = self.tree.getroot()

        self.time_slots = self._parse_time_slots()
        self.media_file: str | None = self._extract_media_file()

    def _parse_time_slots(self) -> dict:
        slots = {}
        time_order = self.root.find("TIME_ORDER")
        if time_order is not None:
            for slot in time_order.findall("TIME_SLOT"):
                slots[slot.get("TIME_SLOT_ID")] = int(slot.get("TIME_VALUE"))
        return slots

    def _extract_media_file(self) -> str | None:
        header = self.root.find("HEADER")
        if header is not None:
            descriptor = header.find("MEDIA_DESCRIPTOR")
            if descriptor is not None:
                media_url = descriptor.get("MEDIA_URL") or descriptor.get(
                    "RELATIVE_MEDIA_URL"
                )
                if media_url:
                    return os.path.basename(media_url)
        return None

    def get_segments(self):
        transcriptions = {}
        for tier in self.root.findall("TIER"):
            ling_type_ref = tier.get("LINGUISTIC_TYPE_REF")
            if ling_type_ref in ["Transcripción", "UtteranceType"]:
                for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                    ann_id = ann.get("ANNOTATION_ID")
                    start_ts = ann.get("TIME_SLOT_REF1")
                    end_ts = ann.get("TIME_SLOT_REF2")

                    start_time = self.time_slots.get(start_ts, 0) / 1000.0
                    end_time = self.time_slots.get(end_ts, 0) / 1000.0

                    text_val = (
                        ann.find("ANNOTATION_VALUE").text
                        if ann.find("ANNOTATION_VALUE") is not None
                        else ""
                    )

                    transcriptions[ann_id] = {
                        "start": start_time,
                        "end": end_time,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "text": text_val,
                        "translation": None,
                    }

        for tier in self.root.findall("TIER"):
            if tier.get("LINGUISTIC_TYPE_REF") == "Traducción":
                for ann in tier.findall(".//REF_ANNOTATION"):
                    ref_id = ann.get("ANNOTATION_REF")
                    if ref_id in transcriptions:
                        text_val = (
                            ann.find("ANNOTATION_VALUE").text
                            if ann.find("ANNOTATION_VALUE") is not None
                            else ""
                        )
                        transcriptions[ref_id]["translation"] = text_val

        return [
            {
                "audio_file": self.media_file,
                "start": seg["start"],
                "end": seg["end"],
                "start_ts": seg["start_ts"],
                "end_ts": seg["end_ts"],
                "duration": seg["end"] - seg["start"],
                "transcription": seg["text"],
                "translation": seg["translation"],
            }
            for seg in transcriptions.values()
            if seg["translation"]
        ]
