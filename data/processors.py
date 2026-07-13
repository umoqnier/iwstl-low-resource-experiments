import logging
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
import tqdm
import yaml

from datasets import load_dataset

logger = logging.getLogger(__name__)


class LanguageProcessor(ABC):
    def __init__(self, name: str, out_dir: str, max_examples: Optional[int] = None):
        self.name = name
        self.out_dir = out_dir
        self.max_examples = max_examples

    @abstractmethod
    def process(self, split: str, streaming: bool) -> list[dict[str, Any]]:
        pass


class MapugungunProcessor(LanguageProcessor):
    def __init__(
        self,
        name: str,
        out_dir: str,
        dataset_id: str,
        text_column: str,
        max_examples: Optional[int] = None,
    ):
        super().__init__(name, out_dir, max_examples)
        self.dataset_id = dataset_id
        self.text_column = text_column

    def process(self, split: str, streaming: bool) -> list[dict[str, Any]]:
        logger.info(f"Processing {self.name} split: {split} from HF...")
        dataset = load_dataset(self.dataset_id, split=split, streaming=streaming)
        wav_dir = os.path.join(self.out_dir, f"{self.name}_wavs")
        os.makedirs(wav_dir, exist_ok=True)

        if self.max_examples is not None:
            dataset = (
                dataset.take(self.max_examples)
                if streaming
                else dataset.select(range(min(len(dataset), self.max_examples)))
            )

        entries = []
        total = (
            self.max_examples if streaming else len(dataset) if not streaming else None
        )

        for idx, item in tqdm.tqdm(
            enumerate(dataset), total=total, desc=f"{self.name} {split}"
        ):
            audio_info = item.get("audio")
            if not audio_info:
                continue
            filepath = os.path.join(wav_dir, f"{self.name}_{split}_{idx}.wav")
            if not os.path.exists(filepath):
                sf.write(filepath, audio_info["array"], audio_info["sampling_rate"])

            duration = len(audio_info["array"]) / audio_info["sampling_rate"]
            text = item.get(self.text_column, "")
            if text:
                entries.append(
                    {
                        "audio_filepath": os.path.abspath(filepath),
                        "duration": duration,
                        "text": text,
                        "pnc": "no",
                        "source_lang": "en",
                        "target_lang": self.name,
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
        max_examples: Optional[int] = None,
    ):
        super().__init__(name, out_dir, max_examples)
        self.data_dir = data_dir
        self.split_mapping = split_mapping

    def process(self, split: str, streaming: bool) -> list[dict[str, Any]]:
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
        self, name: str, out_dir: str, data_dir: str, max_examples: Optional[int] = None
    ):
        super().__init__(name, out_dir, max_examples)
        self.data_dir = Path(data_dir)

    def process(self, split: str, streaming: bool) -> list[dict[str, Any]]:
        logger.info(f"Processing {self.name} split: {split} from EAF files...")
        entries = []
        eaf_files = list(self.data_dir.glob("*.eaf"))

        for eaf_path in tqdm.tqdm(eaf_files, desc=f"{self.name} {split}"):
            if self.max_examples and len(entries) >= self.max_examples:
                break

            try:
                parser = EAFParser(eaf_path)
                segments = parser.get_segments()
                for seg in segments:
                    if self.max_examples and len(entries) >= self.max_examples:
                        break

                    audio_path = self.data_dir / parser.AUDIOS_PATH / seg["audio_file"]
                    if not audio_path.exists():
                        continue

                    entries.append(
                        {
                            "audio_filepath": str(audio_path.absolute()),
                            "duration": seg["duration"],
                            "text": seg["transcription"],
                            "translation": seg["translation"],
                            "pnc": "no",
                            "source_lang": "es",
                            "target_lang": self.name,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing {eaf_path}: {e}")

        return entries


class EAFParser:
    AUDIOS_PATH = "Sound-files-Puebla-Nahuatl"

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
                "duration": seg["end"] - seg["start"],
                "transcription": seg["text"],
                "translation": seg["translation"],
            }
            for seg in transcriptions.values()
            if seg["translation"]
        ]
