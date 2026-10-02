import copy
import glob
import os
import random
import re
import string
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
import tqdm
import wget
import yaml
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
        self.seed = 42
        self.entries = []

    def _load_segments(self, task: str = "multitask") -> list[dict[str, Any]]:
        """Parse EAF files based on the requested task.

        Args:
            task: One of 'ast', 'asr', or 'both'.
                - 'ast': Only files with translations (ELAN-files-Final-proofed...)
                - 'asr': Only transcription-only files (ELAN-files-First-draft-only)
                - 'multi': All available files
        """
        segments = []

        if task in ["ast", "multitask"]:
            # Load from translations folder (299 files with Spanish)
            eaf_files = list(self.translations_path.rglob("*.eaf"))
            logger.info(f"Found {len(eaf_files)} EAF files with translations")

            for eaf_path in tqdm.tqdm(
                eaf_files, desc=f"{self.name} parsing translated EAFs"
            ):
                try:
                    parsed = EAFParser(eaf_path).get_segments_with_tag(
                        has_translation=True
                    )
                    segments.extend(parsed)
                except Exception as e:
                    logger.error(f"Error processing {eaf_path}: {e}")

        if task in ["asr", "multitask"]:
            # Load from transcriptions folder (439 files, no translation)
            eaf_files = list(self.transcription_path.rglob("*.eaf"))
            logger.info(f"Found {len(eaf_files)} EAF files with transcriptions only")

            for eaf_path in tqdm.tqdm(
                eaf_files, desc=f"{self.name} parsing transcription-only EAFs"
            ):
                try:
                    parsed = EAFParser(eaf_path).get_segments_with_tag(
                        has_translation=False
                    )
                    segments.extend(parsed)
                except Exception as e:
                    logger.error(f"Error processing {eaf_path}: {e}")

        logger.info(f"Collected {len(segments)} segments in total for task '{task}'")
        return segments

    def cut_audio_chunk(
        self, audio_path: Path, segment: dict, task: str
    ) -> Path | None:
        # Save chunks into task-specific subdirectories
        self.chunks_dir = self.data_dir / "audio_chunks" / task
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

        segment_id = segment["start_ts"] + "_" + segment["end_ts"]
        chunk_filename = f"{Path(audio_path).stem}_{segment_id}.wav"
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

        logger.debug(f"Saving chunk {segment_id} for {audio_path.stem}")
        sf.write(chunk_path, chunk_data, samplerate)
        return chunk_path

    def _find_audio_file(self, filename: str) -> Path | None:
        """Search for an audio file by name under self.nahuatl_audios.

        The EAF media paths are stale Windows paths that don't match the
        actual directory layout on disk. We search recursively to find it.
        """
        if not filename:
            return None
        found = list(self.nahuatl_audios.rglob(filename))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            logger.warning(
                f"Duplicate audio file '{filename}' found at:\n"
                + "\n".join(str(p) for p in found)
            )
            return found[0]  # pick first match
        return None

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

    def process(self, task: str = "multitask") -> dict:
        """Process Nahuatl dataset for the given task.

        Args:
            task: 'ast' (translation), 'asr' (transcription only), or 'multi'
        """
        logger.info(f"Processing {self.name} for task '{task}' from EAF files...")
        segments = self._load_segments(task=task)

        splits = self.make_splits(segments)

        entries = {name: [] for name in splits}

        for split, seg_list in splits.items():
            missing = set()
            for seg in seg_list:
                # Apply max_examples limit per split
                if self.max_examples and len(entries[split]) >= self.max_examples:
                    logger.warning(
                        f"Max examples set to {self.max_examples}. {split} reach that value."
                    )
                    break
                audio_path = self._find_audio_file(seg["audio_file"])
                if audio_path is None:
                    if seg["audio_file"] not in missing:
                        logger.warning(f"AUDIO FILE NOT FOUND: {seg['audio_file']}")
                        missing.add(seg["audio_file"])
                    continue

                try:
                    # Determine which task this segment belongs to based on has_translation flag
                    segment_task = "ast" if seg.get("has_translation") else "asr"
                    chunk_path = self.cut_audio_chunk(audio_path, seg, segment_task)
                except Exception as e:
                    logger.error(f"Error chunking {seg['audio_file']}: {e}")
                    continue

                # Build entry differently based on task
                if seg.get("has_translation"):
                    # For AST: use translation as target text
                    entries[split].append(
                        {
                            "audio_filepath": str(chunk_path.absolute()),
                            "duration": seg["duration"],
                            "text": normalize_text(seg["translation"]),
                            "transcription": seg["transcription"],
                            "has_translation": True,
                            "source_lang": "en",
                            "target_lang": "es",
                            "task": segment_task,
                            "pnc": "no",
                        }
                    )
                else:
                    # For ASR-only: use transcription as target text
                    entries[split].append(
                        {
                            "audio_filepath": str(chunk_path.absolute()),
                            "duration": seg["duration"],
                            "text": normalize_text(seg["transcription"]),
                            "transcription": seg["transcription"],
                            "has_translation": False,
                            "source_lang": "es",
                            "target_lang": "es",
                            "task": segment_task,
                            "pnc": "no",
                        }
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

    def get_segments_with_tag(self, has_translation: bool = True):
        """Extract segments from the EAF file.

        Args:
            has_translation: If True, only return segments that have a translation tier.
                           If False, only return segments without translation (for transcription-only datasets).
        """
        transcriptions = {}

        # First pass: extract transcription tiers
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

        # Second pass: extract translation tiers if they exist
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

        # Filter based on has_translation flag
        results = [
            {
                "audio_file": self.media_file,
                "start": seg["start"],
                "end": seg["end"],
                "start_ts": seg["start_ts"],
                "end_ts": seg["end_ts"],
                "duration": seg["end"] - seg["start"],
                "transcription": seg["text"],
                "translation": seg["translation"],
                "has_translation": bool(
                    seg.get("translation") and seg["translation"].strip()
                ),
            }
            for seg in transcriptions.values()
        ]

        if has_translation:
            # Return only segments that actually have translations
            return [s for s in results if s["has_translation"]]
        else:
            # Return only segments without translations
            return [s for s in results if not s["has_translation"]] or results


class AN4Processor(LanguageProcessor):
    """
    Processor for the AN4 dataset that generates both ASR (English) and AST
    (English → German) manifest entries.

    Mirrors the tutorial in Multi_Task_Adapters.py: it reads transcription files,
    builds ASR manifests, uses T5-small to translate English text into German for
    AST manifests, and writes combined train/test manifest pairs.
    """

    def __init__(
        self,
        name: str,
        out_dir: str,
        data_dir: str,
        max_examples: int | None = None,
        do_ast: bool = True,
    ):
        super().__init__(name, out_dir, max_examples)
        os.makedirs(out_dir, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.do_ast = do_ast

        # Paths inside the extracted AN4 archive
        self.train_transcripts = (
            self.data_dir / "an4" / "etc" / "an4_train.transcription"
        )
        self.test_transcripts = self.data_dir / "an4" / "etc" / "an4_test.transcription"
        self.wav_subdir = "wav"

        # Output manifests
        self.train_manifest = os.path.join(out_dir, "train_manifest.json")
        self.test_manifest = os.path.join(out_dir, "test_manifest.json")
        self.ast_train_manifest = os.path.join(out_dir, "ast_train_manifest.json")
        self.ast_test_manifest = os.path.join(out_dir, "ast_test_manifest.json")
        self.combined_train_manifest = os.path.join(
            out_dir, "combined_train_manifest.json"
        )
        self.combined_test_manifest = os.path.join(
            out_dir, "combined_test_manifest.json"
        )

        # Cached T5 model (lazy-loaded)
        self._t5_model = None
        self._t5_tokenizer = None

    # ── helpers ──────────────────────────────────────────────

    @property
    def t5_model(self):
        """Lazy-load the T5 model for translation."""
        if self._t5_model is None:
            import torch

            self._t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            self._t5_model = T5ForConditionalGeneration.from_pretrained(
                "google-t5/t5-small"
            )
            if torch.cuda.is_available():
                self._t5_model = self._t5_model.cuda()
        return self._t5_model

    @property
    def t5_tokenizer(self):
        if self._t5_tokenizer is None:
            _ = self.t5_model  # trigger lazy load
        return self._t5_tokenizer

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of English texts into German via T5."""
        prefix = "translate English to German"
        prompts = [f"{prefix}: {t}" for t in texts]
        input_ids = self.t5_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(self.t5_model.device)
        outputs = self.t5_model.generate(input_ids, max_new_tokens=64)
        return [self.t5_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # ── public API ───────────────────────────────────────────

    def process(self):
        """
        Build ASR + AST manifests from AN4 transcription files and return a dict
        mapping split names → list of manifest entries, identical in structure to
        the other LanguageProcessor implementations:

            {"train": [...], "validation": [], "test": [...]}
        """
        self._prepare_data()

        # ── 1. read base ASR manifests ───────────────────────
        train_entries = self._read_transcription_file(
            self.train_transcripts, "an4/wav/an4_clstk"
        )
        test_entries = self._read_transcription_file(
            self.test_transcripts, "an4/wav/an4test_clstk"
        )

        # ── 2. generate AST manifests (EN→DE) ────────────────
        if self.do_ast:
            ast_train = copy.deepcopy(train_entries)
            ast_test = copy.deepcopy(test_entries)
            batch_size = 32

            for i in tqdm.tqdm(
                range(0, len(train_entries), batch_size), desc="AST train"
            ):
                batch_texts = [x["text"] for x in train_entries[i : i + batch_size]]
                translations = self.translate_batch(batch_texts)
                for j, t in enumerate(translations):
                    ast_train[i + j]["text"] = t
                    ast_train[i + j]["task"] = "ast"
                    ast_train[i + j]["target_lang"] = "de"

            for idx, entry in tqdm.tqdm(enumerate(ast_test), desc="AST test"):
                trans = self.translate_batch([entry["text"]])[0]
                ast_test[idx]["text"] = trans
                ast_test[idx]["task"] = "ast"
                ast_test[idx]["target_lang"] = "de"

            from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

            write_manifest(self.ast_train_manifest, ast_train)
            write_manifest(self.ast_test_manifest, ast_test)

        # ── 3. combine ASR + AST → combined manifests ────────
        combined_train_all = list(train_entries) + (
            copy.deepcopy(ast_train) if self.do_ast else []
        )
        combined_test = list(test_entries) + (
            copy.deepcopy(ast_test) if self.do_ast else []
        )

        # Split a portion of train for validation
        random.seed(42)
        random.shuffle(combined_train_all)
        val_size = int(len(combined_train_all) * 0.1)
        combined_val = combined_train_all[:val_size]
        combined_train = combined_train_all[val_size:]

        from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

        write_manifest(self.train_manifest, train_entries)
        write_manifest(self.test_manifest, test_entries)
        write_manifest(self.combined_train_manifest, combined_train)
        write_manifest(self.combined_test_manifest, combined_test)

        # Return dict in the same shape as other processors:
        # {"train": ..., "validation": ..., "test": ...}
        return {
            "train": combined_train,
            "validation": combined_val,
            "test": combined_test,
        }

    def _prepare_data(self):
        """Download, extract, and convert AN4 dataset if not present."""
        data_dir = self.data_dir
        os.makedirs(data_dir, exist_ok=True)

        tar_path = data_dir / "an4_sphere.tar.gz"

        if not tar_path.exists():
            logger.info("Downloading AN4 dataset...")
            an4_url = (
                "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz"
            )
            wget.download(an4_url, str(data_dir))
        else:
            logger.info("Tarfile already exists.")

        if not (data_dir / "an4").exists():
            logger.info("Extracting AN4 dataset...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=data_dir)

            logger.info("Converting .sph to .wav...")
            sph_list = glob.glob(str(data_dir / "an4/**/*.sph"), recursive=True)
            for sph_path in tqdm.tqdm(sph_list, desc="Converting SPH to WAV"):
                wav_path = sph_path[:-4] + ".wav"
                subprocess.run(["sox", sph_path, wav_path])
            logger.info("Finished conversion.")

    def make_splits(self, segments) -> dict:
        """AN4 uses pre-defined splits from the transcription files; this is a no-op."""
        return {
            "train": segments if isinstance(segments, list) else [],
            "validation": [],
            "test": [],
        }

    # ── internal: read transcription file → manifest entries ─

    def _read_transcription_file(
        self, transcripts_path: Path, wav_subdir_rel: str
    ) -> list[dict]:
        """
        Parse an AN4 .transcription file and return manifest-compatible entries.

        Each line looks like:
            <s> transcript text </s> (fileID)
        """
        if not transcripts_path.exists():
            raise FileNotFoundError(f"Transcription file not found: {transcripts_path}")

        entries = []
        with open(transcripts_path, "r") as fin:
            for line in fin:
                # Parse transcript and file ID
                paren_idx = line.find("(")
                if paren_idx == -1:
                    continue
                transcript = line[:paren_idx].lower().strip()
                transcript = transcript.replace("<s>", "").replace("</s>", "")

                file_id = line[paren_idx + 1 : -2]  # e.g. "cen4-fash-b"
                speaker_subdir = file_id[file_id.find("-") + 1 : file_id.rfind("-")]
                audio_filename = file_id + ".wav"
                audio_path = os.path.join(
                    self.data_dir, wav_subdir_rel, speaker_subdir, audio_filename
                )

                if not os.path.isfile(audio_path):
                    continue

                duration = librosa.core.get_duration(path=audio_path)
                entries.append(
                    {
                        "audio_filepath": os.path.abspath(audio_path),
                        "duration": float(duration),
                        "text": transcript,
                        "pnc": "no",
                        "source_lang": "en",
                        "target_lang": "en",
                        "task": "asr",
                    }
                )

        if self.max_examples is not None:
            entries = entries[: self.max_examples]

        return entries
