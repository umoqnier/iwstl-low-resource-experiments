# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experimentación con canary 🐤

# %%
import torch

print(f"Is CUDA available? {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    MODEL = "nvidia/canary-1b-flash"
else:
    MODEL = "nvidia/canary-1b-v2"

gpu_count = torch.cuda.device_count()
print(f"Number of GPUs detected: {gpu_count}")

for i in range(gpu_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

breakpoint()
# %% [markdown]
# ## Modelo

# %%
import nemo.collections.asr as nemo_asr

# %%
model = nemo_asr.models.ASRModel.from_pretrained(MODEL)

# %% [markdown]
# ## Habilitando soporte de Adapters

# %%
model.replace_adapter_compatible_modules()

# %% [markdown]
# ### Verificación de que *targets* son soportados por el modelo

# %%
model.adapter_module_names

# %% [markdown]
# ### Preparando los *Adapters*

# %%
from nemo.collections.common.parts import LinearAdapterConfig

# %%
input_dim = model.cfg.encoder.d_model
adapter_dim = 8

# %%
enc_adapter_cfg = LinearAdapterConfig(in_features=input_dim, dim=adapter_dim)

# %% [markdown]
# ### Agregando *Adapters* (solo encoder)

# %%
model.add_adapter(name="encoder:enc", cfg=enc_adapter_cfg)

# %% [markdown]
# ### Congelando parámetros del modelo y descongelando solo los pesos de *Adapters*

# %%
model.freeze()
model.unfreeze_enabled_adapters()

# %%
model.summarize()

# %% [markdown]
# ### Comprobando los *Adapters habilidatos*

# %%
model.get_enabled_adapters()

# %% [markdown]
# ## Dataset - Mapuche

# %%
from huggingface_hub import login

login()

# %%
# Do it on terminal
# !hf auth login

# %%
# !hf auth whoami

# %%
from datasets import load_dataset, load_dataset_builder

# %% [markdown]
# ### Exploración del dataset

# %% [markdown]
# #### Maputzun

# %%
MAP_DATASET_ID = "mengct00/Mapudungun_iwslt26"

# %%
map_databuilder = load_dataset_builder(MAP_DATASET_ID)

# %%
map_databuilder.info.features

# %%
map_databuilder.info.splits

# %% [markdown]
# #### Quechua

# %%
import soundfile as sf
import os
from pathlib import Path


# %%
def calculate_total_minutes(root_dir, splits=["train", "valid"]):
    results = {}

    for split in splits:
        wav_dir = Path(root_dir) / split / 'wav'
        total_seconds = 0.0

        if not wav_dir.exists():
            print(f"Directory not found: {wav_dir}")
            continue

        for wav_file in wav_dir.glob('*.wav'):
            try:
                data, samplerate = sf.read(wav_file)
                duration = len(data) / samplerate
                total_seconds += duration
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

        total_minutes = total_seconds / 60
        results[split] = total_minutes

    return results


data_dir = "data/que_data/que_spa_unconstrained"
durations = calculate_total_minutes(data_dir)

for split, minutes in durations.items():
    print(f"{split.capitalize()} split: {minutes/60:.2f} hours")

# %%
data_dir = "data/que_data/que_spa_synthetic_translation"
durations = calculate_total_minutes(data_dir, splits=["train"])

for split, minutes in durations.items():
    print(f"{split.capitalize()} split: {minutes/60:.2f} hours")

# %% [markdown]
# ### Preparación del dataset

# %% [markdown]
# Canary hace uso de un *prompt* que maneja que tipo de tarea vamos a resolver. Necesitamos entregarle los datos en un formato que cumpla con el formato de la clase que define el *prompt*.

# %%
# model.prompt?

# %%
model.prompt.TEMPLATE

# %% [markdown]
# Usaremos la clase `PromptedAudioToTextLhotseDataset` predefinida en la biblioteca de Nemo. Esta clase mapea items del manifest definido por nosotros  a items definidos en el *prompt template* del modelo. Asi, mientras el *manifest* corresponda con los *slots* soportados por el modelo, estos seran manejados por el Dataset automáticamente. 

# %%
import torch
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.common.data.prompt_fn import (
    get_prompt_format_fn,
    #registered_prompt_format_fn,
)
from torch.utils.data import Dataset
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors
from lhotse import CutSet
from nemo.collections.common.prompts import PromptFormatter
from lhotse.cut import Cut


class MyCanaryPromptedAudioToTextLhotseDataset(Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``Cut`` a single MonoCut or MixedCut
    * a ``PromptFormatter`` Prepend and append control tokens to the token sequence

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """

    def __init__(self, tokenizer: "TokenizerSpec", prompt: PromptFormatter):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.prompt_format_fn = get_prompt_format_fn(
            Cut, self.prompt
        )  # Use the default canary prompt function

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)
        answers = []
        prompts = []
        prompts_with_answers = []

        for cut in cuts:
            prompted_answers = self.prompt_format_fn(cut, self.prompt)
            answers.append(prompted_answers["answer_ids"])
            prompts.append(prompted_answers["context_ids"])
            prompts_with_answers.append(prompted_answers["input_ids"])

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(
            prompts_with_answers
        )
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            cuts=cuts.drop_in_memory_data(),
        )

    def _collate_tokens(
        self, tokens: list[list[int] | torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens


# %%
import itertools
import yaml
import tqdm
import lightning.pytorch as L
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest, read_manifest
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
#from nemo.collections.common.prompts import CanaryPromptFormatter

# %%
class CanaryMultilingualDataModule(L.LightningDataModule):
    def __init__(
        self, 
        tokenizer, 
        prompt_formatter,
        map_dataset_name: str = "mengct00/Mapudungun_iwslt26",
        que_data_dir: str = "./data/que_data/",
        out_data_dir: str = "./combined_data/", 
        batch_size: int = 8,
        streaming: bool = False,
        max_examples: int | None = None, # Applied per language
        language_mode: str = "multi" # Accepts: "map", "que", "multi"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter
        
        self.map_dataset_name = map_dataset_name
        self.que_data_dir = que_data_dir
        self.out_data_dir = out_data_dir
        self.batch_size = batch_size
        self.streaming = streaming
        self.max_examples = max_examples
        
        if language_mode not in ["map", "que", "multi"]:
            raise ValueError("language_mode must be 'map', 'que', or 'multi'")
        self.language_mode = language_mode

        # AST manifests paths 
        self.train_manifest = os.path.join(out_data_dir, f"train_{self.language_mode}_manifest.json")
        self.val_manifest = os.path.join(out_data_dir, f"val_{self.language_mode}_manifest.json")
        self.test_manifest = os.path.join(out_data_dir, f"test_{self.language_mode}_manifest.json")

    def setup(self, stage=None):
        pass

    def _setup_dataloader(self, config):
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        
        return get_lhotse_dataloader_from_config(
            OmegaConf.create(config),
            global_rank=rank,
            world_size=world_size,
            dataset=MyCanaryPromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer, 
                prompt=self.prompt_formatter
            ),
        )

    def train_dataloader(self):
        return self._setup_dataloader({
            "manifest_filepath": self.train_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": True,
        })

    def val_dataloader(self):
        return self._setup_dataloader({
            "manifest_filepath": self.val_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": False,
        })

    def test_dataloader(self):
        # Optional: handle cases where test split might not exist
        if not os.path.exists(self.test_manifest):
            print("Warning: No test manifest found, returning val_dataloader for test.")
            return self.val_dataloader()
            
        return self._setup_dataloader({
            "manifest_filepath": self.test_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": False,
        })
    
    def _process_mapudungun(self, split: str) -> list:
        print(f"Processing Mapudungun split: {split}...")
        dataset = load_dataset(self.map_dataset_name, split=split, streaming=self.streaming)
        wav_dir = os.path.join(self.out_data_dir, "map_wavs")
        os.makedirs(wav_dir, exist_ok=True)

        if self.max_examples is not None:
            dataset = dataset.take(self.max_examples) if self.streaming else dataset.select(range(min(len(dataset), self.max_examples)))

        entries = []
        try:
            total = self.max_examples if self.streaming else len(dataset)
        except TypeError:
            total = self.max_examples

        for idx, item in tqdm.tqdm(enumerate(dataset), total=total, desc=f"Mapudungun {split}"):
            audio_info = item.get("audio")
            if not audio_info:
                continue
            
            filepath = os.path.join(wav_dir, f"map_{split}_{idx}.wav")
            # Only write if it doesn't exist to speed up re-runs
            if not os.path.exists(filepath):
                sf.write(filepath, audio_info["array"], audio_info["sampling_rate"])
            duration = len(audio_info["array"]) / audio_info["sampling_rate"]

            text = item.get("spa", "")
            if not text:
                for k, v in item.items():
                    if isinstance(k, str) and (k.endswith("_es") or k.endswith("_spa")) and isinstance(v, str):
                        text = v
                        break
            
            if text:
                entries.append({
                    "audio_filepath": os.path.abspath(filepath),
                    "duration": duration,
                    "text": text,
                    "pnc": "yes",
                    "source_lang": "en", # Placeholder for Mapudungun
                    "target_lang": "es",
                })
        return entries

    def _process_quechua(self, split: str) -> list:
        print(f"Processing Quechua split: {split}...")
        
        # Determine which subdirectories map to this split
        subdirs = []
        if split == "train":
            subdirs = [
                "que_spa_unconstrained/train", 
                "que_spa_synthetic_translation/train"
            ]
        elif split == "validation":
            subdirs = ["que_spa_unconstrained/valid"]
        else:
            return [] # No test split defined locally for Quechua

        entries = []
        for subdir in subdirs:
            split_dir = os.path.join(self.que_data_dir, subdir)
            if not os.path.exists(split_dir):
                print(f"Warning: Quechua directory {split_dir} not found. Skipping.")
                continue

            # Figure out local split name ('train' or 'valid') for the .yaml/.spa files
            local_split = os.path.basename(subdir)
            
            yaml_path = os.path.join(split_dir, "txt", f"{local_split}.yaml")
            text_path = os.path.join(split_dir, "txt", f"{local_split}.spa")
            wav_dir = os.path.join(split_dir, "wav")

            if not os.path.exists(yaml_path) or not os.path.exists(text_path):
                print(f"Warning: Missing metadata/text files in {split_dir}")
                continue

            with open(yaml_path, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines()]

            for i, meta in tqdm.tqdm(enumerate(metadata), total=len(metadata), desc=f"Quechua {subdir}"):
                # Enforce total max_examples across ALL subdirectories for Quechua
                if self.max_examples and len(entries) >= self.max_examples:
                    break
                    
                wav_file = meta['wav']
                duration = meta['duration']
                text = texts[i]
                audio_path = os.path.abspath(os.path.join(wav_dir, wav_file))

                if os.path.exists(audio_path):
                    entries.append({
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": text,
                        "pnc": "yes",
                        "source_lang": "fr", # Placeholder for Quechua
                        "target_lang": "es",
                    })
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
            
            # Break outer loop if limit is reached
            if self.max_examples and len(entries) >= self.max_examples:
                break
                
        return entries

    def prepare_data(self):
        if not os.path.exists(self.out_data_dir):
            os.makedirs(self.out_data_dir)

        if not self.max_examples and os.path.exists(self.train_manifest) and os.path.exists(self.val_manifest):
            print(f"Manifests for mode '{self.language_mode}' found. Skipping data preparation.")
            return

        splits = ["train", "validation"] # TODO: Add test when available

        for split in splits:
            print(f"\n--- Preparing {split} split ---")
            
            map_entries = []
            que_entries = []
            
            # Extract based on language_mode flag
            if self.language_mode in ["map", "multi"]:
                map_entries = self._process_mapudungun(split)
            if self.language_mode in ["que", "multi"]:
                que_entries = self._process_quechua(split)
            
            # Interleave entries: 1 Mapudungun, 1 Quechua, 1 Mapudungun...
            combined_entries = []
            for map_entry, que_entry in itertools.zip_longest(map_entries, que_entries):
                if map_entry is not None:
                    combined_entries.append(map_entry)
                if que_entry is not None:
                    combined_entries.append(que_entry)
            
            if combined_entries:
                # Set path based on split name mapping
                if split == "train": 
                    path = self.train_manifest
                elif split == "validation": 
                    path = self.val_manifest
                else: 
                    path = self.test_manifest
                    
                write_manifest(path, combined_entries)
                print(f"Created manifest with {len(combined_entries)} total examples ({len(map_entries)} MAP, {len(que_entries)} QUE): {path}")
            else:
                print(f"No data found for split {split}. Skipping manifest creation.")

        print("Data preparation complete.")


# %%
data_loader = CanaryMultilingualDataModule(tokenizer=model.tokenizer, prompt_formatter=model.prompt, streaming=False, batch_size=4, language_mode="multi")

# %%
data_loader.prepare_data()

# %%
# !head -n 5 {data_loader.train_manifest}

# %%
# !head -n 5 {data_loader.val_manifest}

# %% [markdown]
# # Train Model
#
# Ya que han sido preparados los *adapters* es tiempo de entrenar los pesos de los mismos con los datos.

# %% [markdown]
# Actualizamos parámetros del optimizador

# %%
print(OmegaConf.to_yaml(model.cfg.optim))

# %%
# Setup optimization
model.cfg.optim.lr = 3e-4
model.cfg.optim.sched.warmup_steps = 25

# %% [markdown]
# Configuramos un entrenador lighting

# %%
trainer = L.Trainer(
    max_steps=200,
    accumulate_grad_batches=1,
    logger=False,
    enable_checkpointing=False,
    check_val_every_n_epoch=5,
)

# %%
trainer.fit(model, data_loader)

# %%
model.save_adapters("mapuche_adapters.pt")

# %%
from torchmetrics.text import SacreBLEUScore

sacrebleu = SacreBLEUScore(n_gram=4)
scores = []
preds = []
gts = []
for pred, gt in zip(ast_preds, ast_gt):
    preds.append(pred)
    gts.append([gt])

# bleu = sum(scores) / len(scores)
sacrebleu.update([p.text for p in preds], gts)
bleu = sacrebleu.compute()
print("BLEU", bleu.item() * 100)
