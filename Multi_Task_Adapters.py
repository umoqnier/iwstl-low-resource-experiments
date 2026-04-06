# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="b0373c4a-e565-4e8f-a87f-aae932d3aeed"
"""
You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.

Instructions for setting up Colab are as follows:
1. Open a new Python 3 notebook.
2. Import this notebook from GitHub (File -> Upload Notebook -> "GitHub" tab -> copy/paste GitHub URL)
3. Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)
4. Run this cell to set up dependencies.
5. Restart the runtime (Runtime -> Restart Runtime) for any upgraded packages to take effect


NOTE: User is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.
"""
# If you're using Google Colab and not running locally, run this cell.
import os

# Install dependencies
# !pip install wget
# !apt-get install sox libsndfile1 ffmpeg
# !pip install text-unidecode
# !pip install matplotlib>=3.3.2

## Install NeMo
BRANCH = "main"
# !python -m pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@$BRANCH"

# %%
BRANCH = "main"
# !uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@$BRANCH"

# %% [markdown] id="6c021f07-0576-491d-b73c-6c65c8501351"
# # Multi Task Adaptation with Adapters
#
#
# In earlier tutorials, we utilized a specific model for one task - for example, an ASR model (CTC, RNN-T etc) for the singular task of Speech Recognition. This is very useful if we want to specialize one task per model, but it can be expensive to deploy a fleet of models for each task, and learn routers to pass user tasks to correct models.
#
# We now support Multi Task models in NeMo, such that a single model can perform multiple tasks such as speech recognition, **speech translation**, voice activity detection, and more in the future. With one model supporting multiple tasks, we can simplify the task of deploying models and also hope to leverage individual tasks to improve each other (for example: you do need strong speech recognition first before you start doing translation).
#
# ---
#
# Multi Task (Canary) models are highly capable large neural networks capable of things like speech recognition, **X to English and English to X translation** and able to select whether to transcribe speech with punctuation and capitalization. These huge models are trained on several thousand hours of speech and text data, making it challenging to adapt to new datasets.
#
# In the previous tutorial for [ASR Adapters](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/asr_adapters/ASR_with_Adapters.ipynb), we used small adapter modules to tune a large ASR model on a small amount of data. In this tutorial, we will adapt a [Nvidia Canary](https://huggingface.co/nvidia/canary-1b) model onto a small amount of speech data **for both Automatic Speech Recognition (ASR) and Automatic Speech Translation (AST)**.
#
# In this tutorial, we will also demonstrate a simple way of creating custom **Data Modules from PyTorch Lightning** to design custom datasets and data loaders for the highly flexible Multi Task Models in NeMo ASR. This offers users more flexibility in designing new tasks, and finetuning the models on small amounts of data.

# %% [markdown] id="cbe2f8eb-204f-4d90-bb0a-a49d994f1ed7"
# ----
#
# First, lets instantiate the [Canary](https://huggingface.co/nvidia/canary-1b) model

# %% id="46c3e5c1-b4f2-4f84-89d6-c77bbe7ebe4f"
import os
import json

import nemo.collections.asr as nemo_asr

# %% id="48b9677b-b1d9-4361-becf-ee84fe8d53ca"
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b")

# %% [markdown] id="6c0c87c9-5290-4634-9338-818f181c936a"
# # Enable Adapter Support in Model
#
# New in NeMo 2.0, we now have a simple utility function to convert the model into one that supports adapters, called `replace_adapter_compatible_modules()`.
#
# This will go through the full model and check modules if they support adapters, and then enable that ability. Once used, you can freely use adapter methods.

# %% id="bfd72316-630b-43c3-9a02-65bb2dabe624"
model.replace_adapter_compatible_modules()

# %% [markdown] id="30505bd5-323f-4e90-a941-d0de3f6e55e3"
# ## Check Which Targets Are Supported For This Model
#
# Now that the model has enabled adapter support, lets take a look at which of its modules support adapter modules to be attached to them.
#
# **Note**
# Below, you might see an adapter module with no name `''` - this corresponds to the "default" model target if the target isn't specified. Users can chose to simply skip the module name when adding an adapter, and the model will by default add adapters to the encoder module.

# %% id="13bcf42e-d33a-4364-8d0f-ab59a26ffa7c"
model.adapter_module_names

# %% [markdown] id="67324f6a-ffff-47a7-9ee5-dc93819f6ffd"
# ## Prepare the Adapter
#
# Now that we know which modules are supported, lets create a simple adapter module for the encoder and decoder modules.

# %% id="65ec3b2b-3f84-43ed-8a90-085aee383ea6"
from nemo.collections.common.parts import LinearAdapterConfig

# %% id="47aab832-bfec-4cca-b4ee-868ea1af9869"
input_dim = model.cfg.encoder.d_model
adapter_dim = 8

# %% id="cd519281-ad45-4719-9ad6-561e6192717f"
enc_adapter_cfg = LinearAdapterConfig(in_features=input_dim, dim=adapter_dim)
dec_adapter_cfg = LinearAdapterConfig(in_features=input_dim, dim=adapter_dim)

# %% [markdown] id="f147fc89-ab93-4454-ad6b-909288a452a2"
# ## Add Adapter Modules
#
# Now that we have the adapter configs prepared, lets add them to the model !
#
# We provide the target module by using `target:adapter_name` when calling `add_adapter()` - this tells the model to setup an adapter called `adapter_name` to the module denoted by `target` with the config `cfg`.

# %% id="a23256ce-bc09-4fb0-8c3b-214519b8774b"
model.add_adapter(name="encoder:enc", cfg=enc_adapter_cfg)
model.add_adapter(name="transf_decoder:dec", cfg=dec_adapter_cfg)

print("Added adapters!")

# %% [markdown] id="2dbe9b7b-9a3d-4504-a652-1d90701cbbf8"
# ## Freeze Original Module Parameters and Unfreeze Adapter Weights Only
#
# When tuning adapters, we usually freeze the entire base model and only tune the adapters. This prevents the need for large amounts of data, preserves a lot of memory (since the full model doesnt need backward pass, only the adapters) and makes it easier to adapt huge models.

# %% id="2f8162dd-0373-4e65-aa8a-f458a1633578"
model.freeze()
model.unfreeze_enabled_adapters()

# %% [markdown] id="0b3795a4-fcfe-49ee-a76f-1cb77d99ace1"
# ----
#
# Lets make sure that the number of trainable parameters is a lot smaller (< 1 M) than the total number of params (1 B).

# %% id="58453f40-d72d-4f9b-a427-3fb63787f3d6"
model.summarize()

# %% [markdown] id="aa713f4a-ec16-4e2a-aeb3-ac7c4090f20f"
# ## Check Enabled Adapters
#
# Here, we check that the adapters that we named above (`enc` and `dec`) are both setup and enabled.

# %% id="d69f09d9-411e-420e-8f17-c86391e88fc3"
model.get_enabled_adapters()

# %% [markdown] id="f_XpTJx9hQXy"
# # Customizing Multi Task Models
#
# In the following section, we will take a deeper look into what are the components that compose a Multi Task Model and how users can override each of these parts to create their own customizable multi task models.
#
# ---
#
# In this tutorial, we will only see the internal components such as the prompt format and dataset construction, but not change them.
#
# In a following tutorial, we will show how to add an additional task to a pre-trained Multi Task Model using a pre-trained model as a starting point.

# %% [markdown] id="6f0beb8c-7b12-4169-a3f7-1639bdaf6160"
# # Prompt Handling for Multi Task Models
# Nvidia Canary is our first model that is a Multi Task Model.
#
# Multi Task models utilize a **prompt format**, similar to those used in Large Language Models, in order to denote to the model which task is to be performed, which langauge is being spoken and what language should the output transcript be in, whether to provide punctuation and capitalization or not, and so much more in the future !
#
# Lets take a look at the model's `prompt` for the Canary model that we have created -

# %% id="56a78cd0-afaf-4272-898f-d9e13ba871d3"
model.prompt_format

# %% [markdown] id="9cbaf28a-1f10-4da3-a3ed-53b2239baa49"
# ----
#
# This gives us the prompt format functions name, which we will see below points to a prompt format function that reads in manifest items and maps it to the template.

# %% [markdown] id="087d1f60-3679-4593-840f-8d0fbd8a0e3e"
# ## Reuse / Register a Prompt Format Function
#
# When we print `model.prompt_format` it writes `canary` which is one of the registered prompt templates available in NeMo ASR.
# For simplicity's sake, we will continue to use the same prompt format for this tutorial. However, we enable users to define their own prompt formats and register them as needed.
#
# Let's see what the `canary` prompt format looks like:

# %% id="c202abaf-63ca-4475-a2bb-3b487be8e375"
from nemo.collections.common.data.prompt_fn import (
    get_prompt_format_fn,
    registered_prompt_format_fn,
)
from nemo.collections.common.prompts import CanaryPromptFormatter, PromptFormatter

# %% id="07c56dc3-fe42-49fc-936c-770ec17a29ac"
# sample audio data
import numpy as np
import soundfile as sf
from io import BytesIO
from lhotse import Recording, SupervisionSegment, CutSet


def create_sine_wave(
    duration: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0
):
    """Generate a sine wave of specified duration and frequency."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    return np.sin(2 * np.pi * frequency * t)


audio = create_sine_wave()

# Convert to 16-bit PCM WAV format in memory
buffer = BytesIO()
sf.write(buffer, audio, 16000, format="WAV")
audio_bytes = buffer.getvalue()

# Create a Recording from the bytes
cut = Recording.from_bytes(data=audio_bytes, recording_id="generated_sine").to_cut()

cut.supervisions = [
    SupervisionSegment(
        cut.id,
        cut.recording.id,
        start=0,
        duration=cut.duration,
        text="I said something",
    )
]

# %% id="1c56dcaf-ac27-4e92-8f56-9b5a7daf0034"
canary_prompt_format_fn = get_prompt_format_fn(cut, CanaryPromptFormatter)
# canary_prompt_format_fn?

# %% [markdown] id="1170b57c-f4c7-432f-91bb-1dbf73063d60"
# ### Registering a New Prompt Format Function

# %% [markdown] id="d11a8a05-6ba7-41f3-97ab-43453a59c860"
# Just to show that this is user-configurable, we show how to register a dummy prompt format below:

# %% id="f77378ff-d5de-4b86-bfaf-e62b51c7f9ce"
from nemo.collections.common.prompts import PromptFormatter
from lhotse.cut import Cut


@registered_prompt_format_fn(Cut, PromptFormatter)
def canary_custom(example, formatter):
    """Users can implement this as needed"""
    raise NotImplementedError()


print("Registered prompt")

# %% id="cb02f068-8fee-46e1-8096-910062668173"
temp = get_prompt_format_fn(Cut, PromptFormatter)
temp.__name__

# %% [markdown] id="f14aa85b-71cb-4813-837b-b28a384685dc"
# ## Create / Reuse a Prompt Format
#
# Canary Multi Task Model comes with a pre-defined prompt template, so we need to **provide it data in a format that can be handled by that prompt format** class.
#
# A `PromptFormatter` is a special class that defines the dialog template of the order of turns that occur in a model's prompt. For example, in Language Models, we normally may begin with either a `System` or `User` turn, followed by an `Assistant` turn which produces an output from the model. Similarly in Multi Task models, we enable support for such a usage pattern.
#
# Do note: Current generation of Canary models are not trained to operate on **multi turn conversations (?)**, however future variants of Multi Task models may support such usage.

# %% id="35530cad-84d7-422b-82c5-1bda5c1a4497"
# Let's review the actual prompt formatter clas docs
# model.prompt?

# %% id="0cd0c0d1-da8a-4de6-9efc-86a7dd3ed660"
# Let's see the actual template of this prompt formatter
model.prompt.TEMPLATE


# %% [markdown] id="72956a2f-f051-42d2-9e08-47e954d88e5c"
# ---
#
# We see that the template contains two turns - `user` and `assistant`.
#
# User template looks as follows: `<|startoftranscript|>|source_lang||task||target_lang||pnc|`
# During execution, we remove the `|` in order to fill in the actual value of the slots provided by the the data loader.
#
# User holds the following allowed slots -
# * `source_lang`
# * `target_lang`
# * `task`
# * `pnc`
#
# Similarly, for Assistant template : `|text|<|endoftext|>`
#
# Assistant holds the following allowed slots -
# * `text`

# %% [markdown] id="540c04af-34d1-4b46-b935-40b16f54ca03"
# ### Creating and Using a Custom Prompt Formatter
#
# While we provide a pre-trained model with a pre-defined prompt format, we also enable users to create their own PromptFormatter subclass and change it as needed.
#
# Below, we show a simple modification to the model's PromptFormatter and show how to change it.

# %% id="0adb576c-df58-4b66-b8fa-8e653da6fead"
# Create a new prompt formatter using the original CanaryPromptFormatter class as baseclass
class CanaryPromptFormatterV2(model.prompt.__class__):
    # make sure to provide a new name
    NAME: str = "canary_custom"

    # Make any changes as necessary.
    # For this demonstration, we will not change anything other than the name


# %% id="f7d85683-ddd0-40c5-956d-e14d09243424"
# Next, lets update the model's prompt formatter
model.change_prompt("canary_custom")

# %% [markdown] id="6581f934-a55b-41df-864a-351d1fb0029e"
# ---
#
# We have now successfully changed the prompt format to `canary_custom`.
#
# **Note**: It is important to know that when changing the prompt format, the name of the new prompt format class (`canary_custom` in this case) **has to match** the name of the prompt function registered with `@registered_prompt_format_fn`!

# %% id="c1d84948-8f73-4c31-923f-eaf01d877835"
# Check if everything is ok -
model.prompt.__class__.__name__

# %% id="f617cda0-d16b-400a-b495-dac213d318e1"
model.prompt_format

# %% [markdown] id="cb964964-e978-43e9-befa-9bb0904db82f"
# ---
# For the rest of the tutorial, we will revert back to the original prompt formatter

# %% id="526093a8-86ba-48f0-a60b-55642720fc4e"
model.change_prompt("canary")

# %% [markdown] id="9c4d2986-89b4-4589-ab0e-69683084cfd4"
# ## Creating / Using a Multi Task Dataset
#
# Now that we have learned how to modify the model's prompt formatter and the underlying format function that maps manifest items into slots to inject into the prompt template, next let's take a look at how to use and create custom datasets for training multi task models.
#
# ---
#
# Unlike previous tutorials that showcase how to use pre-defined datasets and point them to your manifest files, we will take a slightly more hands-on approach for multi task modes. This is due to shear flexibility of multi task models - they can do almost any task that you can formulate into a "speech in - text out" problem.
#
# So it is not easy to have a pre-defined dataset class that can handle all new ideas and tasks that researchers can come up with.
#
# Instead, we showcase how to build a custom dataset for yourself and use it with the Multi Task model instead.

# %% [markdown] id="b35ca0c2-8ceb-423f-b9ef-7dd6ec5a6952"
# ---
#
# However, we also provide a base class that can be used as is by users if they dont want the hassle of writing their own datasets.
#
# This is handled by the `PromptedAudioToTextLhotseDataset` -  it maps user defined manifest items to the items defined in the prompt template of the model, so as long as the manifest corresponds to the slots supported by the model, it will be managed by the Dataset automatically.

# %% id="3d35d513-8538-4bcb-b892-898f16ad3f0f"
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
)

# Uncomment below line to see the class definition of PromptedAudioToTextLhotseDataset
# PromptedAudioToTextLhotseDataset?

# %% [markdown] id="51e3a150-40b9-4599-8c6e-0f01698989b4"
# ### Creating a New Prompted Dataset

# %% id="56208452-ea18-44c8-8c71-0daef431dc31"
import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
)


class MyCanaryPromptedAudioToTextLhotseDataset(torch.utils.data.Dataset):
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



# %% [markdown] id="5cb71ba1-ce2e-49c7-8126-be7e7851c812"
# ---
#
# The above class is mostly a demonstration, but it showcases how users might flexibly change the prompt formatter, prompt format function and even the data set that handles these two in a flexible way.
#
# The order of operations is usually this -
#
# 1) Create a new Prompt Formatter class - this denotes the slots that each turn can have (including new task inputs or other values). This class is auto registered.
# 2) Create a new Prompt Format function - Using `@registered_prompt_format_fn` decorator, write a custom function that accepts args and processes the provided input data from a manifest.
# 3) Create a new Dataset class (usually based on the `PromptedAudioToTextLhotseDataset` dataset) that uses the Prompt Format function to convert manifest items into nicely formatted samples that can be passed to the Prompt Formatter.

# %% [markdown] id="a7bf8078-663e-43cb-b045-0c8b6ef08e30"
# # Preparing a Canary Dataset
#
# Now that we have all the pieces together on the model side, let's take a look on the data side.

# %% [markdown] id="83c9eabc-0473-463e-be1f-ab6d5f519a79"
# ## Required Roles Defined by Prompt Format
#
# These are the available 'roles' available in the prompt format - they denote at each turn, one role can be enabled and its input or output can be calculated.

# %% id="11ff9641-53fd-4481-b414-0edc12bf4dc3"
model.prompt.get_roles()

# %% id="203a67e2-74fd-440c-9658-451f41239f36"
for role in model.prompt.get_roles():
    print(role, model.prompt.get_slots(role))
    print()

# %% [markdown] id="8e887f9d-94e7-4843-9da8-f914e24651f3"
# ## Create a Data Module
#
# Data Modules are one way of organizing datasets in PyTorch Lightning. It provides a unified place where data loading and processing can be potentially handled.
#
# **Note**: This isn't strictly necessary - you can achieve the same using just Pytorch dataloaders directly and passing it to Trainer.fit() but we showcase a data module codebase that can be extended by the user.

# %% [markdown] id="51d58931-4166-4ab9-a755-4c5268001192"
# ----
#
# In our CanaryAN4DataModule - we will perform two tasks. One is En ASR - transcribing the AN4 English dataset. Another is En to De AST - directly translating the english audio to German text.
#
# For simplicity's sake, we will use a small off-the-shelf model to perform the translation of English Transcripts to German.

# %% [markdown] id="91ed74ca-5d5e-412d-a813-0659014aa9a3"
# ---
#
# In NeMo 2.0, we utilize [Lhotse](https://github.com/lhotse-speech/lhotse) as our data backbone for speech tasks, which simplifies using custom speech datasets.
#
# Most of the magic is handled by the following code
#
# ```python
# from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
#
# get_lhotse_dataloader_from_config(
#     OmegaConf.create(config),  # Pass in a config that points to the manifest files and other arguments
#     global_rank=self.trainer.global_rank,
#     world_size=self.trainer.world_size,
#     # Pass in the dataset class for Lhotse to handle. This class now receives CutSet as input.
#     dataset=MyCanaryPromptedAudioToTextLhotseDataset(tokenizer=self.tokenizer, prompt=CanaryPromptFormatter(self.tokenizer)),
# )
# ```

# %% id="4a15ab9b-7603-4ac5-890c-92a541a0527c"
import os
import glob
import json
import copy
import subprocess
import tarfile
import wget
import librosa
import tqdm
from omegaconf import OmegaConf

from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as L

from transformers import T5Tokenizer, T5ForConditionalGeneration

from nemo.collections.asr.parts.utils.manifest_utils import (
    read_manifest,
    write_manifest,
)
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path, data_dir):
    with open(transcripts_path, "r") as fin:
        with open(manifest_path, "w") as fout:
            for line in fin:
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find("(") - 1].lower()
                transcript = transcript.replace("<s>", "").replace("</s>", "")
                transcript = transcript.strip()

                file_id = line[line.find("(") + 1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    data_dir,
                    wav_path,
                    file_id[file_id.find("-") + 1 : file_id.rfind("-")],
                    file_id + ".wav",
                )

                duration = librosa.core.get_duration(path=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript,
                    "pnc": "no",
                    "source_lang": "en",
                    "target_lang": "en",
                    "task": "asr",
                }
                json.dump(metadata, fout)
                fout.write("\n")

    return manifest_path


class CanaryAN4DataModule(L.LightningDataModule):
    def __init__(self, tokenizer, data_dir: str = "./an4/", batch_size=8):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size

        # ASR manifests
        self.train_manifest = os.path.join(data_dir, "an4/train_manifest.json")
        self.test_manifest = os.path.join(data_dir, "an4/test_manifest.json")

        # AST manifests
        self.ast_train_manifest = os.path.join(data_dir, "an4/ast_train_manifest.json")
        self.ast_test_manifest = os.path.join(data_dir, "an4/ast_test_manifest.json")

        # Combined manifests
        self.combined_train_manifest = os.path.join(
            data_dir, "an4/combined_train_manifest.json"
        )
        self.combined_test_manifest = os.path.join(
            data_dir, "an4/combined_test_manifest.json"
        )

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # Assign train/val datasets for use in dataloaders
        pass

    def train_dataloader(self):
        config = {
            "manifest_filepath": self.combined_train_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": True,
            "min_duration": 0.3,
            "max_duration": 10.0,
        }
        return self._setup_dataloader(config)

    def val_dataloader(self):
        config = {
            "manifest_filepath": self.combined_test_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": False,
            "min_duration": 0.3,
            "max_duration": 10.0,
        }
        return self._setup_dataloader(config)

    def test_dataloader(self):
        config = {
            "manifest_filepath": self.combined_test_manifest,
            "batch_size": self.batch_size,
            "num_workers": 4,
            "shuffle": False,
            "min_duration": 0.3,
            "max_duration": 10.0,
        }
        return self._setup_dataloader(config)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass

    def _setup_dataloader(self, config):
        """
        The main function that creates the data loader using Lhotse's integration with NeMo.
        """
        return get_lhotse_dataloader_from_config(
            OmegaConf.create(config),
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            # Note the passing of our custom dataset
            dataset=MyCanaryPromptedAudioToTextLhotseDataset(
                tokenizer=self.tokenizer, prompt=CanaryPromptFormatter(self.tokenizer)
            ),
        )

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        data_dir = self.data_dir
        if not os.path.exists(os.path.join(data_dir, "an4_sphere.tar.gz")):
            an4_url = (
                "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz"
            )
            an4_path = wget.download(an4_url, data_dir)
            print(f"Dataset downloaded at: {an4_path}")
        else:
            print("Tarfile already exists.")
            an4_path = os.path.join(data_dir, "an4_sphere.tar.gz")
        
        if not os.path.exists(os.path.join(data_dir, "an4/")):
            # Untar and convert .sph to .wav (using sox)
            tar = tarfile.open(an4_path)
            tar.extractall(path=data_dir)

            print("Converting .sph to .wav...")
            sph_list = glob.glob(
                os.path.join(data_dir, "an4/**/*.sph"), recursive=True
            )
            for sph_path in sph_list:
                wav_path = sph_path[:-4] + ".wav"
                cmd = ["sox", sph_path, wav_path]
                subprocess.run(cmd)
        print("Finished conversion.\n******")

        # Building Manifests
        print("******")
        train_transcripts = os.path.join(data_dir, "an4/etc/an4_train.transcription")
        train_manifest = self.train_manifest
        if not os.path.isfile(train_manifest):
            build_manifest(
                train_transcripts, train_manifest, "an4/wav/an4_clstk", data_dir
            )
            print("Training manifest created.")

        test_transcripts = os.path.join(data_dir, "an4/etc/an4_test.transcription")
        test_manifest = self.test_manifest
        if  not os.path.isfile(test_manifest):
            build_manifest(
                test_transcripts, test_manifest, "an4/wav/an4test_clstk", data_dir
            )
            print("Test manifest created.")
        print("*** Wrote manifests for Eng ***")

        train_manifest_data = read_manifest(self.train_manifest)
        test_manifest_data = read_manifest(self.test_manifest)

        if (
            not os.path.isfile(self.ast_train_manifest)
            or not os.path.isfile(self.ast_test_manifest)
            or not os.path.isfile(self.combined_train_manifest)
            or not os.path.isfile(self.combined_test_manifest)
        ):
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            t5_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

            if torch.cuda.is_available():
                t5_model = t5_model.cuda()

            def pipe(text):
                if isinstance(text, str):
                    text = [text]

                prefix = "translate English to German"
                prompts = [prefix + ": " + x for x in text]
                input_ids = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).input_ids
                input_ids = input_ids.to(t5_model.device)
                outputs = t5_model.generate(input_ids, max_new_tokens=64)
                return [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]

            ast_train_manifest_data = copy.deepcopy(train_manifest_data)
            ast_test_manifest_data = copy.deepcopy(test_manifest_data)

            print("Translating train set")
            train_texts = [x["text"] for x in train_manifest_data]
            BATCH_SIZE = 32

            for i in tqdm.tqdm(
                range(0, len(train_texts), BATCH_SIZE),
                total=len(train_texts) // BATCH_SIZE,
            ):
                batch_texts = train_texts[i : i + BATCH_SIZE]
                batch_texts = pipe(batch_texts)
                for j, text in enumerate(batch_texts):
                    ast_train_manifest_data[i + j]["text"] = text
                    ast_train_manifest_data[i + j]["task"] = "ast"
                    ast_train_manifest_data[i + j]["target_lang"] = "de"

            print("Translating test set")
            for data in tqdm.tqdm(
                ast_test_manifest_data, total=len(ast_test_manifest_data)
            ):
                data["text"] = pipe(data["text"])[0]
                data["task"] = "ast"
                data["target_lang"] = "de"

            write_manifest(self.ast_train_manifest, ast_train_manifest_data)
            write_manifest(self.ast_test_manifest, ast_test_manifest_data)

            print("*** Wrote ast manifests ***")

            combined_train, combined_test = [], []
            combined_train.extend(train_manifest_data)
            combined_train.extend(ast_train_manifest_data)

            combined_test.extend(test_manifest_data)
            combined_test.extend(ast_test_manifest_data)

            write_manifest(self.combined_train_manifest, combined_train)
            write_manifest(self.combined_test_manifest, combined_test)
            print("*** Wrote combined manifests ***")

        else:
            print("*** Wrote ast and combined manifests ***")



# %% [markdown] id="e06e697d-7dc2-489f-a52f-195946bfbf6e"
# ---
#
# Each item in the prepared manifest has the following items by default.
#
# As you will recognize, these are the same keys provided by the `CanaryPromptFormatter` classes `slots` argument, so each of these values in the is mapped back to those slots.
#
# ```python
# metadata = {
#     "audio_filepath": audio_path,
#     "duration": duration,
#     "text": transcript,
#     "pnc": "no",
#     "source_lang": "en",
#     "target_lang": "en",
#     "task": "asr",
# }
# ```
#
# The most important function in the Data Module above is `prepare_data()`:
#
# 1) It first downloads and converts the AN4 audio files to wav files.
# 2) Then it writes a new manifest file with the above keys for ASR task
# 3) It then translates the En transcripts with a `t5-small` model to generate German transcripts
# 4) Finally it writes another manifest for the AST task with these translated texts.
# 5) Finally it builds a combined manifest item for both ASR (en) and AST (en to de) multi-task training
#
# **Note**: We are using prepare_data() only for demonstration. Normally, users should process before experimentation, and so they would only need to implement methods above prepare_data() in their Data Module.

# %% [markdown] id="739f0141-1e0e-4db7-b1f6-9d13589bf50c"
# ## Download and Prepare Dataset

# %% id="323287f1-9a44-49ab-8438-dcbf34bf2ebe"
data_module = CanaryAN4DataModule(tokenizer=model.tokenizer, batch_size=16)

# %% id="123faf0d-05b2-4f12-850f-350a175ba7c1"
data_module.prepare_data()

# %% id="fbec085b-9600-49bd-8739-73e5e8e3773f"
# !head -n 5 {data_module.train_manifest}

# %% id="66bad9ac-3bad-4d84-8b30-830856c06804"
# !head -n 10 {data_module.ast_train_manifest}

# %% [markdown] id="cde19c46-e78c-4d7c-adbf-f1559c9203e1"
# # Evaluate Model before Training
#
# Canary Multi Task model is already very capable, achieving strong scores on multiple benchmarks. So we first evaluate the baseline numbers on the two tasks
#
# 1) ASR: WER calculation on transcripts
#
# 2) AST: SacreBLEU calculation on translations

# %% id="eb4588b4-7d52-4c4e-bb81-2bcb5a227afd"
from nemo.collections.asr.metrics.wer import word_error_rate
from torchmetrics.text import SacreBLEUScore

# %% id="a1c71044-3cb3-453c-bfcd-ee551cecdddf"
asr_test = read_manifest(data_module.test_manifest)
ast_test = read_manifest(data_module.ast_test_manifest)

# %% id="f1d8acd2-aa08-4ba0-b0c6-c5d662243b00"
asr_filepaths = [x["audio_filepath"] for x in asr_test]
asr_gt = [x["text"] for x in asr_test]

ast_filepaths = [x["audio_filepath"] for x in ast_test]
ast_gt = [x["text"] for x in ast_test]

print("Num files:", len(asr_filepaths))

# %%
torch.cuda.is_available()

# %% id="85ace700-97bf-4697-8e1a-5793eb21e678"
if torch.cuda.is_available():
    model = model.cuda()  # move model to gpu
    model = model.to(torch.bfloat16)  # cast full model to bfloat16

# %% id="00f2607a-2f67-47fe-9903-0adae4d9adf5"
asr_preds = model.transcribe(
    asr_filepaths,
    pnc="no",
    task="asr",
    source_lang="en",
    target_lang="en",
    batch_size=32,
)

# %% id="eea5ab20-60d4-4e19-87fb-71f6835941e8"
ast_preds = model.transcribe(
    ast_filepaths,
    pnc="no",
    task="ast",
    source_lang="en",
    target_lang="de",
    batch_size=32,
)

# %% id="69e5bb54-5193-4268-98e1-dc6daae8f6eb"
wer = word_error_rate([p.text for p in asr_preds], asr_gt)
print("WER", wer)

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

# %% [markdown] id="5ee530c9-36a3-47d2-83b9-b2a64080c0eb"
# # Train Model
#
# Finally, now that adapters have been prepared, model has been evaluated for a baseline and the dataset is prepared, it's time to train the adapter weights on the new datasets.
#
# ---
#
# First, we update the optimizer and scheduler config

# %% id="d0a40461-d739-436c-967a-1a0f8a3ad197"
print(OmegaConf.to_yaml(model.cfg.optim))

# %% id="4ba5811a-fc42-4de5-add5-0d26d1c84219"
# Setup optimization
model.cfg.optim.lr = 3e-4
model.cfg.optim.sched.warmup_steps = 25

# %% [markdown] id="d1de270a-d1cb-4080-b571-7acf365d7b99"
# ---
#
# Next, we setup a Lightning Trainer and Experiment Manager

# %% id="b9e34369-21ec-41bf-beae-30b60ab46c14"
from omegaconf import OmegaConf
from nemo.utils import exp_manager

# %% id="46f74863-a34d-4ad0-9d8e-3337ea5edd63"
trainer = L.Trainer(
    max_steps=200,
    accumulate_grad_batches=1,
    logger=False,
    enable_checkpointing=False,
    check_val_every_n_epoch=5,
)

# %% id="414d7887-bed5-46a2-bfe1-8349db1e6b5b"
# # Environment variable generally used for multi-node multi-gpu training.
# # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
# os.environ.pop('NEMO_EXPM_VERSION', None)

# config = exp_manager.ExpManagerConfig(
#     exp_dir=f'experiments/canary/',
#     name=f"Canary-Model-Adapter-Training",
#     checkpoint_callback_params=exp_manager.CallbackParams(
#         monitor="val_wer",
#         mode="min",
#         always_save_nemo=False,
#         save_best_model=False,
#     ),
# )

# config = OmegaConf.structured(config)

# logdir = exp_manager.exp_manager(trainer, config)

# %% [markdown] id="60769859-8ed5-4f9c-b93a-a6875c7c1c73"
# ---
#
# Begin training !

# %% id="2adb8607-a011-440d-bfa8-976c2871e8ef"
trainer.fit(model, data_module)

# %% [markdown] id="MImbKiqQ6ng-"
# ---
#
# Save just the adapter parameters - which is less than 2 MB !

# %% id="-akTdyGM6gum"
model.save_adapters("adapters.pt")
# !ls -l -- *.pt
# !du -sh *.pt

# %% [markdown] id="2525bec5-c42b-48c1-b03c-e8126c346238"
# # Evaluate after Adaptation
#
# Now that the model is done training, lets evaluate its scores on the test set again.
# We should see a markedly higher translation BLEU and lower WER from above.

# %% id="6edb5528-b1b6-4505-8cdc-ee68c715415e"
asr_test = read_manifest(data_module.test_manifest)
ast_test = read_manifest(data_module.ast_test_manifest)

# %% id="384aa5f2-89d5-4080-a717-4d65776fae6b"
asr_filepaths = [x["audio_filepath"] for x in asr_test]
asr_gt = [x["text"] for x in asr_test]

ast_filepaths = [x["audio_filepath"] for x in ast_test]
ast_gt = [x["text"] for x in ast_test]

print("Num files:", len(asr_filepaths))

# %% id="48ce5b4c-d349-4d86-ad3c-ee930bb569ee"
if torch.cuda.is_available():
    model = model.cuda()
    model = model.to(torch.bfloat16)

# %% id="49a37806-286e-4954-8f27-3829cf61d755"
asr_preds = model.transcribe(
    asr_filepaths,
    pnc="no",
    task="asr",
    source_lang="en",
    target_lang="en",
    batch_size=32,
)

# %% id="b701e014-2f71-487c-9300-a3ea89a43a45"
ast_preds = model.transcribe(
    ast_filepaths,
    pnc="no",
    task="ast",
    source_lang="en",
    target_lang="de",
    batch_size=32,
)

# %% id="087054e5-c511-4094-a115-faf4a3b49d51"
from nemo.collections.asr.metrics.wer import word_error_rate
from torchmetrics.text import SacreBLEUScore

# %% id="ef938f8f-b2db-45f6-9b30-4b3bbce2423f"
wer = word_error_rate([p.text for p in asr_preds], asr_gt)
print("WER", wer)

# %% id="5a7c2820-d394-4627-8438-0d810d89b72d"
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

# %% [markdown] id="521df0e6-1d3c-4709-a080-63638315c514"
# # Conclusion
#
# In this tutorial we added adapters to a Multi Task model (Nvidia Canary) and show how to create a custom dataset to finetune a canary model to a new dataset with previous tasks such as ASR and AST. The primary goal of this tutorial was to show how to flexibly adapt a Canary model to any of the pre-existing tasks.
#
# In a future tutorial, we will show how to add additional tasks to a pre-trained Canary, so that you can leverage the pre-trained encoder and decoder for your own custom tasks!
