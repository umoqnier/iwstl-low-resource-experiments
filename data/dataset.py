from typing import Protocol, list, tuple

import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.common.data.prompt_fn import get_prompt_format_fn
from nemo.collections.common.prompts import PromptFormatter
from torch.utils.data import Dataset

from custom_aumentation import augment


class TokenizerSpec(Protocol):
    pad_id: int


class MyCanaryPromptedAudioToTextLhotseDataset(Dataset):
    """Lhotse dataset with prompt formatting."""

    def __init__(self, tokenizer: TokenizerSpec, prompt: PromptFormatter):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.prompt_format_fn = get_prompt_format_fn(Cut, self.prompt)

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)
        audio_np = audio.numpy()
        augmented = [augment(samples=sample, sample_rate=16000) for sample in audio_np]
        audio = torch.from_numpy(np.stack(augmented))

        answers, prompts, prompts_with_answers = [], [], []
        for cut in cuts:
            fmt = self.prompt_format_fn(cut, self.prompt)
            answers.append(fmt["answer_ids"])
            prompts.append(fmt["context_ids"])
            prompts_with_answers.append(fmt["input_ids"])

        transcript, transcript_lens = self._collate_tokens(answers)
        p_with_a, p_with_a_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=p_with_a,
            prompted_transcript_lens=p_with_a_lens,
            cuts=cuts.drop_in_memory_data(),
        )

    def _collate_tokens(
        self, tokens: list[list[int] | torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = [torch.as_tensor(t) for t in tokens]
        lens = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
        padded = collate_vectors(tensors, padding_value=self.padding_value)
        return padded, lens
