# ASR + AST from scratch for nahuatl

## Findings

- **Canary-1Bv2 is not a special model class.** It's the generic `EncDecMultiTaskModel` (`nemo_speech/nemo/collections/asr/models/aed_multitask_models.py`) — FastConformer encoder (32×1024) + Transformer decoder (8×1024), `prompt_format: canary2`. Verified from the cached `.nemo` config.
- **The `en↔X` restriction is not in the code.** It lives in the checkpoint: Canary's SentencePiece vocab contains only Whisper-style **2-letter** ISO lang tokens (`<|es|>`, `<|qu|>`, 273 of them) — I checked: `<|azz|>` and `<|spa|>` are `<unk>` there — plus the training data itself. Nothing in NeMo forces English.
- **The from-scratch recipe already exists**: `nemo_speech/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml` + `nemo_speech/examples/asr/speech_multitask/speech_to_text_aed.py`. It builds exactly this architecture from a config, with a configurable special-token vocabulary (`spl_tokens`), Lhotse dataloading, and multitask metrics (WER when `source_lang==target_lang`, BLEU otherwise — built into the YAML).
- Your `data/datamodule.py` / `data/dataset.py` are already generic: they take **any** tokenizer + prompt formatter (`model.prompt`), so they survive the model swap unchanged. Your pain is visible in the manifests: AST entries literally say `"source_lang": "en"` for Nahuatl audio.

## Data (datasets/nahuatl, verified on disk)

- **ASR pool (~70 h):** all 954 Cuetzalan recordings (ISO `azz`) have time-aligned Nahuatl transcriptions. ELAN files come in two quality tiers (see `Deposit-Nahuatl-for-ASR-Community.md`):
  - **Set 1** = 299 files, proofed by Amith, mostly translated into Spanish → the only gold AST data.
  - **Set 2** = 439 files, first-draft transcriptions only → silver ASR, **not usable for AST**.
- **AST pool (~50 h):** Set 1 segments that actually have a `Traducción` tier — `EAFParser.get_segments_with_tag` in `data/processors.py` already extracts it.
- Deposit total is ~165 h audio (957 WAVs, 48 kHz **stereo** — dual-mic sessions); your ~120 h is the transcribed/translated subset. `audio_chunks/` already holds ~65k mono chunks cut on ELAN annotation boundaries (still 48 kHz; resampling is left to the dataloader — resample at chunk-build time later if workers become the bottleneck).
- **Tepetzintla (151 recordings, `nhi` dialect): zero transcriptions → excluded from training.** Keep as out-of-distribution eval; it has different phonology (λ vs t, implosive ɓ).
- Splits are already **recording-level with a leakage assertion** (`data/processors.py:344`) — keep that; with only ~3–4 speakers, a speaker-level split is even safer for val/test.
- The old manifests (10 h AST + 4 h ASR) are a small subset — everything below runs on the full corpus.

## Plan

### 1. Tokenizer — `build_tokenizers.py` (new, small)
Train one SentencePiece BPE over azz+spa text (`scripts/tokenizers/process_asr_text_tokenizer.py` or direct sentencepiece) with the special tokens passed as `--user_defined_symbols`: the canary2 template set (`<|startoftranscript|>`, `<|emo:*|>`, `<|pnc|>`, `<|itn|>`, `<|timestamp|>`, `<|diarize|>`, `<|nospeech|>`, `<pad>`, `<unk>`) **plus `<|azz|>` and `<|spa|>`**. Load it as `CanaryBPETokenizer` — that's exactly what Canary v2 does (single 16k SPE with specials baked in; no aggregate tokenizer needed since both languages are Latin-script). Vocab **8k–16k** given ~120 h of text (Canary v2 itself uses 16k).

### 2. Manifests — patch `data/processors.py` + regenerate at full scale
- Replace placeholder values: AST → `source_lang: "azz", target_lang: "spa"`, text = Spanish; ASR → `azz→azz`, text = Nahuatl. Pick one code convention (`spa`, not `es`) and keep it in sync with step 1. `lang_field: "target_lang"` in your datamodule already does the right thing.
- Run the ELAN→chunk pipeline over **both** ELAN dirs: ASR pool = Set 1 + Set 2 segments (same audio appearing as both an ASR and an AST sample is fine and desirable — that's how Canary ships multitask data); AST pool = translated Set 1 segments only.
- **Val/test are built only from Set 1 translations** (gold); never evaluate BLEU on Set 2-derived data.
- Sanity-check the dual-mic handling: confirm the chunker takes the channel that matches the annotated speaker before trusting ~40% of the corpus.

### 3. Config — `conf/azz_ast_aed.yaml`
Copy `fast-conformer_aed.yaml`, then:
- Decoder: **8 layers / 1024** (Canary-v2 dims) or 4 for your 12 GB box; encoder 32×1024 to match the warm-start source.
- `tokenizer: {type: bpe, dir: <from step 1>}` with `custom_tokenizer: CanaryBPETokenizer`, `prompt_format: canary2`.
- Optim: full-training values from the recipe (adamw, ~2e-4, InverseSquareRoot, warmup) — not the 1e-3 adapter LR.
- Keep `multitask_metrics_cfg` (WER + BLEU split by lang constraint); checkpoint monitor becomes `val_bleu` or `val_loss`, not `val_wer`.

### 4. Training — `train_scratch.py`
Clone `train.py`'s skeleton (it's good: click CLI, duration batching, logging, resume, callbacks) and swap the model section:
- Construct `EncDecMultiTaskModel(cfg.model)` instead of `from_pretrained` + adapters; drop all adapter code.
- **Warm-start the encoder from canary-1b-v2** (the one piece of pretraining that transfers across languages, and the recipe comment itself says it's "required to stabilize" at 1B scale): load the HF model, `new.encoder.load_state_dict(old.encoder.state_store_dict())` filtered to `encoder.*`, discard the decoder/embeddings (English-centric — exactly what you're escaping).
- Optional but recommended first stage: freeze encoder ~1–2k steps while the random decoder learns to attend, then unfreeze at lower LR. One boolean, not a framework.
- Reuse `CanaryMultilingualDataModule` unchanged, but extend it to accept a **list of manifests with weights** (lhotse `input_cfg` mixture): blend ~70 h ASR + ~50 h AST so **AST is ~50% of batches** — a random-init decoder gets drowned by the ASR stream otherwise, and the ASR stream is what teaches the azz input side.
- Budget: ~20–40k steps (the recipe's 100k default is for 100k+ h corpora); val every ~1k steps.
- With full training, budget `--batch-duration` well below the adapter table in the README.

### 5. Eval — adapt `canary_eval_script.py`
Drop adapter-path loading; load the whole model from Lightning ckpt / `.nemo`. SacreBLEU part stays.

### Order of work
1. `build_tokenizers.py` → sanity: encode one azz + one spa sentence, check lang tokens resolve.
2. Manifest patch + regenerate **full scale** (parse both ELAN sets, record-level split already in place).
3. YAML + `train_scratch.py` → 20-step smoke run (flash-sized dims, CPU ok).
4. Real training; compare against your adapter `en`-placeholder baseline as the control.

**Ponytail notes** — skipped on purpose: no custom model class, no new prompt formatter (canary2 template accepts arbitrary lang tokens), no speech-LLM/Parakeet decoder path, no per-language aggregate tokenizer, no resumable/indexed dataloader migration (`.claude/skills/migrate-to-resumable-dataloader` exists if you ever need it). **Risk picture at ~120 h:** from-scratch decoder + warm encoder is now a defensible mainline bet, not a gamble — but 50 h of *parallel* data is still the ceiling on BLEU, and Spanish generation is learned entirely from those 50 h. Keep the adapter `en`-placeholder baseline as the control. If BLEU plateaus early, the biggest lever is **data, not architecture**: the 439 untranslated Set-2 files are the natural next translation target (the deposit says native-speaker translations were planned for them). Fallback if the 8-layer decoder underfits nothing: shrink it, not the data. I did not check that `CanaryBPETokenizer` + `canary2` prompt-fn end-to-end handles a single-SPE tokenizer with custom lang tokens without the `PROMPT_LANGUAGE_SLOT` quirk — first smoke run will expose it; fix is ~5 lines in the prompt formatter call if it bites.
