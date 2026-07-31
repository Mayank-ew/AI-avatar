# Models

> **Current stack (Wan2.2-S2V).** Replaces the original EchoMimicV2 + RestoreFormer++/GFPGAN +
> GPT-4o design (see `LEGACY` docs for history). Every model here is commercial-use safe.

## The stack

| Role | Model | Notes |
|---|---|---|
| **Avatar generation** | **Wan2.2-S2V-14B** (Alibaba, Apache-2.0) | Audio-driven talking-head video diffusion. Chosen for full-generation quality. `wan.WanS2V.generate(ref_image, audio, ...)`. |
| **Speed** | **Wan2.1-T2V-14B cfg-step-distill LoRA** (lightx2v, Apache-2.0) | 4-step inference (vs ~40). S2V rides the 2.1 dense-14B backbone, so this LoRA drops in — same model, ~4× faster. |
| **TTS — A** | **ElevenLabs `eleven_v3`** (Instant Voice Clone) | Tag-driven expressive delivery; `eleven_multilingual_v2` fallback for untagged text. |
| **TTS — B** | **Fish Audio `s2.1-pro`** | A/B challenger; switchable per request. JSON API, model in a header, voice via `reference_id`. Ignores tags → clean text. |
| **Audio director** | **Groq `openai/gpt-oss-120b`** | Inserts `eleven_v3` tags without changing wording (content-preservation guardrail). Skipped for Fish. |
| **Face / frames** | **InsightFace `buffalo_l`** | Best-frame selection at onboarding (EAR/frontality/brightness gates) + face box for aspect cropping. |
| **Studio reference** | **Gemini 2.5 Flash Image ("nano-banana")** | Identity-preserving re-staging of one frame into a clean studio portrait. |

## Wan2.2-S2V-14B — the generator

- **Input:** a reference image + an audio track → MP4 with plausible head/face motion and
  lip-sync. Text prompt describes the *visual scene* (motion comes from audio).
- **Architecture fact that matters:** S2V-14B is a **dense** model on the **Wan2.1 dense-14B
  backbone** (note "14B", not the 2.2 "A14B" MoE). This is why the *2.1* distill LoRA fits and
  the *2.2* MoE distill LoRAs do **not**.
- **Chunking:** long clips are built as sequential ~5 s chunks (`num_repeat = ceil(audio/5)`),
  each conditioned on the previous via `motion_latents` → **autoregressive, not parallelizable**.
- **Knobs** (`constants.py`): `WAN_MAX_AREA` (480p), `WAN_SAMPLE_STEPS`/`WAN_DISTILL_STEPS`,
  `WAN_INFER_FRAMES=80`, `WAN_OFFLOAD_MODEL=False`, `WAN_CONVERT_DTYPE=True` (bf16),
  `WAN_MAX_SECONDS` (duration/cost cap).

## The 4-step distill LoRA (speed)

- Repo: `lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v`,
  file `loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors` (~631 MB).
- Merged into `WanS2V.noise_model` (the DiT) at container load via a format-tolerant loader
  (`generate_stage._load_lora_into_model`). Runs 4 steps with CFG off (`guide=1.0`).
- **Cross-task caveat:** it's a *text-to-video* LoRA applied to the *audio-driven* model — it may
  need strength tuning (`WAN_DISTILL_LORA_STRENGTH`, start 1.0, up to ~1.5) and can slightly
  soften lip-sync. Toggle off with `WAN_USE_DISTILL=False` to fall back to the stock model.
- Diagnostic on cold start: `Wan distill: LoRA merged into N modules` — N must be large.

## TTS providers

- Selected per request via `tts_provider` (`elevenlabs` | `fishaudio`) + optional overrides;
  the panel exposes a dropdown. `tts.synthesize(..., provider=, fish_reference_id=)` dispatches.
- **ElevenLabs:** the director tags the script; v3 for tagged text. Voice = `profile.voice_id`.
- **Fish Audio:** `POST https://api.fish.audio/v1/tts`, JSON body, `model` header
  (`s2.1-pro-free`), voice via `reference_id` (distinct from an ElevenLabs voice_id).

## Reference Studio (onboarding)

`reference_studio.py`: sample frames → InsightFace picks the clearest frontal, eyes-open,
well-lit face (progressive gate relaxation) → **nano-banana** re-stages it into a clean studio
portrait (identity-preserving; removes mic/clutter). Runs once per host, not per generation.

- **nano-banana needs paid Gemini API billing** — currently blocked (`limit:0`). Fallback: make
  the portrait by hand in the Gemini app and register it (`set_reference`). `crop_reference_to_ratio`
  face-aware crops any reference to an exact aspect, reusing the same crop the generator uses.

## What was evaluated and rejected

| Model | Verdict |
|---|---|
| **EchoMimicV2** | Rejected — identity morphing, waxy faces (the original design; removed). |
| **MuseTalk / LatentSync** | Rejected — mouth-only lip-sync = visible quality step-down vs full generation. |
| **daVinci-MagiHuman / LTX-Video** | Rejected — generate their *own* audio; can't speak the host's voice. |
| **HunyuanVideo-Avatar** | Rejected — same heavy-DiT slowness *and* license excludes EU/UK/South Korea. |
| **Memory snapshots (GPU alpha & CPU two-phase)** | Rejected — GPU alpha doesn't speed storage load; CPU two-phase crashes on Wan's conv3d. Fresh single-GPU load used instead. |

## Roadmap — action/motion replication

Current S2V *synthesizes* motion from audio (doesn't copy a person's actual gestures). To
replicate a real subject's movements (e.g. recreate a webinar host doing her actual actions),
the path is **Wan2.2-Animate-14B** (motion transfer: reference character + driving video), or
Wan-S2V's `pose_video` input. Not yet integrated — flagged as roadmap.
