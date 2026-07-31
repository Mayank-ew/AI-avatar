# Architecture Overview

> **Current architecture (Wan2.2-S2V).** This replaces the original EchoMimicV2 + restorer design
> (see `LEGACY`-bannered docs 03–08 / FULL-SPEC for history).

## Feature overview — two flows

1. **Host onboarding (once per host).** A short clip (or an existing webinar) produces a stored
   **host profile**: a voice (an ElevenLabs Instant Voice Clone, *or* a pre-made ElevenLabs/Fish
   voice assigned via override) + a clean **studio reference portrait**. Stored as
   `host_id → {voice_id, reference_image_path, references_by_ratio, ...}`.
2. **Generation (every request).** Caller sends `host_id` + `script_text` + `target_ratio`
   (+ optional TTS provider / voice overrides). No upload, no per-request voice data — the
   profile supplies everything. Because Wan takes minutes, this is an **async job**.

Entirely serverless on Modal — no persistent process; each request triggers a chain of Modal
Function calls, thin CPU ones (director, TTS, orchestration) and one heavy GPU one (Wan-S2V).

## Onboarding (once per host)

```mermaid
flowchart TD
    O0[Short clip / webinar URL + consent] --> O1{consent attested?}
    O1 -- no --> OR[Reject]
    O1 -- yes --> O2["ffmpeg: extract audio"]
    O2 --> O3{clone voice?}
    O3 -- "IVC" --> O4["ElevenLabs /v1/voices/add → voice_id"]
    O3 -- "--voice-id override" --> O5["reuse a pre-made ElevenLabs/Fish voice"]
    O0 --> O6["Reference Studio: pick best frame (InsightFace)\n→ reimagine into clean studio portrait\n(nano-banana, or set by hand)"]
    O4 --> O7[Write host profile]
    O5 --> O7
    O6 --> O7
    O7 --> O8["profile: {voice_id, reference_image_path, references_by_ratio}"]
```

## Generation (async, per request)

```mermaid
flowchart TD
    G0["POST /generate {host_id, script_text, target_ratio, tts_provider?}"] --> G1[Profile lookup]

    subgraph THIN["director_tts_image — CPU, fast (blocking part of /generate)"]
      G1 --> S0["Stage 0 — Audio Director (Groq gpt-oss-120b)\ninserts eleven_v3 tags; SKIPPED for Fish"]
      S0 --> SA["Stage A — TTS\nElevenLabs v3  OR  Fish Audio s2.1-pro"]
      SA --> AW["write audio.mp3 to intermediate volume"]
    end

    AW --> SP["spawn Function A → return job_id (202)"]

    subgraph WAN["wan_image — H100 GPU (async job)"]
      SP --> B1["Stage B — pick per-aspect reference\n+ face-aware crop to target_ratio"]
      B1 --> C1["Stage C — Wan2.2-S2V-14B\n(4-step distill LoRA, 480p, offload off)"]
      C1 --> MX["mux FULL-QUALITY audio (audio.mp3)\n→ final_output.mp4"]
    end

    MX --> P["GET /status → done + video_url"]
    P --> V["GET /video → streamed MP4 (binary)"]
```

## Data-flow table

| After stage | Artifact | Notes |
|---|---|---|
| Profile lookup | `{voice_id, reference_image_path, references_by_ratio}` | one read per request, keyed by `host_id` |
| Stage 0 | `{clean_text, directed_text, stability_hint}` | ElevenLabs only; Fish uses `clean_text` |
| Stage A | `audio.mp3` | ElevenLabs v3 (tagged) / multilingual_v2, or Fish s2.1-pro |
| Stage B | reshaped reference PNG | reference cropped to `target_ratio`, centered on the face |
| Stage C | `final_output.mp4` (video) | Wan-S2V, 480p, `num_repeat = ceil(audio_secs/5)` sequential chunks |
| mux | `final_output.mp4` (final) | 44.1 kHz original audio muxed in (not the 16 kHz lip-sync track) |

## Environment split (why the images differ)

- **onboarding_image / director_tts_image (CPU)** — API calls only (ffmpeg, ElevenLabs, Fish,
  Groq, store). No GPU.
- **wan_image (H100 GPU)** — the entire animation now lives in **one** GPU function
  (`WanAvatar.run`). Wan2.2-S2V does audio-driven generation end-to-end, so there is **no
  Function B / restorer / temporal-blend / composite** (all removed in the migration).
- **studio_image (CPU)** — InsightFace best-frame + nano-banana reimagine + `crop_reference_to_ratio`.
  Kept off `wan_image` so it builds fast and needs no torch/CUDA.

The thin stages finish synchronously inside `POST /generate` (director + TTS are seconds); then
Function A is **spawned** and the caller polls. Function A reads `audio.mp3` and the reference off
the shared intermediate Volume under a per-request `run_id` (== `job_id`) and writes
`final_output.mp4` back to it, which `/video` streams.

## Speed & cost shape

Total GPU time ≈ `num_repeat × steps × pixels`. `num_repeat = ceil(audio_secs / 5)` (chunks are
autoregressive — each conditions on the previous, so they can't be parallelized). Levers applied:
**480p** (`WAN_MAX_AREA`), **4-step distill LoRA** (vs 40), **offload off**, **bf16**. Result:
~4 min for a 30 s clip warm. Cold start ≈ 2.5 min (fresh GPU build; snapshots are incompatible
with Wan's conv3d — see OPS_RUNBOOK). `WAN_MAX_SECONDS` caps duration as a cost guard.
