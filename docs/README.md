# EasyWebinar AI Avatar Generation Pipeline — Developer Documentation

Design docs for the serverless AI Avatar pipeline. A host onboards **once** (voice + a clean
studio reference); afterwards every request is `host_id` + script + aspect ratio → a talking-head
video in the host's voice and likeness, on [Modal.com](https://modal.com).

> **The pipeline was migrated to Wan2.2-S2V-14B** (audio-driven video diffusion) from the original
> EchoMimicV2 + face-restoration design. Docs `01` and `02` below are **current**; docs `03`–`08`
> and `FULL-SPEC.md` describe the **original** design and are kept for history — each is marked
> `LEGACY` at the top. For operations, see [`../OPS_RUNBOOK.md`](../OPS_RUNBOOK.md); for the top
> level, [`../README.md`](../README.md).

## How to read these docs

| File | Status | Contents |
|---|---|---|
| [01-architecture-overview.md](01-architecture-overview.md) | **current** | Two flows, end-to-end diagrams, data flow, environment split, speed/cost shape |
| [02-models.md](02-models.md) | **current** | Wan2.2-S2V + distill LoRA, ElevenLabs/Fish TTS, Groq director, InsightFace, nano-banana, rejected alternatives, roadmap |
| [03-business-logic.md](03-business-logic.md) | `LEGACY` | Original stage-chaining (crops, pose remap, compositing) — EchoMimicV2 era |
| [04-api-endpoints.md](04-api-endpoints.md) | partly current | HTTP endpoints; note the async `/generate` → `/status` → `/video` flow now (streamed, not base64) |
| [05-infrastructure-modal.md](05-infrastructure-modal.md) | mostly current | Modal Image/Volume/GPU/Secrets/scaling — image set differs (see doc 01) |
| [06-development-testing-protocol.md](06-development-testing-protocol.md) | `LEGACY` | Original phased build/test plan |
| [07-risks-and-open-questions.md](07-risks-and-open-questions.md) | partly current | Licensing + open items; some resolved, some still live (nano-banana billing, ToS) |
| [08-implementation-plan.md](08-implementation-plan.md) | `LEGACY` | Original Phase 0–12 roadmap |
| [FULL-SPEC.md](FULL-SPEC.md) | `LEGACY` | Concatenation of the original spec |

## Current architecture in one breath

Onboard once (voice clone or pre-made voice + a nano-banana studio reference) → per request:
**Groq** director tags the script → **ElevenLabs v3** or **Fish Audio s2.1-pro** TTS →
face-aware crop of the reference to the target aspect → **Wan2.2-S2V-14B** (4-step distill LoRA,
480p, single H100) animates it → full-quality audio muxed → streamed MP4. Async job:
`POST /generate` → poll `/status` → stream `/video`.

See [07-risks-and-open-questions.md](07-risks-and-open-questions.md) for what's still open, but
treat doc `01`/`02` and `OPS_RUNBOOK.md` as the source of truth for the current system.
