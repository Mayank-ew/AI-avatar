> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# Risks and Open Questions

This file exists so a research gap never gets silently treated as a confirmed fact elsewhere
in these docs. Anything listed here is genuinely open — re-verify before shipping, don't
assume it's resolved because it isn't mentioned as a caveat somewhere else.

## Lead risk (legal/product): ElevenLabs ToS on commercial redistribution of a self-cloned voice — UNVERIFIED

The entire host-onboarding feature exists so a host can post generated videos to their own
social media "without any issue" (the user's own framing of the requirement). Research into
ElevenLabs' actual Terms of Service could not confirm their position on a user commercially
redistributing content made with their own self-cloned (IVC) voice — only summary/legal-
commentary third-party sites were found, not the primary ToS text itself. **This is the legal
premise the whole feature's value proposition rests on, and it has not been verified.** Before
shipping this feature to real hosts, read ElevenLabs' live Terms of Service directly (not a
summary) and confirm: (a) self-cloning your own voice for commercial use is permitted, (b) no
additional licensing/attribution requirement applies to the generated audio, and (c) whether
`eleven_v3`'s tag-annotated, potentially more "produced-sounding" delivery changes anything
about how ElevenLabs' ToS treats the output. Do not treat this as resolved by the presence of
the consent-checkbox step at onboarding ([03-business-logic.md](03-business-logic.md) §0) —
that satisfies ElevenLabs' API-level cloning-consent requirement, which is a narrower and
separate question from the ToS's position on commercial redistribution of the output.

> **POC note:** if the POC is an internal/stakeholder demo and its generated videos are never
> actually posted to a real host's public social media, this question's urgency is lower for
> the POC itself — the pipeline can be demonstrated end-to-end without it being resolved.
> It remains a hard blocker before any real host's generated content ships publicly in
> production, regardless of how well the POC demo goes.

## Onboarding-related risks (host profile, voice cloning, storage)

- **Storing biometric-adjacent host data** (a voice clone plus the likeness/video it was
  derived from) raises privacy/compliance questions — retention policy, deletion-on-request,
  and consent record-keeping all need legal review before this handles real hosts' data. Same
  posture as every other risk in this file: open, not resolved by writing it down.
- **Voice-slot/quota scaling risk.** This design creates one ElevenLabs IVC voice per
  onboarded host. ElevenLabs gates custom voices per workspace by plan tier — confirmed numbers
  exist for Professional Voice Clone slots specifically (0 on Free/Starter, 1 on Creator/Pro, up
  to 10 on Business); IVC-specific numbers were not directly confirmed, but the same slot-pool
  concept likely applies. **Host count growth directly consumes this quota** — plan for either a
  higher ElevenLabs account tier or an archival/deletion strategy for inactive hosts' voices
  well before host count reaches the tens, not after hitting a quota error in production.
  > **POC note:** a single dev/demo ElevenLabs account with no archival strategy is fine for a
  > POC with a handful of demo hosts — this only becomes a real constraint at production
  > multi-tenant scale.
- **`requires_verification` field's downstream effect is unconfirmed.** The IVC creation
  response includes this boolean; whether a flagged `voice_id` is usable immediately or blocked
  pending manual/automated review was not confirmed by research. Test empirically with a real
  account (see [06-development-testing-protocol.md](06-development-testing-protocol.md) Phase 3
  item 0) before assuming either behavior.
- **No confirmed voice-expiration/deletion-on-inactivity policy.** Document
  persistence-until-explicitly-deleted as a working assumption, not a confirmed ElevenLabs
  guarantee.
- **Video-to-audio extraction is a design assumption, not a confirmed API behavior.** No source
  confirms `POST /v1/voices/add` rejects a video file outright, only that no source confirms it
  accepts one. The ffmpeg-extraction step in the onboarding flow is built on the safer
  assumption; confirm with one real test upload (audio file vs. attempting a raw video upload)
  before relying on it in production, per
  [06-development-testing-protocol.md](06-development-testing-protocol.md).
- **Re-onboarding's effect on the old ElevenLabs voice_id is an implementation decision, not
  yet made concrete in code.** [04-api-endpoints.md](04-api-endpoints.md)'s `PUT
  /host/{host_id}/onboard` recommends explicitly deleting the superseded `voice_id` to avoid
  silently consuming quota — this needs to actually be implemented, not just documented as a
  recommendation.

## Top risk (technical): custom temporal-blending quality is unproven

Since no permissively-licensed model does video-native, temporally-consistent face
restoration (see the licensing sweep below), the flicker-fix for Stage D is original in-house
code (optical-flow-guided blending, [03-business-logic.md](03-business-logic.md) §6) with no
reference implementation to benchmark visual quality against. This is the highest-uncertainty
part of the whole pipeline. Recommendation: prototype and visually validate it on a short clip
early (see [06-development-testing-protocol.md](06-development-testing-protocol.md) Phase 3)
before committing to it as the shipped approach — do not discover flicker/artifact problems for
the first time during full end-to-end integration.

## Why BFVR-STC was dropped entirely (historical record)

`github.com/Dixin-Lab/BFVR-STC` has **no `LICENSE` file** (confirmed via GitHub API,
`license: null`). Its own README states its code is "mainly modified from CodeFormer," which
is licensed under the **S-Lab License 1.0** — explicitly non-commercial ("redistribution and
use for non-commercial purpose... are permitted"; commercial use requires contacting the
contributors). Absent its own license, default copyright law applies (all rights reserved),
and the CodeFormer lineage adds an explicit non-commercial restriction on top. This is not
usable in a commercial SaaS product.

A dedicated follow-up sweep checked every other model actually built for video-temporal-
consistent face restoration and found the same problem across the board:

| Model | License (verified from actual LICENSE file) | Commercial-safe? |
|---|---|---|
| BFVR-STC | None — inherits CodeFormer's S-Lab 1.0 by lineage | No |
| CodeFormer | S-Lab License 1.0 (non-commercial) | No |
| KEEP | S-Lab License 1.0 (non-commercial, same NTU lab) | No |
| PGTFormer | Xidian University non-commercial license | No |
| DicFace | No LICENSE file at all | No |
| SVFR | No real LICENSE file (README claim of "MIT" unverified against an actual file); weights explicitly non-commercial; built on Stability AI's revenue-capped SVD | No |
| DVFace | Genuine MIT license | Yes, but **no code/weights released yet** — watch-list only |
| RestoreFormer++ | Apache-2.0 (confirmed) | Yes — **chosen as primary restorer** |
| GFPGAN ("clean" arch) | Apache-2.0, avoiding a bundled CC-BY-NC-SA DFDNet component | Yes — **chosen as fallback** |
| Real-ESRGAN | BSD-3-Clause | Yes, but general-purpose, not face-specific |

**Conclusion**: no permissively-licensed model handles video-native face restoration natively.
RestoreFormer++/GFPGAN (single-image, permissive) + a custom temporal-blend layer (original
code, no derivative-work exposure) is the least-bad available option, not a proven equivalent
to BFVR-STC's approach.

## GFPGAN's bundled NVIDIA Source License clause

GFPGAN's repo is Apache-2.0 overall, but bundles the original StyleGAN2-derived CUDA ops under
an NVIDIA Source License (non-commercial) as one architecture option. The "clean"
(StyleGAN2Clean) architecture used here (v1.3/v1.4) avoids that code path — this is why
[02-models.md](02-models.md) specifies "clean" architecture explicitly rather than bare
"GFPGAN." Get a legal sanity-check on this nuance before shipping; it's high-confidence but not
independently re-verified line-by-line against the weight files actually deployed.

## Half-body reference-crop derivation needs empirical tuning

The `expansion_ratio`/`vertical_bias` constants in
[03-business-logic.md](03-business-logic.md) §3a are starting guesses, not validated numbers.
A wrong expansion ratio degrades EchoMimicV2's animation quality in a way that's easy to
misdiagnose as "the model is just bad" rather than "the input framing was wrong." Test against
a range of real base-video framings (already half-body vs. tightly face-framed) early.

## Background color/luminance mismatch on the outer composite

EchoMimicV2's output is diffusion-generated; its background pixels won't pixel-match
Master_Background's real photographic environment even when sourced from the same footage.
Spatial edge feathering does not fix this — color/luminance matching
([03-business-logic.md](03-business-logic.md) §7) is a separate, required step and needs its
own visual validation pass, not an assumption that it "just works."

## The onboarding-time restorer-selection check is itself new and unvalidated

Superseding an earlier version of this risk item (which called for enforcing a per-clip
restorer decision in code): the design has since moved to deciding `preferred_restorer` **once
per host, at onboarding time** (see [02-models.md](02-models.md) §4a and
[03-business-logic.md](03-business-logic.md) §0), stored in the host profile and read, unchanged,
by every future generation. This structurally eliminates the risk of per-frame or per-clip
restorer mixing — Stage D has no fallback/switching logic left to misuse.

What's still genuinely open: **the automated reliability check that makes this decision is new,
unvalidated machinery** — it's only as good as the heuristic used to judge whether
RestoreFormer++'s face-detection/alignment on one sample frame is "reliable enough." Risks:
- A single sample frame might not be representative of the whole onboarding video (e.g., a
  video that starts well-lit/front-facing but has poor framing later on) — the check could
  green-light a restorer that then struggles on other frames from the same host.
- The reliability threshold itself (what counts as "detected/aligned reliably") is a judgment
  call that needs real tuning against a diverse set of test videos, not a guess (see
  [06-development-testing-protocol.md](06-development-testing-protocol.md) Phase 5 of the
  implementation plan for the validation step this needs).
- The manual-override escape hatch (`preferred_restorer_override` at onboarding) is the safety
  net if the automated check misfires — but only if someone actually notices a bad result and
  re-onboards with the override; there's no automatic re-evaluation if a host's generations
  start looking worse over time.

## ElevenLabs `eleven_v3` GA-transition ambiguities

`eleven_v3` moved from alpha/research-preview to General Availability around February 2026.
Several details conflict across sources from that transition and need re-verification against
the live API before finalizing implementation:
- Character limit: 5,000 (widely cited) vs. a third-party report of 3,000 post-GA — **conflicting**.
- Whether `voice_settings.speed` is supported on `eleven_v3` — **conflicting sources**.
- Current Professional Voice Clone (PVC) support quality on v3 — the "use an Instant Voice
  Clone instead" guidance predates the GA announcement and may no longer apply.
- Exact numeric `stability` values behind the Creative/Natural/Robust UI-labeled presets —
  ElevenLabs documents these as named presets, not raw numbers; any numeric mapping used in
  code is a hypothesis to validate empirically, not a documented fact.
- Whether sending v3-tagged text to `eleven_multilingual_v2`/`eleven_flash_v2_5` (e.g. as a
  fallback path) results in tags being spoken literally or silently stripped — untested.

## Modal.com API — remaining unverified details

- Exact 401/429/5xx error body shapes from ElevenLabs (only the 422 shape is confirmed from
  the live OpenAPI spec).
- Whether `GET /v1/voices` still exists as a legacy alias alongside the current
  `GET /v2/voices` (a direct fetch to `/v1/voices` 404'd during research).
- Whether any specific Modal GPU type is region-limited (no explicit documentation found;
  only that requesting >2 GPUs per container increases allocation wait time).
- The `@modal.concurrent(max_inputs=N)` vs. a possibly-superseding `single_use_containers`
  flag for per-container concurrent-input handling — internally inconsistent across sources at
  time of research; **re-check against the installed `modal` package version and its
  changelog before writing this into real code.**
- Exact interaction between Modal's `timeout=` parameter and billing (safe assumption used in
  these docs: billing meters actual container-seconds, `timeout` is a pure safety ceiling —
  not independently confirmed against Modal's billing docs).

## RestoreFormer++/GFPGAN VRAM and performance on Modal GPUs

No VRAM or per-frame timing figures were found in either repo (unlike EchoMimicV2, which has
community-reported numbers). Expected to be much lighter than EchoMimicV2's diffusion pipeline
given the feedforward GAN architecture, but this is an inference, not a documented fact —
benchmark on the actual target Modal GPU tier before finalizing Function B's GPU choice or any
per-request cost estimate.

## GPT-4o pricing/latency figures

The cost/latency figures in [02-models.md](02-models.md) are sourced from third-party
aggregators, not OpenAI's own pricing page directly — reverify against `openai.com/api/pricing`
before finalizing any cost model.
