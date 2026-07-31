> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# Phase-wise Implementation Plan

This is the execution companion to the architecture docs (`01`-`07`) — it turns "what the
system is" into "what order to build it in and how to know each piece actually works." No code
exists yet; this plan starts from an empty repo.

**Sequencing principle**: the genuinely new/unproven pieces of this design (custom temporal
blending, the LLM-director tag pipeline, the onboarding-time restorer-reliability check,
EchoMimicV2's and the restoration stack's fragile dependency environments, the
ffmpeg-vs-video IVC-upload assumption) are de-risked as isolated, cheap spikes **before** they
get wired into a real end-to-end request — not discovered for the first time during full
integration. Phases 1-6 are spikes; Phases 7-11 are integration; Phase 12 is hardening/sign-off.
This deliberately does not build Stage B→C→D→E in a straight line.

---

## Phase 0 — Project Scaffolding & Modal Environment Bring-Up

**Objective**: an empty repo becomes a deployable Modal project skeleton — `modal.App` exists,
secrets are provisioned, and a "hello world" deploys/runs successfully. Nothing avatar-specific
yet.

**Prerequisites**: none.

**Tasks**:
- Initialize repo structure: `app.py` (Modal entrypoint), per-stage modules (`director.py`,
  `tts.py`, `store.py`, `onboard.py`, `prep.py`, `generate_stage.py`, `restore.py`,
  `composite.py`, `main_generate.py` — named per [05-infrastructure-modal.md](05-infrastructure-modal.md)'s
  function layout), `requirements/` per image, `README.md`.
- Create the Modal workspace, run `modal setup`/`modal token new`.
- Define the four `modal.Image` builds from [05-infrastructure-modal.md](05-infrastructure-modal.md)
  §1 as near-empty stubs first (base image + core deps only, no heavy stacks yet):
  `onboarding_image`, `director_tts_image`, `echomimic_image`, `restoration_image`.
- Create the two `modal.Volume`s: `avatar-weights-vol`, `host-media-vol`.
- Create placeholder `modal.Secret`s: `elevenlabs-secret`, `openai-secret`,
  `postgres-conn-string` (per the POC storage note in
  [05-infrastructure-modal.md](05-infrastructure-modal.md) §0), `proxy-auth-token`.
- Write one trivial health-check `@app.function` per image (returns `{"ok": True, "image":
  "<name>"}`) proving each image builds and runs, including GPU allocation for the two GPU
  images.

**Acceptance criteria**: `modal deploy` succeeds for all four images; every health-check
function returns its expected dict via `modal run`; the two GPU health checks confirm
`torch.cuda.is_available()` (or equivalent) actually reports a GPU.

**POC vs Prod**: N/A — identical scaffolding either way.

---

## Phase 1 — Spike: LLM Audio Director → `eleven_v3` Tag Pipeline (Stage 0 + Stage A, standalone)

**Objective**: prove the cheapest, fastest-to-validate risky assumption first — that GPT-4o
reliably produces tag-annotated `directed_text` that (a) passes the content-preservation
guardrail and (b) measurably improves ElevenLabs `eleven_v3` delivery — with zero video/GPU
dependency.

**Prerequisites**: Phase 0 (`director_tts_image`, `openai-secret`/`elevenlabs-secret`).

**Tasks**:
- `director.py`: `direct_script(script_text, voice_character_hint=None) -> {clean_text,
  directed_text, stability_hint}` via GPT-4o structured output, using the closed tag
  vocabulary from [02-models.md](02-models.md) §1.
- `verify_content_preserved(clean_text, directed_text) -> bool`: strip tags/caps/punctuation,
  compare; bounded retry (e.g. max 2) on mismatch, then hard-reject
  (`director_guardrail_failed`), per [03-business-logic.md](03-business-logic.md) §1.
- Character-limit check against `eleven_v3`'s documented limit before sending (flagging the
  5,000-vs-3,000 ambiguity from [07-risks-and-open-questions.md](07-risks-and-open-questions.md)
  — confirm the real current limit here, empirically, as part of this phase).
- `tts.py`: `synthesize(voice_id, directed_text, stability_hint) -> bytes` calling `POST
  /v1/text-to-speech/{voice_id}` with `model_id=eleven_v3`; implement the
  `enable_director=false` fallback path using `eleven_multilingual_v2` + `clean_text`.
- Use any pre-existing/throwaway ElevenLabs voice for this spike — host onboarding doesn't
  exist yet (Phase 8), this is deliberately decoupled.
- A small CLI/`modal run` test harness: raw script text in, mp3 saved to a Volume for manual
  listening.

**Acceptance criteria**: run 5-10 diverse sample scripts through the pipeline; confirm the
guardrail never false-rejects fine output and never lets mismatched content through; confirm
resulting audio is audibly more expressive than an untagged baseline; confirm the
`multilingual_v2` fallback path works. Document the final tag vocabulary that survived vs. got
dropped as noisy/unreliable.

**POC vs Prod**: manual listening + a small bounded retry count is the POC bar; prod would want
automated audio-quality scoring and a larger tag-vocabulary regression suite.

---

## Phase 2 — Spike: Confirm the ElevenLabs IVC Video-vs-Audio Assumption

**Objective**: settle, empirically, whether `POST /v1/voices/add` truly requires audio (not
video) before building the real onboarding endpoint around that assumption — flagged as
UNVERIFIED in [07-risks-and-open-questions.md](07-risks-and-open-questions.md).

**Prerequisites**: Phase 0 (`onboarding_image` with ffmpeg, `elevenlabs-secret`).

**Tasks**:
- `extract_audio(video_path, out_path)` via ffmpeg subprocess call.
- A one-off test: attempt `POST /v1/voices/add` with the raw video file directly once (to
  settle the question), then retry with the ffmpeg-extracted audio; capture both responses.
- Record the decision this produces: is audio-extraction strictly required, or just defensive?
  (Keep it either way for consistency — but know which.)

**Acceptance criteria**: one real short test video, one real dev ElevenLabs account, both code
paths exercised against the live API; the resulting `voice_id`/`requires_verification`
captured; the extraction-required-vs-defensive question answered and written down (feeds
directly into Phase 8's implementation).

**POC vs Prod**: uses the single dev ElevenLabs account (per the POC quota note in
[07-risks-and-open-questions.md](07-risks-and-open-questions.md)); no mechanism divergence
between POC and prod here — this is a one-time factual confirmation.

---

## Phase 3 — Spike: EchoMimicV2 Environment Bring-Up (weights + minimal inference)

**Objective**: prove the heaviest, most fragile dependency stack (torch 2.5.1+cu124, diffusers
0.31.0, xformers 0.0.28.post3, InsightFace, EchoMimicV2Pipeline) actually installs and runs on
Modal GPU hardware, using EchoMimicV2's own bundled sample assets — not the real crop/pose
pipeline yet.

**Prerequisites**: Phase 0 (`avatar-weights-vol`, `echomimic_image` stub).

**Tasks**:
- Finalize `echomimic_image` per [05-infrastructure-modal.md](05-infrastructure-modal.md) §1:
  exact pinned versions, git-cloned `echomimic_v2` repo, static ffmpeg-4.4 binary +
  `FFMPEG_PATH`.
- Implement `download_weights()` exactly as specified in
  [05-infrastructure-modal.md](05-infrastructure-modal.md) §2 (`BadToBest/EchoMimicV2`,
  `stabilityai/sd-vae-ft-mse`, `lambda/sd-image-variations-diffusers` with
  `allow_patterns=["unet/*"]`, raw-URL Whisper `tiny.pt`); run once via `modal run`; `.commit()`
  the Volume.
- A minimal smoke-test function loading `EchoMimicV2Pipeline` from the downloaded weights,
  running inference against the upstream repo's own demo reference image/audio/pose —
  deliberately bypassing Phase 9's custom crop/pose-remap logic.
- Confirm GPU VRAM footprint; pick a concrete tier (`A10`/`L40S`/`A100`) for Function A.

**Acceptance criteria**: `download_weights()` completes; assert all expected files exist at
documented paths/sizes (per [02-models.md](02-models.md)'s weight tables — a truncated download
is a common, hard-to-diagnose failure mode); the smoke test produces a playable output video
from EchoMimicV2's own sample assets. Record actual VRAM usage and wall-clock time.

**POC vs Prod**: `min_containers=0` accepted here and for every GPU function going forward (per
the POC note in [05-infrastructure-modal.md](05-infrastructure-modal.md) §5) — cold starts are
fine for a POC demo; prod would set `min_containers=1`+ once real traffic is known.

---

## Phase 4 — Spike: Restoration Models Environment Bring-Up (RestoreFormer++ / GFPGAN)

**Objective**: prove the second fragile dependency stack (opencv/basicsr/facexlib/gfpgan +
git-cloned RestoreFormerPlusPlus, deliberately isolated from Function A's torch/diffusers pins)
installs and runs `enhance()` correctly.

**Prerequisites**: Phase 0 (`restoration_image` stub, `avatar-weights-vol`).

**Tasks**:
- Finalize `restoration_image` per [05-infrastructure-modal.md](05-infrastructure-modal.md) §1:
  separate torch install (no `diffusers`), git-cloned `RestoreFormerPlusPlus`, `gfpgan`
  package (v1.4 "clean" architecture — see [02-models.md](02-models.md) §4a for why "clean"
  specifically).
- Extend `download_weights()` (or add a sibling function) to fetch RestoreFormer++/GFPGAN
  weights — **confirm the actual hosting mechanism here** (HF `snapshot_download` vs. direct
  release-asset download; do not assume, per
  [07-risks-and-open-questions.md](07-risks-and-open-questions.md)) and implement whichever
  applies.
- A smoke test running both restorers' `enhance(frame, has_aligned=False, paste_back=True)`
  directly against sample face frames — confirming face-detection + FFHQ alignment +
  inverse-affine paste-back work with no hand-rolled cropping (per
  [02-models.md](02-models.md) §4a).
- Benchmark GPU tier candidates (`T4`/`L4`/`A10`) for per-frame wall-clock time on both
  restorers.

**Acceptance criteria**: both restorers process sample frames end-to-end, visibly sharper than
an unrestored baseline; both images build independently with no dependency conflicts (proving
the Function A/B split actually resolves the conflict documented in
[02-models.md](02-models.md)); GPU benchmark numbers recorded.

**POC vs Prod**: weight-hosting mechanism decision is POC-pragmatic (whatever's fastest to
confirm and get working); GPU tier can start conservative (`T4`) for POC cost savings.

---

## Phase 5 — Spike: `choose_preferred_restorer()` — the Onboarding-Time Decision Mechanism

**Objective**: validate the mechanism that replaced the original "per-clip restorer selection"
risk (see [07-risks-and-open-questions.md](07-risks-and-open-questions.md)) — an automated,
onboarding-time, per-host decision of `"restoreformer++"` vs. `"gfpgan"`, stored once in the
host profile and never re-decided at generation time.

**Prerequisites**: Phase 4 (restoration image + weights working).

**Tasks**:
- Implement `choose_preferred_restorer(sample_frame) -> "restoreformer++" | "gfpgan"` in
  `onboard.py`: extract one representative frame (via ffmpeg at a fixed timestamp), run
  RestoreFormer++'s detection/alignment step only (not full `enhance()`), apply an explicit,
  concrete reliability rule (e.g., face detected above a confidence/bbox-size threshold, with
  alignment landmarks within expected bounds) to decide pass (→ `"restoreformer++"`) or fail
  (→ `"gfpgan"`).
- Implement the manual-override path: honor a `preferred_restorer_override` parameter if
  supplied (per [04-api-endpoints.md](04-api-endpoints.md)'s onboarding schema), bypassing the
  automated check entirely.
- Test against a deliberately diverse sample set (lighting, framing, occlusion, angle) —
  record how often the automated pick matches a manual visual-quality judgment call on the
  same frame.

**Acceptance criteria**: run against 5-8 varied sample frames; manually verify each automated
decision against visual inspection of both restorers' output on that frame; document the
chosen threshold values and known failure modes (this feeds directly into
[07-risks-and-open-questions.md](07-risks-and-open-questions.md)'s "onboarding-time
restorer-selection check is itself new and unvalidated" item — this phase is how that risk
gets addressed, not eliminated). This function's signature is now frozen for Phase 8.

**POC vs Prod**: best-effort heuristic + manual-override escape hatch is the POC bar; prod
might later replace the heuristic with a more robust classifier once real failure modes are
observed in production traffic.

---

## Phase 6 — Spike: Custom Temporal-Blending Prototype (fully isolated)

**Objective**: de-risk the single most novel piece of code in the system — the optical-flow
temporal-blending pass — using canned restored-frame sequences, before it's ever wired into
Function B.

**Prerequisites**: Phase 4 (real restored frames available as test input). Can run in parallel
with Phase 5.

**Tasks**:
- Assemble a short test sequence (24-48 frames) of restored 768×768 frames exhibiting visible
  flicker (produce by running Phase 4's restorer across a real short clip's frames).
- Implement (in a standalone harness first, not yet inside `restore.py`): `estimate_flow`
  (e.g. OpenCV Farneback), `warp_and_blend` (3-5 frame sliding window), eye/mouth-region
  up-weighting via the face-landmark masks the alignment step already produces — per
  [03-business-logic.md](03-business-logic.md) §6.
- Build a visual-validation harness: before/after side-by-side rendering, plus a crude
  frame-to-frame pixel-diff-variance metric in eye/mouth regions as a quantitative proxy.
- Explicitly stress-test: fast head motion, blinking, mouth open/close transitions (highest
  smearing-artifact risk from over-aggressive blending).

**Acceptance criteria**: measurable flicker reduction on the test sequence without introducing
visible smearing/ghosting during motion, confirmed by human visual sign-off on at least one
representative clip — not just the crude metric improving. Iterate on window size/weighting
until this bar is met; do not proceed to Phase 10 until it is.

**POC vs Prod**: manual visual sign-off is the POC bar (per the "unproven" flag already in
[07-risks-and-open-questions.md](07-risks-and-open-questions.md)); prod would want broader
coverage across many real host videos and possibly a learned perceptual-quality metric instead
of the crude pixel-diff proxy.

---

## Phase 7 — Metadata Store & Storage Layer (pluggable `ProfileStore`)

**Objective**: a working storage abstraction — host profiles can be written/read/updated
against the real POC Postgres instance, behind a clean interface so the eventual prod swap
(per [05-infrastructure-modal.md](05-infrastructure-modal.md) §0's POC note) never touches
calling code.

**Prerequisites**: Phase 0. Can run in parallel with Phases 1-6.

**Tasks**:
- `store.py`: define `HostProfile` (`host_id, voice_id, base_video_path, preferred_restorer,
  created_at`) and a `ProfileStore` interface: `get(host_id) -> HostProfile | None`,
  `put(host_id, profile) -> None` (upsert — re-onboarding must overwrite, not duplicate),
  `delete(host_id)`.
- `PostgresProfileStore`: concrete implementation using the `postgres-conn-string` Secret;
  write the table DDL (`CREATE TABLE IF NOT EXISTS host_profiles (host_id PRIMARY KEY,
  voice_id, base_video_path, preferred_restorer, created_at)`).
- Provision the actual small hosted Postgres/MySQL instance (per the POC note) and wire the
  real connection string into the Secret.
- Tests: create → fetch by `host_id` → overwrite via re-onboarding (confirm upsert, no
  duplicate row) → fetch a nonexistent `host_id` (confirm a clean not-found signal, not an
  unhandled exception — this is what Phase 11's `profile_lookup` error keys off of).

**Acceptance criteria**: all tests pass against the real hosted instance (not mocked);
re-onboarding confirmed to overwrite in place; nonexistent-host lookup returns a clean signal.

**POC vs Prod**: POC = small hosted Postgres via `modal.Secret`, single profile per host, old
`voice_id` left orphaned on re-onboard (see [03-business-logic.md](03-business-logic.md) §0's
POC note). Prod = swap to `modal.Dict` or a properly pooled/scaled managed DB behind the same
`ProfileStore` interface, and implement the explicit old-`voice_id` deletion on re-onboard.

---

## Phase 8 — Onboarding Endpoint (Stage -1), Fully Integrated

**Objective**: `POST`/`PUT /host/{host_id}/onboard` is a real, deployed endpoint performing the
full onboarding flow, using the now-proven pieces from Phases 2, 5, and 7.

**Prerequisites**: Phases 2, 5, 7.

**Tasks**:
- `onboard.py`: `@app.function(image=onboarding_image)` +
  `@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)` `host_onboard(host_id,
  video, consent_attested, preferred_restorer_override=None)` per
  [05-infrastructure-modal.md](05-infrastructure-modal.md) §4's pseudocode.
- Consent gate as the very first check: reject immediately (no processing at all) if
  `consent_attested` is false/absent.
- Wire in order: ffmpeg audio extraction (Phase 2's confirmed approach) → `POST
  /v1/voices/add` → write video to `host-media-vol` at
  `/host_media/{host_id}/onboarding_video.mp4` → `choose_preferred_restorer()` (Phase 5) against
  a sample frame from the just-uploaded video, or honor the override → `ProfileStore.put()`
  (Phase 7) with the complete `HostProfile`.
- Implement `PUT /host/{host_id}/onboard` re-onboarding reusing the same handler, confirming
  upsert semantics.
- Wire the shared bearer-token proxy auth (POC: single token, per
  [05-infrastructure-modal.md](05-infrastructure-modal.md) §4's POC note).
- Response: `{status, host_id, voice_id, requires_verification, preferred_restorer}` per
  [04-api-endpoints.md](04-api-endpoints.md).

**Acceptance criteria**: (a) a request with missing/false `consent_attested` is rejected
immediately with no ElevenLabs call and no writes; (b) a valid real test video produces a real
`voice_id`, a video landing in the Volume at the correct path, and a fully-populated Postgres
row including a non-null `preferred_restorer`; (c) re-onboarding the same `host_id` overwrites
the row (confirmed single row, not duplicated).

**POC vs Prod**: single shared bearer token (prod: per-tenant); orphaned old `voice_id` on
re-onboard left as-is (prod: explicit deletion).

---

## Phase 9 — Function A: Canvas/Face/Pose Prep + EchoMimicV2 (Stages B + C, real inputs)

**Objective**: given a real onboarded `base_video_path` and a real Phase-1-produced audio file,
Function A produces real half-body animated frames — where the empirically-tuned crop
constants finally get validated against real host videos, not sample assets.

**Prerequisites**: Phase 3, Phase 8 (real onboarded videos to test against), Phase 1 (real
audio as test input).

**Tasks**:
- `prep.py` (Stage B), per [03-business-logic.md](03-business-logic.md) §2-3:
  - `crop_pad_to_ratio(video_path, target_ratio) -> Master_Background` (even width/height
    enforced).
  - `select_face_bbox(frame, insightface_results)`: largest-bbox-by-area rule; explicit
    zero-face (reject, clear error) and multi-face (deterministic pick) handling.
  - `expand_to_halfbody_crop(face_bbox, expansion_ratio, vertical_bias) -> reference_crop`:
    tunable constants as module-level config (not hardcoded inline), square-pad, resize to
    768×768.
  - Record the Master_Background paste-back bbox separately (used only in Phase 10).
  - `extract_pose(base_video_path)` (self-driven, per
    [03-business-logic.md](03-business-logic.md) §4) + `remap_pose_to_crop_space(...)`.
- Empirically tune `expansion_ratio`/`vertical_bias` against several real onboarded test videos
  with different framings — this was flagged as needing empirical tuning in
  [07-risks-and-open-questions.md](07-risks-and-open-questions.md); iterate until half-body
  framing looks right across the test set.
- `generate_stage.py` (Stage C): wire `reference_crop` + audio + `remapped_pose` into
  `EchoMimicV2Pipeline`, capped at 240 frames/768×768.
- Combine into one Function A Modal function (shares the GPU container).

**Acceptance criteria**: run against 3-4 real onboarded test videos with varying framing;
visually confirm the reference crop consistently produces a sensible chest-up framing (not a
tight face crop, not badly off-center); confirm zero-face/multi-face edge cases produce clear
distinct errors; confirm playable 768×768 output with lip movement roughly matched to the
audio.

**POC vs Prod**: no storage/auth divergence; GPU tier from Phase 3's benchmarks,
`min_containers=0`.

---

## Phase 10 — Function B: Restoration + Temporal Blend + Composite + Mux (Stages D + E, real inputs)

**Objective**: given Function A's raw frames, Function B produces the final muxed,
aspect-ratio-correct `.mp4` — integrating Phases 4, 5, and 6, plus the new
compositing/color-matching logic that hasn't been built until now.

**Prerequisites**: Phases 4, 5, 6, 9.

**Tasks**:
- `restore.py` (Stage D): read `preferred_restorer` from the `HostProfile`
  (`ProfileStore.get()`); call the corresponding restorer's `enhance(frame,
  has_aligned=False, paste_back=True)` on each full 768×768 frame directly — no hand-rolled
  crop/resize, per [02-models.md](02-models.md) §4a.
- Port Phase 6's temporal-blend prototype into `restore.py`'s real frame-sequence pipeline (3-5
  frame sliding window, eye/mouth up-weighting via the alignment step's landmark masks).
- `composite.py` (Stage E): paste each de-flickered frame into the recorded paste-back bbox
  (from Phase 9); implement spatial edge feathering AND color/luminance matching (histogram
  matching or Poisson blending — pick one concretely, per
  [03-business-logic.md](03-business-logic.md) §7's note that feathering alone is
  insufficient).
- ffmpeg mux: `-c:v libx264 -c:a aac -pix_fmt yuv420p -aspect {target_ratio} -shortest` →
  final `.mp4`.
- Combine Stage D + E into one Function B Modal function on the GPU tier benchmarked in
  Phase 4.

**Acceptance criteria**: feed Phase 9's real test-video outputs through Function B; visually
confirm (a) restored frames are sharper than raw EchoMimicV2 output, (b) flicker is visibly
reduced vs. an ablation run with temporal-blending disabled, (c) no visible seam/color mismatch
at the composite boundary, (d) final `.mp4` plays with correctly synced audio at the requested
aspect ratio.

**POC vs Prod**: GPU tier chosen conservatively from Phase 4 (`T4`/`L4` candidate),
`min_containers=0`.

---

## Phase 11 — Full `/generate` Endpoint: End-to-End Orchestration

**Objective**: `POST /generate` orchestrates profile lookup → Stage 0 → Stage A → Function A →
Function B and returns the final video — the first point the *entire* system runs as one real
request.

**Prerequisites**: all prior phases.

**Tasks**:
- `main_generate.py`: `@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)`
  `generate(payload)` per [05-infrastructure-modal.md](05-infrastructure-modal.md) §4's
  pseudocode, accepting `{host_id, script_text, target_ratio, voice_character_hint?,
  enable_director?, elevenlabs_model_override?, restorer_override?}`.
- Profile lookup first: `ProfileStore.get(host_id)`; on miss, fail fast with
  `{stage: "profile_lookup", ...}` before any downstream work, per
  [04-api-endpoints.md](04-api-endpoints.md).
- `restorer = payload.get("restorer_override") or profile["preferred_restorer"]` — confirm this
  is the *only* place a request-level override can affect restorer choice; Stage D itself has
  none.
- Orchestrate the call chain across the three runtime environments (director_tts → echomimic →
  restoration), deciding and documenting the concrete inter-function data-passing mechanism
  (Modal function-call chaining vs. shared Volume paths for intermediate artifacts) — the
  architecture docs don't fully pin this down, so make the decision explicit here.
- Structured per-stage error handling (`director`, `tts`, `face_detection`, `echomimic`,
  `restoration`, `compositing`, `mux`) so any failure identifies its stage, not a generic 500.
- Decide the final response mechanism concretely (video bytes vs. a signed/Volume-backed URL).

**Acceptance criteria**: a full request against a real onboarded `host_id` and a real script
produces a final playable video — correct lip-sync, correct aspect ratio, restored/de-flickered
quality, expressive audio — with no manual intervention between stages. Deliberately test one
failure path (unknown `host_id`) and confirm the structured `profile_lookup` error, not a stack
trace.

**POC vs Prod**: single shared bearer token; `min_containers=0` across all functions (cold-start
latency accepted); no per-tenant isolation.

---

## Phase 12 — Multi-Host Regression & POC Sign-off

**Objective**: validated across multiple distinct hosts and scripts, known edge cases handled
gracefully, POC considered demo-ready — and the explicit list of what must change before
production is written down in one place.

**Prerequisites**: Phase 11.

**Tasks**:
- Onboard 3-5 distinct test hosts with meaningfully different video framing/lighting/voice
  characteristics through the real Phase 8 endpoint.
- Run `generate()` against each with a range of scripts (varying length, varying
  `voice_character_hint`, at least one near the `eleven_v3` character limit, `target_ratio`
  values requiring both crop and pad).
- Re-test edge cases deliberately: zero-face/multi-face frames, director-guardrail
  rejection/retry, unknown `host_id`, missing consent; confirm both `preferred_restorer` values
  get exercised across the test hosts (or force via Phase 5's manual override if the automated
  check doesn't naturally produce both).
- Record cold-start times and total per-request latency across both GPU functions; decide if
  acceptable for a live demo or whether to temporarily bump `min_containers=1` for a demo
  session.
- Write a short internal ops runbook (not a `docs/` file — an ops document): re-running
  `download_weights()`, rotating the shared bearer token, onboarding a new demo host live,
  known limitations to state during a demo.

**Acceptance criteria**: all test hosts produce acceptable-quality videos on first/second
attempt; every deliberately-tested edge case fails gracefully with a structured, stage-
identified error; both restorer code paths exercised at least once; the environment is
reproducible from scratch (empty Volumes → `download_weights()` → onboard → generate).

**POC vs Prod**: this phase is the formal "POC accepted" gate. The full list of POC-lenient
shortcuts carried forward — small hosted Postgres storage, single shared bearer token,
`min_containers=0`, single dev ElevenLabs account with no archival strategy, deferred ElevenLabs
ToS legal review — is the explicit punch list to resolve before any production/public-facing
deployment (cross-referencing the "POC note" callouts already in
[03-business-logic.md](03-business-logic.md),
[05-infrastructure-modal.md](05-infrastructure-modal.md), and
[07-risks-and-open-questions.md](07-risks-and-open-questions.md)).

---

## Critical files (touched across multiple phases)

- `app.py` — `modal.App`, the four `modal.Image` builds, Volume/Secret declarations (Phase 0,
  touched throughout)
- `director.py` / `tts.py` — Stage 0 + Stage A, validated standalone in Phase 1
- `store.py` — `ProfileStore` interface + `PostgresProfileStore`, gating all host-profile
  reads/writes (Phase 7)
- `onboard.py` — Stage -1, integrating IVC, ffmpeg, `choose_preferred_restorer()`, and storage
  (Phase 8)
- `prep.py` / `generate_stage.py` (Function A) — Stage B+C, where the empirically-tuned crop
  constants live (Phase 9)
- `restore.py` / `composite.py` (Function B) — Stage D+E, the custom temporal blend, and
  compositing/mux (Phase 10)
- `main_generate.py` — final orchestration endpoint tying every phase together (Phase 11)
