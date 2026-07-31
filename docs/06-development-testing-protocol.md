> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# Development & Testing Protocol

Restates the original spec's 4-phase protocol, corrected for the current architecture (host
onboarding as its own flow, plus the three generation-time environments), with a phase
ordering that de-risks the genuinely new/unproven pieces (host onboarding, the LLM
director→tag pipeline, and the custom temporal-blend pass) before they're wired into the full,
expensive end-to-end video pipeline.

## Phase 1 — Scaffolding

Write `app.py` (or split across modules) with the four `modal.Image` definitions, the
`modal.Volume` declarations (weights + host media), and the `modal.Dict` host-profile store
from [05-infrastructure-modal.md](05-infrastructure-modal.md). Run `modal deploy app.py` to
confirm all four images build without dependency errors — this catches pip conflicts cheaply,
before any GPU time is spent. Confirm specifically that Function A's and Function B's images
build independently without cross-contamination (e.g., accidentally sharing a base layer that
pulls in the wrong `torch` version).

## Phase 2 — Weight download

Run `download_weights()` (via `modal run app.py::download_weights`, not the request path) to
populate `/vol`. Confirm via a follow-up `modal run` invocation (or a small debug function)
that all expected files exist at their documented paths and sizes match
[02-models.md](02-models.md)'s weight tables — a partial/truncated download is a common,
hard-to-diagnose failure mode for large HF snapshot downloads.

## Phase 3 — De-risk the two unproven pieces in isolation, before the full pipeline

This ordering is a deliberate deviation from the original spec's "just build steps A-E in
order" — the genuinely new pieces of this design (not present in the original spec, and
without a battle-tested upstream implementation to fall back on) should be validated cheaply
and in isolation first:

0. **Host onboarding flow, standalone, before anything else.** No GPU needed. Upload a short
   real test video through `host_onboard`, and confirm: the consent gate actually rejects a
   request with `consent_attested=false` (fail fast, no ffmpeg/ElevenLabs call attempted); the
   audio-extraction step produces a playable `.mp3`; the resulting `voice_id` is immediately
   usable in a direct `POST /v1/text-to-speech/{voice_id}` call (sanity-checks the "video isn't
   accepted directly" assumption from [02-models.md](02-models.md) §2a, and the "usable
   immediately regardless of `requires_verification`" assumption from
   [07-risks-and-open-questions.md](07-risks-and-open-questions.md) — both should be confirmed
   empirically here, not assumed); confirm the response includes a populated, sane
   `preferred_restorer` value (`"restoreformer++"` or `"gfpgan"`, never null/missing — see
   [02-models.md](02-models.md) §4a); and a subsequent `generate` call with only `host_id` (no
   `voice_id`/video/restorer fields) succeeds via the profile lookup. This is the cheapest,
   fastest thing to validate in the whole pipeline — do it first.
1. **Director → TTS tag pipeline, standalone.** No GPU needed. Send a handful of sample
   scripts through the GPT-4o director, run the content-preservation guardrail
   (`directed_text` strips back to `clean_text`), and listen to the resulting `eleven_v3`
   audio. Validate: does the tag density/placement actually sound more expressive than plain
   narration? Does the `stability_hint` mapping (creative/natural) produce the expected
   tradeoff? This is fast and cheap to iterate on before any video pipeline exists.
2. **Custom temporal-blending pass, on a short clip.** Generate (or use a placeholder) a short
   sequence of restored face frames, and visually inspect the optical-flow blend output for
   teeth/eye flicker reduction vs. artifacts introduced by the blend itself (ghosting, motion
   blur on fast head turns). Since there is no upstream reference implementation for this
   piece (unlike Stage C), this is the step most likely to need several iterations — budget
   time for it, don't assume it works on the first attempt.

Only after all three are independently validated, build the full Steps B→C→D→E chain
end-to-end.

## Phase 4 — Full pipeline integration

Implement Stage B (canvas format, face-selection, reference-crop derivation, pose extraction +
remap) → Stage C (EchoMimicV2) → Stage D (restoration + validated temporal-blend) → Stage E
(composite + mux), per [03-business-logic.md](03-business-logic.md).

## Phase 5 — End-to-end testing

Create a local `test.py` that exercises the onboarding flow once, then generation (which should
never need `voice_id`/video fields again after that):

```python
import requests

AUTH = {"Authorization": "Bearer <app-level-token>"}

# 1. Onboard a test host once
with open("test_presenter_clip.mp4", "rb") as f:
    onboard_resp = requests.post(
        "https://<your-modal-app>--host-onboard.modal.run",
        headers=AUTH,
        data={"host_id": "test_host_1", "consent_attested": "true"},
        files={"video": f},
    )
onboard_resp.raise_for_status()
onboard_result = onboard_resp.json()
assert onboard_result["status"] == "success"
print("voice_id:", onboard_result["voice_id"])
print("requires_verification:", onboard_result["requires_verification"])

# 2. Generate — note: no voice_id, no video field, just host_id + script + ratio
payload = {
    "host_id": "test_host_1",
    "script_text": "Welcome to today's webinar on serverless AI pipelines. "
                    "We're excited to show you what's possible.",
    "target_ratio": "9:16",
}

resp = requests.post(
    "https://<your-modal-app>--generate.modal.run",
    json=payload,
    headers=AUTH,
)
resp.raise_for_status()
result = resp.json()
assert result["status"] == "success"
print("directed_text:", result["directed_text"])
print("video_url:", result["video_url"])

# 3. Re-run generation with a different script for the SAME host_id — should succeed without
#    any re-onboarding, proving the profile is actually being reused, not re-collected.
```

**Debugging checklist — dump these intermediate artifacts before trusting end-to-end output**,
since a bad final video could stem from any of the independent stages:
- The onboarding profile itself (`voice_id`, `base_video_path`, `preferred_restorer`) —
  confirm the profile lookup actually returned what was stored, before assuming a downstream
  stage is at fault.
- `directed_text` (Stage 0) — does it look reasonable, tags placed sensibly?
- The rendered `/tmp/audio.mp3` (Stage A) — does it sound as expected in isolation?
- The face-selection result and reference-crop bounding box (Stage B §3a) — sanity-check
  against the source frame; a wrong `expansion_ratio`/`vertical_bias` (see
  [03-business-logic.md](03-business-logic.md)) shows up here as an obviously too-tight or
  too-loose crop, before it ever reaches EchoMimicV2.
- The remapped pose sequence (Stage B §3c) — spot-check a few frames' keypoints overlaid on
  the reference crop.
- Sample EchoMimicV2 output frames (Stage C) — check for the ghosting artifacts known to occur
  above 768×768, and general animation quality.
- Sample restored-vs-blended frame pairs, side by side (Stage D) — this is where flicker
  reduction (or new artifacts from the temporal blend) should be visible. If restoration
  quality itself looks off (not a flicker issue), check *which* restorer the host's profile
  actually stored (`preferred_restorer`) — Stage D no longer decides this at runtime, so a bad
  result now more likely traces back to the onboarding-time reliability check picking the
  wrong restorer for that host's video, not a Stage D bug.
- The final composited frame vs. Master_Background (Stage E) — check the color/luminance
  matching at the paste boundary, not just spatial feathering.

## Ephemeral filesystem and OOM management (carried over from the original spec, applied to
the two-Function split)

- **All intermediate files go to `/tmp/`** inside whichever container is running — Modal
  containers are ephemeral, nothing outside `/tmp/` (or the mounted Volume) persists or is
  writable in the expected way.
- **OOM management applies within Function A**, between its own sub-models (VAE → reference
  UNet → denoising UNet → pose encoder → Whisper) if any are loaded/unloaded in sequence
  rather than all at once — `del model; torch.cuda.empty_cache()` before loading the next.
  Function B never needs to coexist in memory with Function A's models at all — the function
  boundary itself is the big, free offload point, per
  [03-business-logic.md](03-business-logic.md) §9.
