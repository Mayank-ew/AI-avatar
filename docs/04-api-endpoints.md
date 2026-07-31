> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# API Endpoints

Three endpoints: the one-time host-onboarding endpoint, the main generation endpoint (both
public-facing, called by EasyWebinar's backend), and a one-time weight-download utility
endpoint (internal/ops use only, not public).

## 1. `POST /host/onboard` — one-time host onboarding

**Auth**: same posture as `/generate` below (`requires_proxy_auth=True` + app-level bearer
token).

**Purpose**: accept a host's short video once, extract its audio, create an ElevenLabs Instant
Voice Clone, decide which face restorer (RestoreFormer++ or GFPGAN) works reliably for this
host, and persist the resulting `{voice_id, base_video_ref, preferred_restorer}` profile — see
[03-business-logic.md](03-business-logic.md) §0 for the full contract and
[05-infrastructure-modal.md](05-infrastructure-modal.md) for the metadata store and video
storage this writes to.

### Request schema (multipart/form-data — this endpoint accepts a file upload, not pure JSON)

```
host_id:            string, required — caller-supplied stable identifier for this host
                     (e.g. EasyWebinar's own user/account ID)
video:               file, required — short video of the host (used for both voice cloning
                     and half-body animation reference; ~1-3 minutes recommended, matching
                     ElevenLabs IVC's ideal audio-duration range)
consent_attested:    boolean, required — must be true; the request is rejected outright if
                     false or absent (see 03-business-logic.md §0 — this is a hard gate,
                     not a soft prompt)
voice_character_hint: string, optional — e.g. "warm, calm narrator", stored alongside the
                     profile and passed to the LLM director on every subsequent generation
preferred_restorer_override: string, optional — "restoreformer++" | "gfpgan"; if supplied,
                     bypasses the automated restorer-reliability check entirely and stores
                     this value directly (the manual-override escape hatch described in
                     02-models.md §4a)
```

### Response schema

**Success (200)**:
```json
{
  "status": "success",
  "host_id": "string — echoed back",
  "voice_id": "string — the ElevenLabs voice_id created for this host",
  "requires_verification": "boolean — from ElevenLabs' IVC response; downstream effect UNVERIFIED, see 07-risks-and-open-questions.md",
  "preferred_restorer": "restoreformer++ | gfpgan — the restorer decided (or overridden) for this host; stored in the profile and used, unchanged, for every future generation"
}
```

**Error (4xx/5xx)**:
```json
{
  "status": "error",
  "stage": "consent_check | audio_extraction | voice_clone | restorer_check | storage",
  "message": "string — human-readable"
}
```

A missing/false `consent_attested` must fail fast at `consent_check`, before any ffmpeg or
ElevenLabs call runs.

### Re-onboarding (replace an existing profile)

`PUT /host/{host_id}/onboard` — same request/response shape as above. Per the "single profile,
replaceable" design decision, this **overwrites** the existing
`{voice_id, base_video_ref, preferred_restorer}` record — including re-running the restorer
reliability check against the newly uploaded video (a host's new onboarding video could
plausibly restore differently than their old one). Implementation should explicitly delete the
old ElevenLabs `voice_id` (freeing its quota slot) rather than leaving it orphaned — see
[07-risks-and-open-questions.md](07-risks-and-open-questions.md).

### Example request

```bash
curl -X POST https://<your-modal-app>--host-onboard.modal.run \
  -H "Modal-Key: <proxy-auth-key>" \
  -H "Modal-Secret: <proxy-auth-secret>" \
  -H "Authorization: Bearer <app-level-token>" \
  -F "host_id=host_123" \
  -F "video=@presenter_intro.mp4" \
  -F "consent_attested=true"
```

## 2. `POST /generate` — main generation endpoint

**Auth**: `requires_proxy_auth=True` on the `@modal.fastapi_endpoint()` (Modal's own edge-level
auth, rejects unauthorized calls before any container spins up — zero cost for bad requests),
**plus** an app-level bearer token check (FastAPI `Depends`/`HTTPBearer` against a value pulled
from a `modal.Secret`) if EasyWebinar's backend is the sole intended caller. Don't rely on
proxy-auth alone if the token also needs to be validated by application logic (e.g., per-tenant
rate limiting).

### Request schema

```json
{
  "host_id": "string, required — resolves to {voice_id, base_video_ref, preferred_restorer} via a profile lookup (see 01-architecture-overview.md); no voice_id, video, or restorer choice is ever passed directly to this endpoint under normal operation",
  "script_text": "string, required — the plain narration script",
  "target_ratio": "string, required — e.g. \"16:9\", \"9:16\", \"1:1\"",
  "voice_character_hint": "string, optional — overrides the hint stored at onboarding time for this one request, passed to the LLM director",
  "enable_director": "boolean, optional, default true — set false to bypass Stage 0 and send clean_text directly to TTS (uses eleven_multilingual_v2 instead of eleven_v3, since there are no tags to be responsive to)",
  "elevenlabs_model_override": "string, optional — force a specific model_id, overriding the director-based auto-selection",
  "restorer_override": "string, optional — \"restoreformer++\" | \"gfpgan\"; overrides the host's stored preferred_restorer for this ONE generation only (e.g. for testing/reprocessing a bad result) without changing their onboarded profile. Absent this field, Stage D always uses profile.preferred_restorer — there is no per-request default and no per-frame fallback, see 02-models.md §4a."
}
```

If `host_id` doesn't resolve to a stored profile (never onboarded, or deleted), the request
fails fast with a `profile_lookup` stage error — before Stage 0 or any GPU Function is ever
invoked.

### Response schema

**Success (200)**:
```json
{
  "status": "success",
  "video_url": "string — S3 URL (or base64-encoded video, per deployment choice)",
  "directed_text": "string — the tag-annotated script actually sent to TTS, for auditability/debugging",
  "stability_hint": "creative | natural",
  "duration_seconds": "number",
  "target_ratio": "string — echoed back",
  "resolution": "string — e.g. \"1080x1920\""
}
```

Including `directed_text` in the response is deliberate — it lets EasyWebinar's backend (or a
developer debugging a bad render) see exactly what was sent to ElevenLabs without needing to
inspect Modal logs.

**Error (4xx/5xx)**:
```json
{
  "status": "error",
  "stage": "profile_lookup | director | tts | face_detection | echomimic | restoration | compositing | mux",
  "message": "string — human-readable",
  "detail": "object | null — upstream error body if available (ElevenLabs 422 shape is confirmed: {\"detail\": [{\"loc\", \"msg\", \"type\"}]}; other upstream error shapes are UNVERIFIED — see 07-risks-and-open-questions.md)"
}
```

The `stage` field matters operationally: it tells the caller (and whoever's debugging) which
of the pipeline's independent Modal Functions failed, since Function A and Function B fail
independently and for different reasons (e.g., "no face detected" is always a Function A
`face_detection` failure; a torch OOM is almost always `echomimic` on Function A).

### Example request

```bash
curl -X POST https://<your-modal-app>--generate.modal.run \
  -H "Content-Type: application/json" \
  -H "Modal-Key: <proxy-auth-key>" \
  -H "Modal-Secret: <proxy-auth-secret>" \
  -H "Authorization: Bearer <app-level-token>" \
  -d '{
    "host_id": "host_123",
    "script_text": "Welcome to today'\''s webinar on serverless AI pipelines.",
    "target_ratio": "9:16"
  }'
```

## 3. `POST /admin/download-weights` — one-time weight-download utility

**Not public.** Internal/ops endpoint (or plain `modal run app.py::download_weights`, not a
web endpoint at all — a `@app.function()` without `@modal.fastapi_endpoint()` is arguably the
right shape here, invoked via `modal run` rather than HTTP). If exposed as HTTP at all, gate it
behind the strictest available auth (proxy-auth + a separate admin-only secret) since it writes
to the shared weights Volume.

**Purpose**: pull all model weights into the persistent `modal.Volume` once, ahead of any real
request — per the original spec's "do NOT download weights dynamically during the web endpoint
request" requirement (still correct, still enforced here).

### What it downloads (see [05-infrastructure-modal.md](05-infrastructure-modal.md) for the
full snippet)

| Weight set | Mechanism |
|---|---|
| EchoMimicV2 checkpoints | `huggingface_hub.snapshot_download(repo_id="BadToBest/EchoMimicV2")` |
| VAE | `snapshot_download(repo_id="stabilityai/sd-vae-ft-mse")` |
| Reference UNet base | `snapshot_download(repo_id="lambda/sd-image-variations-diffusers", allow_patterns=["unet/*"])` |
| Whisper tiny | raw URL download (not HF) |
| RestoreFormer++ / GFPGAN weights | HF `snapshot_download` or direct release-asset download — confirm exact hosting at implementation time (do not assume `snapshot_download` blindly; see [07-risks-and-open-questions.md](07-risks-and-open-questions.md)) |

### Response schema

```json
{
  "status": "success",
  "downloaded": ["BadToBest/EchoMimicV2", "stabilityai/sd-vae-ft-mse", "..."],
  "volume_committed": true
}
```

Must call `volume.commit()` at the end so the write is visible to other containers (Function A
and Function B) that only `reload()` the Volume, not remount it fresh each call.
