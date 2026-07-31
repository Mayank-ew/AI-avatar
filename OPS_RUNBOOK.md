# Ops Runbook — EasyWebinar AI Avatar Pipeline

Operational guide for the **current (Wan2.2-S2V)** pipeline: setup, onboarding, generation, the
live-demo procedure, and troubleshooting.

> **Windows note:** use `py -m modal ...` (not a bare `modal ...`) — it sidesteps the common
> `modal.exe`-not-on-PATH issue. All commands below assume you run from the project root.

## 1 · One-time setup

```bash
pip install modal && py -m modal setup
```

**Credentials** — drop a `.env` next to `app.py` (auto-shipped to every function via
`Secret.from_dotenv`, since `SECRET_MODE="dotenv"`):

```
ELEVENLABS_API_KEY=...      GROQ_API_KEY=...        GEMINI_API_KEY=...     FISH_API_KEY=...
APP_BEARER_TOKEN=...        MODAL_PROXY_KEY=wk-...   MODAL_PROXY_SECRET=ws-...
ONBOARD_URL=https://<workspace>--easywebinar-avatar-pipeline-host-onboard.modal.run
GENERATE_URL=https://<workspace>--easywebinar-avatar-pipeline-generate.modal.run
```
Also create a Modal proxy-auth token (modal.com/settings/proxy-auth-tokens) → that's
`MODAL_PROXY_KEY`/`MODAL_PROXY_SECRET`; endpoints use `requires_proxy_auth=True`.

**Deploy + weights (once):**
```bash
py -m modal deploy app.py
py -m modal run app.py::download_weights          # ~49GB Wan2.2-S2V-14B → weights volume
py -m modal run app.py::download_distill_lora      # ~631MB 4-step distill LoRA
py -m modal run app.py::health_all                 # every image builds + sees its GPU/keys
```
`modal deploy` prints the endpoint URLs (onboard / generate / control_panel). The
**control_panel** URL is the demo UI.

## 2 · Onboard a host

**A. Normal (clean clip) — via the control panel:** fill Host ID + upload a short video (or paste
a URL) + tick consent → *Onboard host*. This clones the voice (ElevenLabs IVC) and builds the
studio reference. (nano-banana is API-blocked → the reference auto-step soft-fails; set it by
hand, below.)

**B. From a webinar / small host-tab, or with a pre-made voice (manual, CLI):**
```bash
# 1. put a base clip + your reference portrait on the volume
py -m modal volume put host-media-vol <clip>.mp4 <host_id>/onboarding_video.mp4 --force
py -m modal volume put host-media-vol <portrait>.png <host_id>/reference_studio.png --force

# 2. CREATE the profile (this is what makes the host exist / appear in the panel).
#    --voice-id skips voice cloning and reuses a pre-made ElevenLabs voice; for Fish use any
#    placeholder (Fish uses its own reference_id at generation).
py -m modal run app.py::app.onboard_existing --host-id <host_id> --consent true --voice-id <voice_or_placeholder>

# 3. attach the reference portrait
py -m modal run app.py::app.set_reference --host-id <host_id>
```
Notes: `set_reference` only **updates** an existing profile — run `onboard_existing` first or it
errors with "no profile". The `reference_error` line in step 2 is the blocked nano-banana step —
ignore it; step 3 overrides.

**Per-aspect / reel references** (avoid a zoomed 9:16 crop):
```bash
# register a hand-made 9:16 portrait for the 9:16 aspect
py -m modal volume put host-media-vol <reel>.png <host_id>/reference_studio_9x16.png --force
py -m modal run app.py::app.set_reference --host-id <host_id> --ratio 9:16
# or auto face-aware-crop an existing reference to an exact aspect (CPU, cheap):
py -m modal run app.py::crop_reference_to_ratio --host-id <host_id> --ratio 9:16
```

Grab the best frame / built reference for hand-editing:
```bash
py -m modal run app.py::app.pick_reference_frame --host-id <host_id>
py -m modal volume get host-media-vol <host_id>/reference_source.png .
```

## 3 · Generate

In the control panel: **Host ID** + script + **aspect ratio** + **TTS provider**
(ElevenLabs, or Fish Audio + a `reference_id`) → *Generate*. It submits an async job, polls
`/status`, then streams the finished MP4 into the panel (auto-scrolls + plays). Click
*(list onboarded hosts)* to see stored `host_id`s.

## 4 · Live-demo procedure (avoid dead air)

1. **Redeploy** if you changed anything: `py -m modal deploy app.py`.
2. **~5 min before presenting, fire one throwaway short generation** to warm the H100.
   `scaledown_window=600` keeps it warm ~10 min, so the live run skips the ~2.5-min cold start.
   (For a guaranteed zero-cold-start window, set `min_containers=1` in `generate_stage.py`, deploy,
   and set it back to `0` after.)
3. Hit **Generate**, then talk over the architecture page — a ~15–25 s clip renders in ~2–4 min
   warm and the video autoplays when done.
4. **Backup:** keep a pre-rendered MP4 open in another tab in case the live run hiccups.

## 5 · Troubleshooting

- **Panel stuck "Rendering…" / no video:** the run may be done on Modal already. Video is served
  from `GET /video/{job_id}` (streamed). If needed, pull it straight off the volume:
  ```bash
  py -m modal volume ls avatar-intermediate-vol            # find the run_id (== job_id)
  py -m modal volume get avatar-intermediate-vol <run_id>/final_output.mp4 out.mp4
  ```
- **Slow / expensive run:** check the log line `num_repeat=N` — cost scales with it
  (`ceil(audio_secs/5)`). Long scripts = many chunks. `WAN_MAX_SECONDS` caps it.
- **Distill not applied:** cold-start log should show `Wan distill: LoRA merged into N modules`
  (N large). If N=0 the LoRA didn't match — it falls back to stock steps automatically.
- **Fish 429 / quota:** `s2.1-pro-free` is rate-limited; the panel surfaces Fish's error.
- **nano-banana `limit:0`:** Gemini image API needs paid billing — use the manual reference path.

## 6 · Cost / speed knobs (`constants.py`)

| Knob | Current | Effect |
|---|---|---|
| `WAN_MAX_AREA` | `409600` (480p) | pixels/step; 720p (`921600`) ≈ 2.3× slower |
| `WAN_USE_DISTILL` / `WAN_DISTILL_STEPS` | `True` / `4` | 4-step vs ~40; the big speedup |
| `WAN_MAX_SECONDS` | `30` | hard duration cap (cost guard) |
| `WAN_ENABLE_SNAPSHOT` | `False` | snapshots break on Wan conv3d — leave off |
| `min_containers` (generate_stage) | `0` | set `1` for zero cold start (idle H100 cost) |

## 7 · POC → production punch list

| Item | POC state | Production requirement |
|---|---|---|
| Profile store | `modal.Dict` | pooled/backed-up managed DB if needed |
| Auth | single shared bearer token | per-tenant tokens + rate limiting |
| GPU warmth | `min_containers=0` | `min_containers>=1` sized to traffic |
| Voice likeness/consent | consent checkbox | retention + deletion-on-request policy, legal review |
| ElevenLabs/Fish ToS (commercial redistribution) | internal demo | read live ToS before public content |
| nano-banana | manual (billing off) | enable Gemini pay-as-you-go → automate references |
| Video delivery | streamed `/video` (done) | signed S3/GCS URL for external clients |
