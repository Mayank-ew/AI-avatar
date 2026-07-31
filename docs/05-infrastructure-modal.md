> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# Infrastructure — Modal.com Setup

This corrects the original spec's Modal usage to the **current** Modal API. The original spec
used `modal.Stub`, `@stub.function`, `@modal.web_endpoint`, and `gpu="RTX4090"` — all
deprecated or invalid. Everything below uses the current API surface; re-verify against the
installed `modal` package version at implementation time regardless, since Modal's scaling/
concurrency parameter names have been in flux (see the callout in §5).

## 0. Host profile storage — metadata store + video storage

Per the host-onboarding design (the pipeline owns this storage itself, not an integration with
an existing EasyWebinar backend — see [03-business-logic.md](03-business-logic.md) §0):

```python
# Metadata store: host_id -> {voice_id, base_video_path, preferred_restorer, created_at}
host_profiles = modal.Dict.from_name("host-profiles", create_if_missing=True)

# Video storage: the actual uploaded onboarding video per host
host_media_volume = modal.Volume.from_name("host-media-vol", create_if_missing=True)
```

`preferred_restorer` (`"restoreformer++"` or `"gfpgan"`) is decided once, during onboarding —
via an automated reliability check against a sample frame from the uploaded video, or a manual
override if supplied (see [02-models.md](02-models.md) §4a and
[03-business-logic.md](03-business-logic.md) §0). It is never decided at generation time, and
Stage D never contains fallback/switching logic of its own — it just reads this stored value.

- **`modal.Dict` chosen over standing up a real database for this first cut.** The access
  pattern is exactly one write (at onboarding) and one read-by-key (at every generation) — a
  key-value store matches that directly, with no separate database to provision, connect-pool,
  or operate. If host-management features grow (an admin dashboard listing all hosts, usage
  analytics, relational queries across hosts), migrate to a real Postgres then — building that
  now would be premature for a single get/put lookup.
- **`host_media_volume` is a separate `modal.Volume` from the weights volume** (§2 below) —
  different write pattern (weights: written once by an ops script; host videos: written per
  onboarding request from a public-facing endpoint) and different lifecycle, so keeping them
  as distinct Volumes avoids conflating an ops concern with a per-tenant data concern. Store
  each host's video at a per-host path, e.g. `/host_media/{host_id}/onboarding_video.mp4`.

> **POC note:** for the proof-of-concept build, swap `host_profiles` for a small hosted
> Postgres/MySQL instance (e.g. a free-tier Supabase/Neon/PlanetScale database), reached from
> the Modal Functions over a connection string stored in a `modal.Secret` — same
> `{host_id → voice_id, base_video_path, preferred_restorer, created_at}` shape, just a plain
> SQL table with
> `host_id` as the primary key instead of a `modal.Dict`. This is fine for demoing the feature
> and architecture end-to-end (and makes it trivial to open a normal SQL client and show the
> stored profile data during a demo), but it is **not** what should carry into production as-is
> — a single small hosted instance has no connection pooling, backup/HA story, or scaling plan
> for real multi-tenant traffic. Before production, either move to the `modal.Dict` approach
> documented above (Modal-native, no external service to operate) or a properly provisioned,
> pooled, backed-up managed database — whichever fits EasyWebinar's actual operational setup at
> that point. Everything downstream (the profile-lookup contract in
> [03-business-logic.md](03-business-logic.md), the `host_id`-only `/generate` schema in
> [04-api-endpoints.md](04-api-endpoints.md)) is identical either way — only this one storage
> call changes.

## 1. Four container images, not one

Per [01-architecture-overview.md](01-architecture-overview.md)'s environment split:

```python
import modal

app = modal.App("easywebinar-avatar-pipeline")

# --- Image 0: thin CPU-only, host onboarding (Stage -1) ---
onboarding_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")  # required to extract audio from the uploaded onboarding video —
                             # ElevenLabs' IVC endpoint is not confirmed to accept video directly
    .pip_install("requests", "python-multipart")
)

# --- Image 1: thin CPU-only, Stage 0 (LLM Director) + Stage A (ElevenLabs TTS) ---
director_tts_image = (
    modal.Image.debian_slim()
    .pip_install("openai", "requests")
)

# --- Image 2: heavy GPU, Function A — canvas/pose prep + EchoMimicV2 ---
echomimic_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        "xformers==0.0.28.post3",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers==0.31.0", "transformers==4.46.3", "einops==0.8.0",
        "omegaconf==2.3.0", "opencv-python", "insightface", "onnxruntime-gpu==1.20.1",
        "numpy==1.26.4", "moviepy==1.0.3", "huggingface_hub==0.26.2",
        "accelerate==1.1.1", "imageio==2.36.0", "imageio-ffmpeg==0.5.1",
    )
    .pip_install("facenet_pytorch==2.6.0", extra_options="--no-deps")
    .run_commands(
        "git clone https://github.com/antgroup/echomimic_v2 /workspace/echomimic_v2",
        # static ffmpeg 4.4 binary — EchoMimicV2's infer.py checks FFMPEG_PATH explicitly,
        # apt's ffmpeg is not a substitute for this
        "wget -q https://.../ffmpeg-4.4-amd64-static.tar.xz -O /tmp/ffmpeg.tar.xz "
        "&& tar -xf /tmp/ffmpeg.tar.xz -C /workspace",
    )
    .env({"FFMPEG_PATH": "/workspace/ffmpeg-4.4-amd64-static"})
)

# --- Image 3: lighter GPU, Function B — face restore + temporal blend + composite + mux ---
restoration_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",  # independent of Function A's env — no shared install
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("opencv-python", "basicsr", "facexlib", "gfpgan", "numpy", "huggingface_hub")
    .run_commands(
        "git clone https://github.com/wzhouxiff/RestoreFormerPlusPlus /workspace/restoreformer",
    )
)
```

Function A and Function B do **not** share an image — their pinned dependency sets hard-conflict
(EchoMimicV2 needs `diffusers==0.31.0`/specific `torch`/`numpy` pins; a from-scratch restoration
environment has no reason to carry the diffusion stack at all). Building separate `modal.Image`s
per the idiomatic Modal pattern is correct here, not a workaround. The onboarding image is
separate again from `director_tts_image` even though both are thin/CPU-only — onboarding is the
only place `ffmpeg` is needed outside the two GPU Functions (for audio extraction), and there's
no reason to carry that apt package into the per-generation Stage 0/A path that never touches a
video file.

## 2. Persistent volume for weights

```python
weights_volume = modal.Volume.from_name("avatar-weights-vol", create_if_missing=True)
```

- Mount via `volumes={"/vol": weights_volume}` on any `@app.function()` that needs weights.
- **Consistency is not automatic.** Writes inside a container are local until
  `weights_volume.commit()` is called — other already-running containers must
  `weights_volume.reload()` to see the update; mounting once at container start does not
  auto-refresh. The one-time download utility (below) must `commit()` explicitly before
  Function A/B are expected to see the weights.

```python
@app.function(image=director_tts_image, volumes={"/vol": weights_volume}, timeout=1800)
def download_weights():
    from huggingface_hub import snapshot_download
    import os, urllib.request

    snapshot_download(repo_id="BadToBest/EchoMimicV2", local_dir="/vol/echomimic")
    snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir="/vol/sd-vae-ft-mse")
    snapshot_download(
        repo_id="lambda/sd-image-variations-diffusers",
        local_dir="/vol/sd-image-variations-diffusers",
        allow_patterns=["unet/*"],
    )
    os.makedirs("/vol/whisper", exist_ok=True)
    urllib.request.urlretrieve(
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        "/vol/whisper/tiny.pt",
    )
    # RestoreFormer++/GFPGAN weight hosting: confirm exact mechanism at implementation time
    # (HF snapshot_download vs. direct release-asset download) — do not assume, see
    # 07-risks-and-open-questions.md.

    weights_volume.commit()
    return {"status": "success"}
```

Run once via `modal run app.py::download_weights` (or the `/admin/download-weights` endpoint
per [04-api-endpoints.md](04-api-endpoints.md)) — **never** inside the request-serving path.

## 3. GPU choice

`gpu="RTX4090"` from the original spec **is not a valid Modal GPU string** — no such SKU
exists on Modal's platform. Valid identifiers: `T4`, `L4`, `A10`, `L40S`, `A100`,
`A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100`, `H200`, `B200`, `B300` (note: `A10`, not the
AWS-ism `A10G`).

- **Function A (EchoMimicV2)**: needs ~16GB+ VRAM minimum (community-confirmed) — `A10`
  (24GB) or `L40S`/`A100` for headroom. Benchmark before committing to the cheapest viable tier.
- **Function B (RestoreFormer++/GFPGAN + temporal blend)**: expected much lighter
  (feedforward GAN, no diffusion stack) — `T4`/`L4`/`A10` are plausible starting points, but
  VRAM/perf here is **UNVERIFIED**; benchmark on the actual target GPU before finalizing.
- **Host onboarding**: no `gpu=` parameter at all — it's an ffmpeg call plus an ElevenLabs API
  call, same CPU-only shape as Stage 0/A.

```python
@app.function(
    image=echomimic_image,
    gpu="A10",
    timeout=600,
    volumes={"/vol": weights_volume},
    secrets=[modal.Secret.from_name("elevenlabs-secret")],
)
def run_echomimic(...):
    ...

@app.function(
    image=restoration_image,
    gpu="L4",
    timeout=300,
    volumes={"/vol": weights_volume},
)
def run_restoration_and_composite(...):
    ...
```

## 4. Web endpoints, auth, secrets

`@modal.web_endpoint` is deprecated — use `@modal.fastapi_endpoint()`. Per
[04-api-endpoints.md](04-api-endpoints.md), there are two public endpoints — onboarding and
generation — plus the internal weight-download utility:

```python
@app.function(
    image=onboarding_image,
    volumes={"/host_media": host_media_volume},
    secrets=[modal.Secret.from_name("elevenlabs-secret")],
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def host_onboard(host_id: str, video: UploadFile, consent_attested: bool,
                  preferred_restorer_override: str | None = None):
    if not consent_attested:
        return {"status": "error", "stage": "consent_check",
                "message": "consent_attested must be true"}

    video_path = f"/host_media/{host_id}/onboarding_video.mp4"
    save_upload(video, video_path)  # write the multipart upload to the Volume

    audio_path = f"/host_media/{host_id}/onboarding_audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_path],
        check=True,
    )

    resp = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        data={"name": host_id},
        files={"files": open(audio_path, "rb")},
    )
    resp.raise_for_status()
    body = resp.json()

    # Decide preferred_restorer ONCE, here, at onboarding — never at generation time.
    # See 02-models.md §4a for the reliability-check mechanism this wraps.
    if preferred_restorer_override:
        preferred_restorer = preferred_restorer_override
    else:
        sample_frame = extract_sample_frame(video_path)  # e.g. via ffmpeg, one representative frame
        preferred_restorer = choose_preferred_restorer(sample_frame)  # "restoreformer++" | "gfpgan"

    host_media_volume.commit()
    host_profiles[host_id] = {
        "voice_id": body["voice_id"],
        "base_video_path": video_path,
        "preferred_restorer": preferred_restorer,
        "created_at": ...,  # pass in a timestamp from the caller; Modal scripts can't call
                             # datetime.now()/time.time() at script-definition time, only at
                             # request-handling time inside the function body
    }
    return {
        "status": "success", "host_id": host_id,
        "voice_id": body["voice_id"],
        "requires_verification": body.get("requires_verification"),
        "preferred_restorer": preferred_restorer,
    }


@app.function(
    image=director_tts_image,
    secrets=[modal.Secret.from_name("elevenlabs-secret"), modal.Secret.from_name("openai-secret")],
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def generate(payload: dict):
    profile = host_profiles.get(payload["host_id"])
    if profile is None:
        return {"status": "error", "stage": "profile_lookup",
                "message": "host_id has no onboarded profile"}

    voice_id = profile["voice_id"]
    base_video_path = profile["base_video_path"]
    # restorer_override (04-api-endpoints.md) is the only thing that can override this for a
    # single request; absent that, Stage D always uses the profile's stored value — there is
    # no other decision-making left in the generation path.
    restorer = payload.get("restorer_override") or profile["preferred_restorer"]
    # ... proceed to Stage 0 (director) using payload["script_text"], then Stage A using
    # voice_id, then call run_echomimic(...) / run_restoration_and_composite(..., restorer) with
    # base_video_path as the source for Master_Background/reference-crop derivation
```

- `ELEVENLABS_API_KEY` and `OPENAI_API_KEY` live in `modal.Secret`s, injected only into the
  Functions that need them — never into a response.
- `requires_proxy_auth=True` rejects unauthorized calls at Modal's edge (zero container cost
  for bad requests); clients send `Modal-Key`/`Modal-Secret` headers, tokens created per-
  workspace at `modal.com/settings/proxy-auth-tokens`. Layer an app-level bearer token via
  FastAPI `Depends`/`HTTPBearer` on top if additional application-level checks are needed —
  apply this to **both** `host_onboard` and `generate`, since both are public-facing.

> **POC note:** `requires_proxy_auth=True` plus a single shared bearer token (one static value
> checked against a `modal.Secret`, no per-tenant issuance/rotation/rate-limiting) is sufficient
> to demo the pipeline securely. Production needs the fuller posture this section already
> describes (per-tenant tokens, rate limiting tied to EasyWebinar's own account system) before
> real hosts are onboarded — the shared-token approach doesn't distinguish one caller from
> another, which is fine for a controlled demo, not for multi-tenant production traffic.

## 5. Scaling parameters — current names

The original spec's `container_idle_timeout`/`keep_warm`/`concurrency_limit` are deprecated.
Current equivalents on `@app.function()`:

| Current name | Replaces | Purpose |
|---|---|---|
| `min_containers` | `keep_warm` | Keep N containers warm to avoid cold-start GPU+model-load latency |
| `max_containers` | `concurrency_limit` | Hard cap on concurrent containers |
| `scaledown_window` | `container_idle_timeout` | How long an idle container stays warm before scaling to zero |
| `buffer_containers` | — (new) | Extra idle containers maintained during active load, to absorb bursts |

```python
@app.function(
    image=echomimic_image,
    gpu="A10",
    min_containers=1,          # keep one GPU warm to avoid cold-start on the first request
    max_containers=5,
    scaledown_window=300,
    timeout=600,
    volumes={"/vol": weights_volume},
)
```

> **POC note:** set `min_containers=0` for both GPU Functions during the POC — a demo doesn't
> need to avoid the one-time cold-start/model-load delay on the first request of a session, and
> paying to keep a GPU container warm 24/7 for infrequent demo traffic isn't worth the cost.
> Accept the cold start each time instead. Production should set `min_containers=1` (or higher,
> sized to expected concurrent webinar traffic) exactly as shown above, once real usage
> patterns justify the always-on GPU cost.

**Concurrent-input-per-container handling is the murkiest part of the current Modal API** —
there are conflicting reports of `@modal.concurrent(max_inputs=N)` vs. a possibly-superseding
`single_use_containers` flag. **Re-verify against `modal.com/docs/guide/concurrent-inputs` and
the installed `modal` package's changelog at implementation time** before committing to either
name in code — do not trust this doc's phrasing on this one specific point without a fresh
check.

## 6. Pricing (for cost estimation)

Per-second GPU billing (confirm current rates at `modal.com/pricing` before finalizing cost
estimates — these are a snapshot):

| GPU | $/sec |
|---|---|
| T4 | 0.000164 |
| L4 | 0.000222 |
| A10 | 0.000306 |
| A100 40GB | 0.000583 |
| L40S | 0.000542 |
| H100 | 0.001097 |

Rough per-request cost model: GPT-4o tokens (Stage 0) + ElevenLabs characters (Stage A,
billed by ElevenLabs separately) + Function A GPU-seconds (EchoMimicV2, the dominant cost —
minutes, not seconds, per the ~7min/120-frame standard pipeline timing in
[02-models.md](02-models.md)) + Function B GPU-seconds (expected much smaller). Whether Modal's
`timeout=` parameter itself affects billing (vs. being a pure safety ceiling on actual
container-seconds consumed) is **UNVERIFIED** — treat `timeout` as a kill-switch, not a
billing multiplier, pending confirmation.
