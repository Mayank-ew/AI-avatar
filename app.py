"""
EasyWebinar AI Avatar Generation Pipeline — Modal entrypoint.

This file owns the shared Modal primitives every stage depends on: the `modal.App`, the four
container images (docs/05 §1), the persistent Volumes, the host-profile `modal.Dict`, the
Secret references, the health-check functions (Phase 0), and the one-time weight-download
utility (docs/05 §2).

Multi-file registration pattern: the per-stage modules (`director`, `tts`, `onboard`, ...)
decorate their functions with `@app.function(...)` importing `app` and the images FROM THIS
FILE. So this file must define those objects BEFORE importing the stage modules — which it does
at the very bottom. This is the standard Modal multi-file layout; the bottom imports are what
make `modal deploy app.py` discover every stage's functions.

Deploy:      modal deploy app.py
Download:    modal run app.py::download_weights
Health:      modal run app.py::health_onboarding   (and ::health_director_tts, etc.)
"""

import os

import modal

import constants

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("easywebinar-avatar-pipeline")

# ---------------------------------------------------------------------------
# Volumes (docs/05 §0, §2)
# ---------------------------------------------------------------------------

# Model weights — written once by download_weights(), read by Functions A & B.
weights_volume = modal.Volume.from_name("avatar-weights-vol", create_if_missing=True)

# Per-host uploaded onboarding videos — written per onboarding request, read by Function A.
host_media_volume = modal.Volume.from_name("host-media-vol", create_if_missing=True)

# Intermediate A->B artifact handoff (docs/01 "Functions A and B hand off ... via a shared
# modal.Volume"). Frames/pose/audio for a single generation live here under a run_id.
intermediate_volume = modal.Volume.from_name("avatar-intermediate-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Host-profile metadata store (docs/05 §0)
# ---------------------------------------------------------------------------
# modal.Dict is the Modal-native production store. The POC uses PostgresProfileStore instead
# (constants.PROFILE_STORE_BACKEND); both sit behind store.ProfileStore so calling code is
# identical either way. This Dict is declared here so the modal_dict backend can use it.
host_profiles = modal.Dict.from_name("host-profiles", create_if_missing=True)

# Async generation jobs (Wan2.2-S2V takes minutes/clip → /generate is submit→poll). Maps
# job_id -> {status, call_id, run_id, ...}. See main_generate.py.
generation_jobs = modal.Dict.from_name("avatar-generation-jobs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Secrets (docs/05 §4). TWO ways to supply them — pick whichever you prefer:
#
#   (A) Local .env file  [convenient for solo dev]
#       Copy .env.example -> .env, fill in the values. If a .env exists next to app.py at
#       DEPLOY time, its values are read and shipped into every function's containers (via
#       modal.Secret.from_dotenv()). Nothing to run beforehand — just `modal deploy`.
#       Env vars expected in .env: ELEVENLABS_API_KEY, OPENAI_API_KEY, POSTGRES_CONN_STRING,
#       APP_BEARER_TOKEN.
#
#   (B) Named Modal secrets  [better isolation; recommended for prod]
#       Create them once in the workspace, no local file:
#         modal secret create elevenlabs-secret    ELEVENLABS_API_KEY=...
#         modal secret create openai-secret        OPENAI_API_KEY=...
#         modal secret create postgres-conn-string POSTGRES_CONN_STRING=...
#         modal secret create proxy-auth-token     APP_BEARER_TOKEN=...
#
# SECRET_MODE must be a FIXED CONSTANT — never derived from anything that differs between your
# machine and the container (e.g. os.path.exists('.env'), which is True locally but False in the
# remote container). Modal re-imports this file inside the container to re-derive each function's
# dependency list and compares its LENGTH to what the local client sent; if the mode flips
# between the two, the counts mismatch and Modal aborts with "Function has N dependencies but
# container got M object ids". So we hardcode it.
#   "dotenv" -> one from_dotenv() secret carrying all .env vars (this deployment's choice).
#   "named"  -> one modal.Secret.from_name() per name (requires `modal secret create` first).
# To switch to named mode, change this one line to "named" and create the 4 secrets (mode B).
SECRET_MODE = "dotenv"

_DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def secret(name: str) -> modal.Secret:
    """A single named secret (mode (B))."""
    return modal.Secret.from_name(name)


def secrets_for(*names: str) -> list[modal.Secret]:
    """
    Return the secret list to attach to a function. Deterministic per SECRET_MODE — identical
    locally and in-container. In dotenv mode, one from_dotenv() secret (carrying all vars) is
    returned regardless of `names`; in named mode, one per requested name.
    """
    if SECRET_MODE == "dotenv":
        return [modal.Secret.from_dotenv(_DOTENV_PATH)]
    return [modal.Secret.from_name(n) for n in names]

# ---------------------------------------------------------------------------
# Images (docs/05 §1) — four independent images, no shared heavy layer.
# ---------------------------------------------------------------------------

# All images pin python_version explicitly. Without it, modal.Image.debian_slim() defaults to
# matching the LOCAL Python running `modal deploy` — on a machine running Python 3.14, that
# breaks torch==2.5.1 (no cu124 wheel exists for cp314, since 3.14 postdates that PyTorch
# release). 3.11 is fully supported by every pinned package in this project (torch 2.5.1,
# xformers 0.0.28.post3, diffusers 0.31.0, insightface, onnxruntime-gpu, gfpgan, basicsr).
PYTHON_VERSION = "3.11"

# Image 0: thin CPU-only, host onboarding (Stage -1). ffmpeg for audio extraction.
onboarding_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg")
    # fastapi[standard] is required because host_onboard uses @modal.asgi_app (multipart upload);
    # unlike @modal.fastapi_endpoint, asgi_app does not auto-inject the web framework.
    # yt-dlp powers the server-side "onboard from URL" path (downloads happen on Modal, so a
    # locked-down laptop that can't upload files to modal.run only needs to send a text URL).
    .pip_install("requests", "python-multipart", "psycopg2-binary", "fastapi[standard]", "yt-dlp")
)

# Image 1: thin CPU-only, Stage 0 (LLM Director) + Stage A (ElevenLabs TTS) + orchestration.
director_tts_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("openai", "requests", "psycopg2-binary", "fastapi[standard]")
)

# Image 2: heavy GPU, Function A — Wan2.2-S2V-14B audio-driven animation (replaces EchoMimicV2).
# Python 3.10 to match Wan's tested env. torch 2.4/cu124. flash-attn is installed from a prebuilt
# wheel matched to torch2.4/cu124/cp310 — building it from source in the image is the classic
# failure/timeout, so if this exact wheel URL 404s at build time, swap it for the current asset
# from github.com/Dao-AILab/flash-attention/releases (this is the expected W1 fix point).
WAN_PY = "3.10"
wan_image = (
    modal.Image.debian_slim(python_version=WAN_PY)
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
        "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands("git clone https://github.com/Wan-Video/Wan2.2 /workspace/Wan2.2")
    .run_commands(
        # Install Wan's own requirements (S2V-specific ones too). Keep going if an optional
        # extra fails — the S2V requirements pull some heavy/optional pkgs we don't all need.
        "cd /workspace/Wan2.2 && pip install -r requirements.txt",
        "cd /workspace/Wan2.2 && (pip install -r requirements_s2v.txt || true)",
    )
    # peft is imported unconditionally by wan/__init__ -> animate.py (WanAnimate) but isn't always
    # pulled by the requirements install; add it explicitly. numpy<2 for the cv2/torch stack.
    .pip_install("huggingface_hub", "numpy<2", "peft")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

# Image 4: reference-studio — best-frame selection (InsightFace) + nano-banana reimagining.
# CPU-only (the heavy work is a Gemini API call; InsightFace runs one clip's frames on CPU fine).
# Kept separate from the heavy wan_image so it builds fast and needs no torch/CUDA.
studio_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless", "insightface", "onnxruntime==1.20.1",
        "numpy==1.26.4", "pillow",
        # google-genai is the current official SDK for Gemini 2.5 Flash Image ("nano-banana").
        "google-genai",
    )
)

# Small dedicated image for the ops-only weight download. docs/05 §2 shows this on
# director_tts_image, but that image has no huggingface_hub — so a dedicated tiny image with
# huggingface_hub + requests is the correct shape (deviation noted).
weights_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("huggingface_hub", "requests")
)

# Thin image for the demo control panel (frontend.py) — serves HTML + proxies to the pipeline.
frontend_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("fastapi[standard]", "requests")
)

# Every image carries ALL project modules, INCLUDING "app" itself. Functions defined in the
# sibling modules (onboard/generate_stage/main_generate/reference_studio/frontend) all do
# `from app import ...`, so when Modal imports one of those modules inside a container to run its
# function, `app` must be importable there too — not just auto-handled as the entrypoint. Heavy
# libs (cv2/torch/insightface) are imported INSIDE function bodies, so module import stays cheap.
_ALL_SOURCES = (
    "app",
    "constants", "store", "director", "tts", "generate_stage",
    "onboard", "main_generate", "frontend", "reference_studio",
)
onboarding_image = onboarding_image.add_local_python_source(*_ALL_SOURCES)
director_tts_image = director_tts_image.add_local_python_source(*_ALL_SOURCES)
wan_image = wan_image.add_local_python_source(*_ALL_SOURCES)
studio_image = studio_image.add_local_python_source(*_ALL_SOURCES)
weights_image = weights_image.add_local_python_source(*_ALL_SOURCES)
frontend_image = frontend_image.add_local_python_source(*_ALL_SOURCES)

# GPU tier for Function A (Wan2.2-S2V-14B). Needs 80GB with offload+dtype-convert → H100.
GPU_FUNCTION_A = "H100"

# ---------------------------------------------------------------------------
# Weight download utility (docs/05 §2, docs/04 §3). Ops-only — never in the request path.
# Run once:  modal run app.py::download_weights
# ---------------------------------------------------------------------------

@app.function(
    image=weights_image,
    volumes={constants.WEIGHTS_ROOT: weights_volume},
    timeout=7200,  # ~49GB pull
)
def download_weights():
    """Download the Wan2.2-S2V-14B weights (~49GB) into the weights volume. Run once:
    modal run app.py::download_weights"""
    from huggingface_hub import snapshot_download

    ckpt_dir = constants.WAN_CKPT_DIR
    snapshot_download(repo_id=constants.WAN_HF_REPO, local_dir=ckpt_dir)
    weights_volume.commit()
    result = {"status": "success", "downloaded": [constants.WAN_HF_REPO],
              "ckpt_dir": ckpt_dir, "volume_committed": True}
    print(result)
    return result


@app.function(
    image=weights_image,
    volumes={constants.WEIGHTS_ROOT: weights_volume},
    timeout=1800,
)
def download_distill_lora():
    """Download the Wan2.1-T2V-14B cfg-step-distill LoRA (~631MB) that lets 2.2-S2V run in ~4
    steps. Run once BEFORE deploying with WAN_USE_DISTILL=True:
        modal run app.py::download_distill_lora"""
    import os

    from huggingface_hub import hf_hub_download

    dest_dir = constants.WAN_DISTILL_LORA_DIR
    os.makedirs(dest_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=constants.WAN_DISTILL_LORA_REPO,
        filename=constants.WAN_DISTILL_LORA_FILE,
        local_dir=dest_dir,
    )
    weights_volume.commit()
    ok = os.path.exists(constants.WAN_DISTILL_LORA_PATH)
    result = {"status": "success" if ok else "path_mismatch", "downloaded_to": path,
              "expected_path": constants.WAN_DISTILL_LORA_PATH, "expected_exists": ok}
    print(result)
    return result


@app.function(
    image=weights_image,
    volumes={constants.WEIGHTS_ROOT: weights_volume},
)
def verify_weights():
    """Assert the Wan2.2-S2V weights are present with sane total size."""
    import os

    weights_volume.reload()
    ckpt_dir = constants.WAN_CKPT_DIR
    if not os.path.isdir(ckpt_dir):
        result = {"status": "incomplete", "message": f"missing {ckpt_dir} — run download_weights"}
        print(result)
        return result
    total = 0
    files = 0
    for dp, _dn, fns in os.walk(ckpt_dir):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
            files += 1
    ok = total >= 40e9  # expect ~49GB
    result = {"status": "success" if ok else "incomplete", "ckpt_dir": ckpt_dir,
              "files": files, "total_bytes": total, "total_gb": round(total / 1e9, 1)}
    print(result)
    return result

# ---------------------------------------------------------------------------
# Phase 0 health checks — one per image; prove each image builds and runs.
# GPU checks confirm torch actually sees a GPU (Phase 0 acceptance criteria).
# ---------------------------------------------------------------------------

@app.function(image=onboarding_image)
def health_onboarding():
    return {"ok": True, "image": "onboarding_image"}


@app.function(image=director_tts_image)
def health_director_tts():
    return {"ok": True, "image": "director_tts_image"}


@app.function(image=wan_image, gpu=GPU_FUNCTION_A,
              volumes={constants.WEIGHTS_ROOT: weights_volume})
def health_wan():
    """W1 gate: prove the Wan image builds, sees the H100, flash-attn imports, the Wan repo is
    importable, and (if downloaded) the checkpoint dir is present."""
    import os
    import sys

    import torch

    weights_volume.reload()
    if constants.WAN_REPO not in sys.path:
        sys.path.insert(0, constants.WAN_REPO)
    ok_wan = True
    try:
        import wan  # noqa: F401
    except Exception as e:  # noqa: BLE001
        ok_wan = repr(e)
    ok_fa = True
    try:
        import flash_attn  # noqa: F401
    except Exception as e:  # noqa: BLE001
        ok_fa = repr(e)
    return {
        "ok": True,
        "image": "wan_image",
        "cuda_available": torch.cuda.is_available(),
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "wan_import": ok_wan,
        "flash_attn_import": ok_fa,
        "ckpt_present": os.path.isdir(constants.WAN_CKPT_DIR),
    }


@app.function(image=studio_image, secrets=secrets_for("gemini-secret"))
def health_studio():
    import os

    import cv2  # noqa: F401
    import insightface  # noqa: F401

    ok_genai = True
    try:
        from google import genai  # noqa: F401
    except Exception:  # noqa: BLE001
        ok_genai = False
    return {
        "ok": True,
        "image": "studio_image",
        "google_genai_import": ok_genai,
        "gemini_api_key_present": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.local_entrypoint()
def health_all():
    """Run every health check locally: modal run app.py::health_all"""
    print(health_onboarding.remote())
    print(health_director_tts.remote())
    print(health_wan.remote())
    print(health_studio.remote())


# ---------------------------------------------------------------------------
# Register every stage's Modal functions. MUST stay at the bottom (see module docstring).
# ---------------------------------------------------------------------------
import onboard          # noqa: E402,F401  Stage -1 endpoints
import main_generate    # noqa: E402,F401  /generate async orchestration endpoint
import generate_stage   # noqa: E402,F401  Function A (Wan2.2-S2V animation)
import frontend         # noqa: E402,F401  demo control-panel web UI
import reference_studio  # noqa: E402,F401  onboarding-time studio-reference reimagining
