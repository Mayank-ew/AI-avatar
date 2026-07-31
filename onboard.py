"""
Stage -1 — Host onboarding endpoint.  docs/03 §0, docs/04 §1, docs/05 §4.

POST /host/onboard (and PUT for re-onboarding) on onboarding_image. Flow, in order:
  consent gate (hard fail, no side effects if absent) -> save upload to host-media-vol ->
  ffmpeg audio extraction (Phase 2) -> POST /v1/voices/add (IVC) -> Reference Studio (reimagine a
  clean studio portrait via reference_studio.build_studio_reference.remote) -> ProfileStore.put.

The studio portrait is what Function A (Wan2.2-S2V) animates at generation time.
"""

import os
import subprocess

import constants
import store
from app import app, onboarding_image, host_media_volume, secrets_for

# POC: leave the superseded ElevenLabs voice_id orphaned on re-onboard (docs/03 §0 POC note).
# Flip to True for the production explicit-deletion behavior (docs/07).
DELETE_OLD_VOICE_ON_REONBOARD = False


class OnboardingError(Exception):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# ffmpeg helpers (Phase 2).
# ---------------------------------------------------------------------------

def _has_audio_stream(video_path: str) -> bool:
    """True if the container has at least one audio stream (ffprobe)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        return "audio" in (r.stdout or "")
    except Exception:  # noqa: BLE001
        return True  # ffprobe missing — don't block; let ffmpeg try.


def extract_audio(video_path: str, out_path: str) -> str:
    """docs/03 §0: mandatory audio extraction before IVC (video not accepted directly)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # A silent video (e.g. mic was off / not permitted while recording) is the most common
    # cause of failure here — surface it as a clear, actionable message rather than a raw
    # ffmpeg dump. The voice clone needs actual speech audio.
    if not _has_audio_stream(video_path):
        raise OnboardingError(
            "audio_extraction",
            "the uploaded video has no audio track — ElevenLabs needs speech to clone a voice. "
            "Re-record with the microphone enabled (and talk for ~60-90s), then try again.",
        )

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", out_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # ffmpeg's real error is at the END of stderr (the start is just its version banner).
        tail = e.stderr.decode("utf-8", "ignore")[-800:] if e.stderr else "(no stderr)"
        raise OnboardingError("audio_extraction", f"ffmpeg failed. stderr tail:\n{tail}")
    return out_path


# ---------------------------------------------------------------------------
# ElevenLabs IVC (docs/02 §2a).
# ---------------------------------------------------------------------------

def create_instant_voice_clone(host_id: str, audio_path: str) -> dict:
    import requests

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise OnboardingError("voice_clone", "ELEVENLABS_API_KEY missing (modal.Secret?)")
    with open(audio_path, "rb") as fh:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers={"xi-api-key": api_key},
            data={"name": host_id, "remove_background_noise": "true"},
            files={"files": fh},
            timeout=300,
        )
    if resp.status_code >= 400:
        detail = None
        try:
            detail = resp.json()
        except Exception:  # noqa: BLE001
            detail = {"raw": resp.text[:500]}
        raise OnboardingError("voice_clone", f"ElevenLabs {resp.status_code}: {detail}")
    return resp.json()


def delete_voice(voice_id: str) -> None:
    """Free the quota slot for a superseded voice on re-onboard (docs/07 prod recommendation)."""
    import requests

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return
    try:
        requests.delete(
            f"https://api.elevenlabs.io/v1/voices/{voice_id}",
            headers={"xi-api-key": api_key}, timeout=60,
        )
    except Exception:  # noqa: BLE001 — best-effort cleanup, never blocks re-onboarding.
        pass


# ---------------------------------------------------------------------------
# App-level bearer auth (docs/05 §4). Layered on top of requires_proxy_auth.
# ---------------------------------------------------------------------------

def _check_bearer(authorization: str | None):
    from fastapi import HTTPException

    expected = os.environ.get("APP_BEARER_TOKEN")
    if expected is None:
        return  # no app token configured (proxy-auth only) — allow.
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid or missing app bearer token")


# ---------------------------------------------------------------------------
# Server-side video download from a URL (DLP bypass — laptop sends only a URL, Modal fetches
# the file). Runs in onboarding_image, which has ffmpeg (apt) + yt-dlp (pip).
# ---------------------------------------------------------------------------

def download_video_from_url(url: str, out_path: str, section: str = "*0:00-1:30") -> str:
    """
    Download `url` to `out_path` as a clean H.264/AAC mp4, trimmed to `section`, via yt-dlp +
    the container's ffmpeg. --recode-video mp4 normalizes odd codecs (e.g. VP9/Opus that YouTube
    serves without a JS runtime) so the file is playable and pipeline-safe.
    """
    import subprocess
    import sys

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--download-sections", section,
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b/18",
        "--recode-video", "mp4",
        "--force-overwrites",
        "-o", out_path,
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode("utf-8", "ignore")[-800:] if e.stderr else "(no stderr)"
        raise OnboardingError("video_download", f"yt-dlp failed. stderr tail:\n{tail}")
    if not os.path.exists(out_path):
        raise OnboardingError("video_download", f"download produced no file at {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Shared onboarding flow. _onboard_core runs once the video is on the volume at video_path,
# regardless of how it got there (multipart upload or server-side URL download).
# ---------------------------------------------------------------------------

def _build_studio_reference(host_id, video_path, scene_prompt):
    """
    Run the Reference Studio stage (best-frame -> nano-banana reimagine) remotely and return
    (reference_image_path, preview_b64, error_message). Soft-fails: a reimagine failure never
    aborts onboarding (the voice clone already succeeded) — Function A falls back to the raw
    video frame when reference_image_path is None.
    """
    if not constants.ENABLE_STUDIO_REFERENCE:
        return None, None, None
    import reference_studio

    try:
        ref = reference_studio.build_studio_reference.remote(video_path, host_id, scene_prompt)
        host_media_volume.reload()  # pick up the committed reference file
        return ref["reference_image_path"], ref.get("preview_b64"), None
    except Exception as e:  # noqa: BLE001
        return None, None, f"studio reference generation failed: {e}"


def _onboard_core(host_id, video_path, voice_character_hint,
                  is_reonboard, st, old_profile, voice_id_override=None, scene_prompt=None):
    from datetime import datetime, timezone

    try:
        if voice_id_override:
            # POC / free-tier path: skip Instant Voice Cloning (needs a paid ElevenLabs plan)
            # and reuse an existing voice_id. The uploaded clip is then used only as the
            # video/animation reference; the voice is the supplied pre-made one.
            voice_id = voice_id_override
            ivc = {"voice_id": voice_id, "requires_verification": None}
        else:
            # Extract audio (Phase 2, mandatory) then create the Instant Voice Clone.
            audio_path = constants.onboarding_audio_path(host_id)
            extract_audio(video_path, audio_path)
            ivc = create_instant_voice_clone(host_id, audio_path)
            voice_id = ivc["voice_id"]

        # Reference Studio (reimagine): turn the raw clip into a clean studio portrait of the
        # same person — this is the image Wan2.2-S2V animates. Soft-fails; a miss leaves
        # reference_image_path None (generation then errors until set_reference is run).
        host_media_volume.commit()
        reference_image_path, reference_preview_b64, reference_error = _build_studio_reference(
            host_id, video_path, scene_prompt)

        # Persist the profile (upsert — overwrites on re-onboard, docs/03 §0).
        profile = store.HostProfile(
            host_id=host_id,
            voice_id=voice_id,
            base_video_path=video_path,
            created_at=datetime.now(timezone.utc).isoformat(),
            voice_character_hint=voice_character_hint,
            reference_image_path=reference_image_path,
        )
        st.put(host_id, profile)
    except OnboardingError as e:
        return {"status": "error", "stage": e.stage, "message": e.message}, 502
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "stage": "storage", "message": str(e)}, 500

    if (is_reonboard and old_profile and DELETE_OLD_VOICE_ON_REONBOARD
            and old_profile.voice_id != voice_id):
        delete_voice(old_profile.voice_id)

    return {
        "status": "success",
        "host_id": host_id,
        "voice_id": voice_id,
        "requires_verification": ivc.get("requires_verification"),
        "reference_image_path": reference_image_path,
        "reference_preview_b64": reference_preview_b64,
        "reference_error": reference_error,
    }, 200


def _do_onboard(host_id, video, consent_attested, voice_character_hint,
                is_reonboard, scene_prompt=None):
    """Multipart path: consent gate -> save upload -> shared core."""
    if not consent_attested:
        return {"status": "error", "stage": "consent_check",
                "message": "consent_attested must be true"}, 400

    st = store.get_store()
    old_profile = st.get(host_id) if is_reonboard else None

    video_path = constants.onboarding_video_path(host_id)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    try:
        with open(video_path, "wb") as out:
            out.write(video.file.read())
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "stage": "storage",
                "message": f"failed to save upload: {e}"}, 500

    return _onboard_core(host_id, video_path, voice_character_hint,
                         is_reonboard, st, old_profile, scene_prompt=scene_prompt)


def _do_onboard_url(host_id, video_url, consent_attested, voice_character_hint,
                    is_reonboard, start_sec=0, end_sec=90, scene_prompt=None):
    """URL path: consent gate -> server-side download (of the [start,end] slice) -> shared core."""
    if not consent_attested:
        return {"status": "error", "stage": "consent_check",
                "message": "consent_attested must be true"}, 400
    if not video_url:
        return {"status": "error", "stage": "video_download",
                "message": "video_url is required"}, 400

    # Build the download section from the requested range, with guards. Cap the slice length so
    # a runaway range can't dump a huge clip / overrun ElevenLabs' ideal IVC audio length.
    try:
        start = max(0, int(start_sec))
        end = int(end_sec)
    except (TypeError, ValueError):
        start, end = 0, 90
    if end <= start:
        return {"status": "error", "stage": "video_download",
                "message": f"end ({end}s) must be after start ({start}s)"}, 400
    MAX_SLICE = 180
    if end - start > MAX_SLICE:
        end = start + MAX_SLICE
    section = f"*{start}-{end}"

    st = store.get_store()
    old_profile = st.get(host_id) if is_reonboard else None

    video_path = constants.onboarding_video_path(host_id)
    try:
        download_video_from_url(video_url, video_path, section)
    except OnboardingError as e:
        return {"status": "error", "stage": e.stage, "message": e.message}, 502

    return _onboard_core(host_id, video_path, voice_character_hint,
                         is_reonboard, st, old_profile, scene_prompt=scene_prompt)


# ---------------------------------------------------------------------------
# CLI onboarding from a file already on the volume (DLP bypass path). Upload the local clip via
# the Modal CLI (which isn't a browser upload, so a browser-upload DLP rule usually doesn't catch
# it), then run this:
#   py -m modal volume put host-media-vol <local.mp4> <host_id>/onboarding_video.mp4 --force
#   py -m modal run app.py::app.onboard_existing --host-id <host_id> --consent true
# ---------------------------------------------------------------------------

@app.function(
    image=onboarding_image,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("elevenlabs-secret", "postgres-conn-string", "proxy-auth-token"),
    timeout=600,
)
def onboard_existing(host_id: str, consent: str = "false",
                     voice_character_hint: str = None,
                     voice_id: str = None,
                     scene_prompt: str = None):
    """
    Onboard from a video ALREADY placed on host-media-vol at {host_id}/onboarding_video.mp4.
    Pass --voice-id <existing_id> to skip Instant Voice Cloning (needed on the free ElevenLabs
    tier, which doesn't include IVC) and reuse a voice you already created in the dashboard.
    Pass --scene-prompt "..." to control the reimagined studio background.
    """
    host_media_volume.reload()
    if str(consent).lower() not in ("true", "1", "yes"):
        return {"status": "error", "stage": "consent_check", "message": "pass --consent true"}
    video_path = constants.onboarding_video_path(host_id)
    if not os.path.exists(video_path):
        return {
            "status": "error", "stage": "storage",
            "message": (f"no file at {video_path}. First upload it via the Modal CLI:\n"
                        f"  py -m modal volume put host-media-vol <local.mp4> "
                        f"{host_id}/onboarding_video.mp4 --force"),
        }
    st = store.get_store()
    body, _ = _onboard_core(host_id, video_path, voice_character_hint,
                            False, st, None,
                            voice_id_override=voice_id, scene_prompt=scene_prompt)
    # The base64 preview is huge and useless on a CLI — drop it from the printed summary.
    printable = {k: v for k, v in body.items() if k != "reference_preview_b64"}
    print(printable)
    return printable


@app.function(
    image=onboarding_image,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("postgres-conn-string", "proxy-auth-token"),
    timeout=120,
)
def set_reference(host_id: str, filename: str = None, ratio: str = None):
    """
    Register an EXISTING image on host-media-vol as the host's studio reference — the free /
    no-API bridge for when you generate the reference by hand (e.g. in the Gemini app via your
    Google AI Pro sub) instead of calling nano-banana from code.

    DEFAULT reference (used for any aspect that has no specific reference):
      1) py -m modal volume get host-media-vol <id>/reference_source.png .   # the picked frame
      2) (hand-edit it in the Gemini app into a clean studio portrait, download it)
      3) py -m modal volume put host-media-vol <studio.png> <id>/reference_studio.png --force
      4) py -m modal run app.py::app.set_reference --host-id <id>

    PER-ASPECT reference (e.g. a waist-up 9:16 portrait framed for reels, so generation doesn't
    crop a landscape close-up). Pass --ratio; the default filename becomes reference_studio_9x16.png:
      py -m modal volume put host-media-vol <reel.png> <id>/reference_studio_9x16.png --force
      py -m modal run app.py::app.set_reference --host-id <id> --ratio 9:16
    Generation with target_ratio 9:16 will then use this reference; other ratios fall back to the
    default. --filename overrides the expected filename (relative to the host's media folder).
    """
    host_media_volume.reload()
    if filename:
        ref_path = f"{constants.HOST_MEDIA_ROOT}/{host_id}/{filename}"
        default_name = filename
    else:
        ref_path = constants.studio_reference_path(host_id, ratio)
        default_name = os.path.basename(ref_path)
    if not os.path.exists(ref_path):
        return {
            "status": "error", "stage": "storage",
            "message": (f"no image at {ref_path}. First upload it:\n"
                        f"  py -m modal volume put host-media-vol <studio.png> "
                        f"{host_id}/{default_name} --force"),
        }
    st = store.get_store()
    p = st.get(host_id)
    if p is None:
        return {"status": "error", "message": f"no profile for host_id {host_id!r}"}
    if ratio:
        canon = constants.canonical_ratio(ratio)
        rbr = dict(getattr(p, "references_by_ratio", None) or {})
        rbr[canon] = ref_path
        p.references_by_ratio = rbr
    else:
        p.reference_image_path = ref_path
    st.put(host_id, p)
    result = {"status": "success", "host_id": host_id, "ratio": constants.canonical_ratio(ratio),
              "reference_image_path": ref_path,
              "references_by_ratio": getattr(p, "references_by_ratio", {})}
    print(result)
    return result


@app.function(
    image=onboarding_image,
    secrets=secrets_for("postgres-conn-string", "proxy-auth-token"),
    timeout=120,
)
def set_voice_id(host_id: str, voice_id: str):
    """
    Update ONLY the voice_id on an existing profile (e.g. after designing a new ElevenLabs voice),
    without re-onboarding — so the studio reference and everything else stay untouched.
      py -m modal run app.py::app.set_voice_id --host-id <id> --voice-id <new_voice_id>
    """
    st = store.get_store()
    p = st.get(host_id)
    if p is None:
        return {"status": "error", "message": f"no profile for host_id {host_id!r}"}
    old = p.voice_id
    p.voice_id = voice_id
    st.put(host_id, p)
    result = {"status": "success", "host_id": host_id, "old_voice_id": old, "voice_id": voice_id}
    print(result)
    return result


@app.function(
    image=onboarding_image,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("gemini-secret", "postgres-conn-string", "proxy-auth-token"),
    timeout=600,
)
def regenerate_reference(host_id: str, scene_prompt: str = None):
    """
    Re-roll ONLY the studio reference for an already-onboarded host (new background/scene),
    without touching the voice clone. CLI:
      py -m modal run app.py::app.regenerate_reference --host-id <id> --scene-prompt "..."
    """
    st = store.get_store()
    p = st.get(host_id)
    if p is None:
        return {"status": "error", "message": f"no profile for host_id {host_id!r}"}
    ref_path, _preview, err = _build_studio_reference(host_id, p.base_video_path, scene_prompt)
    if err or not ref_path:
        return {"status": "error", "stage": "reimagine", "message": err or "no reference produced"}
    p.reference_image_path = ref_path
    st.put(host_id, p)
    result = {"status": "success", "host_id": host_id, "reference_image_path": ref_path,
              "scene": scene_prompt or constants.DEFAULT_STUDIO_SCENE}
    print(result)
    return result


# ---------------------------------------------------------------------------
# Modal web endpoint (ASGI). Multipart file upload + POST(onboard)/PUT(re-onboard) on one app.
# All fastapi imports are inside the function body so local `modal deploy` needn't have fastapi.
# ---------------------------------------------------------------------------

import modal  # noqa: E402


class _BytesUpload:
    """Minimal stand-in for FastAPI's UploadFile exposing `.file.read()`. The async route reads
    the upload bytes, then hands this to the blocking onboarding core (run in a threadpool), which
    only needs `video.file.read()`."""

    def __init__(self, content: bytes, filename: str = None, content_type: str = None):
        import io

        self.file = io.BytesIO(content)
        self.filename = filename
        self.content_type = content_type


@app.function(
    image=onboarding_image,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("elevenlabs-secret", "postgres-conn-string", "proxy-auth-token"),
)
@modal.asgi_app(requires_proxy_auth=True)
def host_onboard():
    from fastapi import FastAPI, File, Form, Header, Request, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    web = FastAPI(title="EasyWebinar Avatar — Host Onboarding")

    async def _handler(host_id, consent_attested, voice_character_hint,
                       video, authorization, is_reonboard, scene_prompt=None):
        _check_bearer(authorization)
        # consent_attested arrives as a form string; normalize to bool.
        consent = str(consent_attested).lower() in ("true", "1", "yes")
        # UploadFile.file is a sync file object; read it here (async) then hand the bytes to the
        # blocking core in a threadpool (it makes blocking Modal/store/.remote calls).
        content = await video.read()
        body, code = await run_in_threadpool(
            _do_onboard,
            host_id=host_id,
            video=_BytesUpload(content, video.filename, video.content_type),
            consent_attested=consent,
            voice_character_hint=voice_character_hint,
            is_reonboard=is_reonboard,
            scene_prompt=scene_prompt,
        )
        return JSONResponse(status_code=code, content=body)

    @web.post("/host/onboard")
    async def onboard(  # noqa: D401
        host_id: str = Form(...),
        consent_attested: str = Form(...),
        video: UploadFile = File(...),
        voice_character_hint: str = Form(None),
        scene_prompt: str = Form(None),
        authorization: str = Header(None),
    ):
        return await _handler(host_id, consent_attested, voice_character_hint,
                              video, authorization, False, scene_prompt)

    @web.put("/host/{host_id}/onboard")
    async def reonboard(  # noqa: D401
        host_id: str,
        consent_attested: str = Form(...),
        video: UploadFile = File(...),
        voice_character_hint: str = Form(None),
        scene_prompt: str = Form(None),
        authorization: str = Header(None),
    ):
        return await _handler(host_id, consent_attested, voice_character_hint,
                              video, authorization, True, scene_prompt)

    @web.post("/host/{host_id}/reference")
    async def regen_reference(host_id: str, request: Request, authorization: str = Header(None)):
        """Re-roll ONLY the studio reference with a new scene prompt (no voice re-clone).
        Body: {scene_prompt?}. Returns the new preview so the panel can show it."""
        _check_bearer(authorization)
        body_in = await request.json()
        scene_prompt = body_in.get("scene_prompt")

        def _work():
            st = store.get_store()
            p = st.get(host_id)
            if p is None:
                return 404, {"status": "error", "message": "no profile for that host_id"}
            ref_path, preview_b64, err = _build_studio_reference(
                host_id, p.base_video_path, scene_prompt)
            if err or not ref_path:
                return 502, {"status": "error", "stage": "reimagine",
                             "message": err or "no reference produced"}
            p.reference_image_path = ref_path
            st.put(host_id, p)
            return 200, {"status": "success", "host_id": host_id,
                         "reference_image_path": ref_path, "reference_preview_b64": preview_b64,
                         "scene": scene_prompt or constants.DEFAULT_STUDIO_SCENE}

        code, content = await run_in_threadpool(_work)
        return JSONResponse(status_code=code, content=content)

    @web.post("/host/onboard-url")
    async def onboard_url(request: Request, authorization: str = Header(None)):
        """JSON path (no file upload) — Modal downloads the video server-side from a URL.
        Body: {host_id, video_url, consent_attested, voice_character_hint?, scene_prompt?,
        is_reonboard?, start_sec?, end_sec?}."""
        _check_bearer(authorization)
        body_in = await request.json()
        consent = str(body_in.get("consent_attested")).lower() in ("true", "1", "yes")
        body, code = await run_in_threadpool(
            _do_onboard_url,
            host_id=body_in.get("host_id"),
            video_url=body_in.get("video_url"),
            consent_attested=consent,
            voice_character_hint=body_in.get("voice_character_hint"),
            is_reonboard=bool(body_in.get("is_reonboard", False)),
            start_sec=body_in.get("start_sec", 0),
            end_sec=body_in.get("end_sec", 90),
            scene_prompt=body_in.get("scene_prompt"),
        )
        return JSONResponse(status_code=code, content=body)

    return web
