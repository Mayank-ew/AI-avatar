"""
/generate orchestration — ASYNC job flow.  docs/04 §2, WAN_MIGRATION_PLAN.md §4.

Wan2.2-S2V takes minutes per clip, so generation can't be a synchronous HTTP request. Flow:

  POST /generate  -> director (Stage 0) + TTS (Stage A) [fast] -> write audio.mp3 ->
                     .spawn() Function A (Wan, slow) -> store job -> return {job_id} immediately
  GET  /status/{job_id} -> poll the spawned call: pending | done (+ video_url) | error
  GET  /video/{job_id}  -> stream the finished mp4 (binary; not base64 — robust for any size)

Runs on director_tts_image (thin, CPU). Intermediate artifacts live on the shared intermediate
volume under a per-request run_id (== job_id). Function A writes final_output.mp4 there.
"""

import os

import constants
import store
from app import (
    app,
    director_tts_image,
    intermediate_volume,
    generation_jobs,
    secrets_for,
)

_INTER = constants.INTERMEDIATE_ROOT


def _error(stage: str, message: str, detail=None):
    return {"status": "error", "stage": stage, "message": message, "detail": detail}


def _prepare_and_spawn(payload: dict) -> tuple[dict, int]:
    """Director + TTS (fast), then spawn the slow Wan Function A and return a job id."""
    import uuid

    import director
    import tts
    import generate_stage

    # --- Validate ---
    host_id = payload.get("host_id")
    script_text = payload.get("script_text")
    target_ratio = payload.get("target_ratio")
    if not host_id or not script_text or not target_ratio:
        return _error("profile_lookup",
                      "host_id, script_text and target_ratio are all required"), 400

    enable_director = payload.get("enable_director", True)
    voice_character_hint = payload.get("voice_character_hint")
    model_override = payload.get("elevenlabs_model_override")
    wan_prompt = payload.get("wan_prompt")  # optional visual-scene prompt override
    tts_provider = (payload.get("tts_provider") or constants.TTS_PROVIDER_DEFAULT).lower()
    fish_reference_id = payload.get("fish_reference_id")
    is_fish = tts_provider in ("fishaudio", "fish")

    # --- Profile lookup ---
    try:
        profile = store.get_store().get(host_id)
    except Exception as e:  # noqa: BLE001
        return _error("profile_lookup", f"profile store error: {e}"), 500
    if profile is None:
        return _error("profile_lookup", "host_id has no onboarded profile"), 404

    # Prefer a per-aspect reference (e.g. a reel-framed 9:16 portrait) when one is registered for
    # the requested ratio; otherwise use the default reference (Wan then crops it to the aspect).
    refs_by_ratio = getattr(profile, "references_by_ratio", None) or {}
    canon_ratio = constants.canonical_ratio(target_ratio)
    reference_image_path = refs_by_ratio.get(canon_ratio) or getattr(
        profile, "reference_image_path", None)
    if not reference_image_path:
        return _error("profile_lookup",
                      "this host has no studio reference image — re-onboard (or run "
                      "set_reference) so a reference_image_path exists"), 422

    voice_id = profile.voice_id
    hint = voice_character_hint or profile.voice_character_hint
    run_id = uuid.uuid4().hex
    run_dir = f"{_INTER}/{run_id}"

    # --- Stage 0: director --- (ElevenLabs-only; Fish ignores [tags], so skip it for Fish)
    directed_text = script_text
    stability_hint = "natural"
    clean_text = script_text
    if enable_director and not is_fish:
        try:
            directed = director.direct_script(script_text, hint)
            clean_text = directed["clean_text"]
            directed_text = directed["directed_text"]
            stability_hint = directed["stability_hint"]
        except director.DirectorGuardrailError as e:
            return _error("director", str(e)), 422
        except Exception as e:  # noqa: BLE001
            return _error("director", f"director call failed: {e}"), 502

    # --- Stage A: TTS ---
    try:
        audio_bytes, tts_meta = tts.synthesize(
            voice_id=voice_id,
            directed_text=directed_text,
            clean_text=clean_text,
            stability_hint=stability_hint,
            enable_director=enable_director,
            model_override=model_override,
            provider=tts_provider,
            fish_reference_id=fish_reference_id,
        )
    except tts.CharacterLimitError as e:
        return _error("tts", str(e)), 422
    except tts.TTSError as e:
        return _error("tts", str(e)), 502
    except Exception as e:  # noqa: BLE001
        return _error("tts", f"tts failed: {e}"), 502

    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/audio.mp3", "wb") as f:
        f.write(audio_bytes)
    intermediate_volume.commit()

    # --- Spawn Function A (Wan, slow, snapshotted @app.cls) and return a job id immediately ---
    try:
        call = generate_stage.WanAvatar().run.spawn(
            run_id, reference_image_path, target_ratio, wan_prompt)
    except Exception as e:  # noqa: BLE001
        return _error("animation", f"failed to spawn Wan Function A: {e}"), 502

    generation_jobs[run_id] = {
        "status": "pending",
        "call_id": call.object_id,
        "run_id": run_id,
        "host_id": host_id,
        "target_ratio": target_ratio,
        "directed_text": directed_text,
        "stability_hint": stability_hint,
        "tts_model": tts_meta["model_id"],
        "tts_provider": tts_meta.get("provider"),
    }
    return {"status": "submitted", "job_id": run_id, "run_id": run_id}, 202


def _job_status(job_id: str) -> tuple[dict, int]:
    import modal

    rec = generation_jobs.get(job_id)
    if rec is None:
        return _error("status", "unknown job_id"), 404
    if rec.get("status") == "error":
        return {"status": "error", "job_id": job_id, "stage": "animation",
                "message": rec.get("message", "generation failed")}, 200

    try:
        call = modal.FunctionCall.from_id(rec["call_id"])
        meta = call.get(timeout=0)
    except TimeoutError:
        return {"status": "pending", "job_id": job_id}, 200
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["message"] = str(e)
        generation_jobs[job_id] = rec
        return {"status": "error", "job_id": job_id, "stage": "animation",
                "message": f"Wan Function A failed: {e}"}, 200

    # Done. The mp4 is on the volume; the client fetches it from GET /video/{job_id} (STREAMED),
    # not as base64 in this JSON — base64 bloats the payload ~1.33x and made longer clips fail to
    # render in the panel. This status stays small and reliable.
    final_path = meta.get("final_output_path")
    rec.update(status="done", resolution=meta.get("resolution"),
               duration_seconds=meta.get("duration_seconds"), final_output_path=final_path)
    generation_jobs[job_id] = rec
    return {
        "status": "done",
        "job_id": job_id,
        "run_id": rec["run_id"],
        "video_url": f"/video/{job_id}",
        "video_volume_path": final_path,
        "resolution": meta.get("resolution"),
        "duration_seconds": meta.get("duration_seconds"),
        "directed_text": rec.get("directed_text"),
        "stability_hint": rec.get("stability_hint"),
        "tts_model": rec.get("tts_model"),
        "tts_provider": rec.get("tts_provider"),
        "target_ratio": rec.get("target_ratio"),
        "backend": meta.get("backend"),
    }, 200


import modal  # noqa: E402


@app.function(
    image=director_tts_image,
    timeout=900,
    volumes={constants.INTERMEDIATE_ROOT: intermediate_volume},
    secrets=secrets_for("elevenlabs-secret", "openai-secret",
                        "postgres-conn-string", "proxy-auth-token"),
)
@modal.asgi_app(requires_proxy_auth=True)
def generate():
    from fastapi import FastAPI, Header, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse

    web = FastAPI(title="EasyWebinar Avatar — Generate (async)")

    def _check_bearer(authorization):
        from fastapi import HTTPException

        expected = os.environ.get("APP_BEARER_TOKEN")
        if expected is None:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid or missing app bearer token")

    @web.post("/generate")
    async def generate_route(request: Request, authorization: str = Header(None)):
        _check_bearer(authorization)
        payload = await request.json()
        body, code = await run_in_threadpool(_prepare_and_spawn, payload)
        return JSONResponse(status_code=code, content=body)

    @web.get("/status/{job_id}")
    async def status_route(job_id: str, authorization: str = Header(None)):
        _check_bearer(authorization)
        body, code = await run_in_threadpool(_job_status, job_id)
        return JSONResponse(status_code=code, content=body)

    @web.get("/video/{job_id}")
    async def video_route(job_id: str, authorization: str = Header(None)):
        """Stream the finished mp4 off the intermediate volume (path is deterministic from job_id)."""
        _check_bearer(authorization)
        from fastapi.responses import FileResponse

        def _resolve():
            intermediate_volume.reload()
            path = f"{_INTER}/{job_id}/final_output.mp4"
            return path if os.path.exists(path) else None

        path = await run_in_threadpool(_resolve)
        if not path:
            return JSONResponse(status_code=404,
                                content={"status": "error", "message": "video not ready/found"})
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

    @web.get("/profile/{host_id}")
    async def profile_route(host_id: str, authorization: str = Header(None)):
        _check_bearer(authorization)
        try:
            p = await run_in_threadpool(lambda: store.get_store().get(host_id))
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        if p is None:
            return JSONResponse(status_code=404,
                                content={"status": "error", "message": "no profile for that host_id"})
        return JSONResponse(content={"status": "success", **p.to_dict()})

    @web.get("/hosts")
    async def hosts_route(authorization: str = Header(None)):
        _check_bearer(authorization)
        try:
            ids = await run_in_threadpool(lambda: store.get_store().list_host_ids())
            return JSONResponse(content={"status": "success", "host_ids": ids})
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    return web
