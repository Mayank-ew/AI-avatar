"""
Reference Studio — turn a host's arbitrary onboarding video into a clean, front-facing studio
portrait of the SAME person, used as the animation reference by Function A.

Why this exists: EchoMimicV2 (and every portrait animator) is trained on bright, frontal,
centered, half-body references. Real uploads are dark / side-lit / off-center, which is what
produced the "morphing stranger" output. Instead of demanding users record a perfect video, we
reimagine one good frame into the ideal reference with Google's Gemini 2.5 Flash Image
("nano-banana"), an identity-preserving image editor. The face is preserved; the lighting,
framing, and background are normalized (and can be re-staged into a branded studio scene).

Runs ONCE per host at onboarding (or on a re-roll) — never in the per-generation path — so it
costs one image per host, not per video. Onboarding (onboarding_image) calls the Modal function
here via .remote(), mirroring how it delegates the restorer check to restore.run_restorer_check.

Needs GEMINI_API_KEY in the environment (shipped via the dotenv Modal secret).
"""

import os

import constants
from app import app, studio_image, host_media_volume, secrets_for


class ReferenceStudioError(Exception):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# Best-frame selection — pick the clearest, most frontal face across the clip.
# ---------------------------------------------------------------------------

def _load_face_analyzer():
    from insightface.app import FaceAnalysis

    # detection -> bbox + 5 keypoints (frontality); landmark_3d_68 -> 68 points (eye/mouth ratios).
    analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"])
    # ctx_id=0 uses GPU when present; on a CPU-only container onnxruntime falls back to CPU.
    analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return analyzer


def _eye_aspect_ratio(pts):
    """EAR for one eye from 6 landmark points (p1..p6). ~0.3 = wide open, <0.15 = shut/downcast."""
    import numpy as np

    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return float((a + b) / (2.0 * c + 1e-6))


def pick_best_frame(video_path: str, max_samples: int = None):
    """
    Sample frames across the video and pick the best reference face. A frame must pass quality
    GATES — eyes open (EAR), roughly frontal, not too dark — then we rank survivors by
    det_score * sqrt(area) * capped-sharpness * brightness. If nothing passes the gates, they are
    relaxed step by step (drop brightness, then frontal, then eyes) so we always return a frame.
    """
    import cv2
    import numpy as np

    max_samples = max_samples or constants.STUDIO_BEST_FRAME_SAMPLES

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        idxs = list(range(0, max_samples * 10, 10))
    else:
        step = max(1, total // max_samples)
        idxs = list(range(0, total, step))[:max_samples]

    analyzer = _load_face_analyzer()
    candidates = []   # list of dicts with rgb + metrics
    first_rgb = None

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if first_rgb is None:
            first_rgb = rgb

        h, w = bgr.shape[:2]
        faces = analyzer.get(bgr)
        if not faces:
            continue

        def _area(f):
            a0, b0, a1, b1 = f.bbox
            return max(0.0, a1 - a0) * max(0.0, b1 - b0)

        face = max(faces, key=_area)
        det = float(getattr(face, "det_score", 0.0))
        if det < constants.FACE_MIN_DET_SCORE:
            continue

        x0, y0, x1, y1 = [float(v) for v in face.bbox]
        area_frac = _area(face) / float(w * h)

        kps = np.asarray(face.kps, dtype=np.float32)
        eye_mid = (kps[0] + kps[1]) / 2.0
        inter_eye = float(np.linalg.norm(kps[1] - kps[0])) + 1e-6
        yaw_off = abs(float(kps[2][0]) - float(eye_mid[0])) / inter_eye

        ear = 0.0
        lm = getattr(face, "landmark_3d_68", None)
        if lm is not None:
            lm = np.asarray(lm)[:, :2]
            ear = (_eye_aspect_ratio(lm[36:42]) + _eye_aspect_ratio(lm[42:48])) / 2.0

        xi0, yi0 = max(0, int(x0)), max(0, int(y0))
        xi1, yi1 = min(w, int(x1)), min(h, int(y1))
        crop = bgr[yi0:yi1, xi0:xi1]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        luma = float(gray.mean())
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        rank = (det
                * np.sqrt(area_frac)
                * min(sharp, constants.STUDIO_SHARP_CAP)
                * (luma / 128.0))
        candidates.append({
            "rgb": rgb, "det": det, "area_frac": area_frac, "yaw_off": yaw_off,
            "ear": ear, "luma": luma, "sharp": sharp, "rank": rank,
        })

    cap.release()

    if not candidates:
        if first_rgb is None:
            raise ReferenceStudioError(
                "best_frame", f"could not decode any frame from {video_path}")
        return first_rgb, {"had_face": False}

    # Progressive gating: try the strictest gate set, relax if nothing survives.
    def _passes(c, eyes, frontal, bright):
        if eyes and c["ear"] < constants.STUDIO_EYE_OPEN_MIN:
            return False
        if frontal and c["yaw_off"] > constants.STUDIO_FRONTAL_MAX:
            return False
        if bright and c["luma"] < constants.STUDIO_MIN_FACE_LUMA:
            return False
        return True

    gate_sets = [
        (True, True, True), (True, True, False), (True, False, False), (False, False, False),
    ]
    chosen, gate_used = None, None
    for eyes, frontal, bright in gate_sets:
        survivors = [c for c in candidates if _passes(c, eyes, frontal, bright)]
        if survivors:
            chosen = max(survivors, key=lambda c: c["rank"])
            gate_used = {"eyes": eyes, "frontal": frontal, "bright": bright}
            break

    meta = {
        "had_face": True,
        "gates_applied": gate_used,
        "ear": round(chosen["ear"], 3),
        "yaw_off": round(chosen["yaw_off"], 3),
        "luma": round(chosen["luma"], 1),
        "sharp": round(chosen["sharp"], 1),
        "area_frac": round(chosen["area_frac"], 3),
        "candidates_scanned": len(candidates),
    }
    return chosen["rgb"], meta


# ---------------------------------------------------------------------------
# nano-banana (Gemini 2.5 Flash Image) reimagining.
# ---------------------------------------------------------------------------

def _build_prompt(scene: str) -> str:
    scene = (scene or constants.DEFAULT_STUDIO_SCENE).strip()
    return (
        "You are given a photograph of a real person. Generate a new, photorealistic image of "
        "THE EXACT SAME PERSON. Preserving their identity is the single most important "
        "requirement: keep the same face shape, facial features, eyes, nose, mouth, jawline, "
        "skin tone, hair, facial hair, and apparent age. Do NOT beautify, stylize, slim, "
        "de-age, age, or alter the face in any way — it must be unmistakably the same "
        "individual, as if photographed again in a different room.\n\n"
        f"Re-stage them: {scene}. "
        "Framing: centered, front-facing, looking directly into the camera, from the chest/waist "
        "up (half-body / head-and-shoulders clearly visible). "
        "Lighting: bright, soft, even, flattering studio lighting; the face fully and clearly "
        "lit with no harsh shadows. "
        "Expression: relaxed and confident, mouth closed, eyes open looking at the lens. "
        "Output a single sharp, high-resolution, photorealistic image."
    )


def reimagine_reference(source_rgb, scene: str) -> bytes:
    """Call nano-banana with the source frame + scene prompt; return the generated PNG bytes."""
    import io

    import numpy as np
    from PIL import Image

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ReferenceStudioError(
            "reimagine",
            "GEMINI_API_KEY is not set — add it to your .env (or the gemini-secret). Get a key "
            "at https://aistudio.google.com/apikey.",
        )

    try:
        from google import genai
    except Exception as e:  # noqa: BLE001
        raise ReferenceStudioError("reimagine", f"google-genai SDK import failed: {e!r}")

    src_pil = Image.fromarray(np.asarray(source_rgb)).convert("RGB")
    prompt = _build_prompt(scene)

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=constants.NANO_BANANA_MODEL,
            contents=[prompt, src_pil],
        )
    except Exception as e:  # noqa: BLE001
        raise ReferenceStudioError("reimagine", f"nano-banana request failed: {e!r}")

    # Pull the first inline image part out of the response.
    img_bytes = None
    try:
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    img_bytes = data if isinstance(data, (bytes, bytearray)) else None
                    if img_bytes is None and isinstance(data, str):
                        import base64
                        img_bytes = base64.b64decode(data)
                    break
            if img_bytes:
                break
    except Exception as e:  # noqa: BLE001
        raise ReferenceStudioError("reimagine", f"could not parse nano-banana response: {e!r}")

    if not img_bytes:
        # Most commonly a safety block or a text-only response.
        raise ReferenceStudioError(
            "reimagine",
            "nano-banana returned no image (possibly a safety block or text-only response). "
            "Try a different source video or a simpler scene prompt.",
        )

    # Normalize to PNG.
    out = io.BytesIO()
    Image.open(io.BytesIO(img_bytes)).convert("RGB").save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Modal function — the whole studio pipeline for one host. Called via .remote().
# ---------------------------------------------------------------------------

@app.function(
    image=studio_image,
    timeout=600,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("gemini-secret", "proxy-auth-token"),
)
def build_studio_reference(video_path: str, host_id: str, scene: str = None) -> dict:
    """
    Best-frame -> nano-banana reimagine -> save reference_studio.png on host-media-vol.
    Returns {reference_image_path, source_frame_path, preview_b64, scene, best_frame_meta}.
    Raises on hard failure (caller decides whether to fall back to the raw video frame).
    """
    import base64

    import cv2

    host_media_volume.reload()

    if not os.path.exists(video_path):
        raise ReferenceStudioError("best_frame", f"onboarding video not found at {video_path}")

    source_rgb, meta = pick_best_frame(video_path)
    scene = scene or constants.DEFAULT_STUDIO_SCENE

    # Save the chosen source frame (debugging / lets the panel show "before vs after").
    src_path = constants.studio_source_frame_path(host_id)
    os.makedirs(os.path.dirname(src_path), exist_ok=True)
    cv2.imwrite(src_path, cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))

    png_bytes = reimagine_reference(source_rgb, scene)

    ref_path = constants.studio_reference_path(host_id)
    with open(ref_path, "wb") as f:
        f.write(png_bytes)
    host_media_volume.commit()

    return {
        "reference_image_path": ref_path,
        "source_frame_path": src_path,
        "preview_b64": base64.b64encode(png_bytes).decode("ascii"),
        "scene": scene,
        "best_frame_meta": meta,
    }


@app.function(
    image=studio_image,
    timeout=300,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("postgres-conn-string", "proxy-auth-token"),
)
def crop_reference_to_ratio(host_id: str, ratio: str, source: str = None,
                            register: str = "true") -> dict:
    """
    DYNAMIC fallback crop — force a reference to an EXACT aspect, face-aware, reusing the exact
    same crop the generator uses. Use it when nano-banana (or your hand-made image) isn't already
    the right aspect, so you don't have to size it perfectly. CPU-only (cheap, no GPU) and lets you
    eyeball the framing BEFORE spending a GPU render:
      py -m modal run app.py::crop_reference_to_ratio --host-id <id> --ratio 9:16
      py -m modal volume get host-media-vol <id>/reference_studio_9x16.png .   # eyeball it
    Saves reference_studio_<slug>.png and (by default) registers it for that ratio.
    --source overrides which image to crop (default: the ratio's registered ref, else the default).
    --register false to only produce the file without touching the profile.
    """
    import base64

    from PIL import Image

    import generate_stage  # reuse the generator-side face-aware crop (no divergence)
    import store

    host_media_volume.reload()
    st = store.get_store()
    p = st.get(host_id)
    if p is None:
        return {"status": "error", "message": f"no profile for host_id {host_id!r}"}

    canon = constants.canonical_ratio(ratio)
    src_path = (source
                or (getattr(p, "references_by_ratio", None) or {}).get(canon)
                or p.reference_image_path)
    if not src_path or not os.path.exists(src_path):
        return {"status": "error", "stage": "storage",
                "message": (f"no source reference to crop (looked at {src_path!r}). Set a default "
                            f"reference first with set_reference.")}

    out_path = constants.studio_reference_path(host_id, ratio)
    result_path = generate_stage._prep_reference_for_ratio(src_path, ratio, out_path)
    # If the source was already the right aspect (or the crop bailed), _prep returns it unchanged —
    # copy it to out_path so there's always a stable per-aspect file to register.
    if result_path != out_path:
        Image.open(src_path).convert("RGB").save(out_path)
    host_media_volume.commit()

    registered = str(register).lower() in ("true", "1", "yes")
    if registered:
        rbr = dict(getattr(p, "references_by_ratio", None) or {})
        rbr[canon] = out_path
        p.references_by_ratio = rbr
        st.put(host_id, p)

    with open(out_path, "rb") as f:
        preview_b64 = base64.b64encode(f.read()).decode("ascii")
    printable = {"status": "success", "host_id": host_id, "ratio": canon,
                 "reference_image_path": out_path, "source": src_path, "registered": registered}
    print(printable)  # preview omitted from the CLI print (huge)
    return {**printable, "preview_b64": preview_b64}


@app.function(
    image=studio_image,
    timeout=300,
    volumes={constants.HOST_MEDIA_ROOT: host_media_volume},
    secrets=secrets_for("postgres-conn-string", "proxy-auth-token"),
)
def pick_reference_frame(host_id: str) -> dict:
    """
    Best-frame ONLY — no nano-banana, no Gemini key, no cost. Picks the cleanest frontal,
    eyes-open, well-lit frame from the host's onboarding video and saves it to
    {host_id}/reference_source.png. This is the free bridge for the manual flow: download this
    frame, hand-edit it into a studio portrait (e.g. in the Gemini app), then set_reference.
      py -m modal run app.py::app.pick_reference_frame --host-id <id>
      py -m modal volume get host-media-vol <id>/reference_source.png .
    """
    import store

    host_media_volume.reload()
    p = store.get_store().get(host_id)
    if p is None:
        return {"status": "error", "message": f"no profile for host_id {host_id!r}"}
    if not os.path.exists(p.base_video_path):
        return {"status": "error", "message": f"onboarding video missing at {p.base_video_path}"}

    import cv2

    source_rgb, meta = pick_best_frame(p.base_video_path)
    src_path = constants.studio_source_frame_path(host_id)
    os.makedirs(os.path.dirname(src_path), exist_ok=True)
    cv2.imwrite(src_path, cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
    host_media_volume.commit()

    result = {"status": "success", "host_id": host_id, "source_frame_path": src_path,
              "best_frame_meta": meta}
    print(result)
    return result
