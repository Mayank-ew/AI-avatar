"""
Function A (GPU) — Wan2.2-S2V-14B audio-driven animation, with MEMORY SNAPSHOTS.

Runs on `wan_image` (H100) as a snapshotted @app.cls. The ~44GB `wan.WanS2V` pipeline is built
ONCE inside @modal.enter(snap=True); Modal captures that state (GPU included, via the alpha
enable_gpu_snapshot option) into a memory snapshot. Cold starts then RESTORE the snapshot instead
of re-reading 44GB off the volume — so cold starts are cheap and we can keep the idle window
short (min idle-GPU cost) while still avoiding the multi-minute reload on every call.

Input: the reimagined studio portrait (reference_studio.py) + the Stage-A ElevenLabs voice.
Output: /intermediate/{run_id}/final_output.mp4 with audio muxed in. There is no Function B.

Heavy imports (torch/wan) live inside the methods so importing this module in the thin CPU
containers stays cheap.
"""

import json
import math
import os
import subprocess

import modal

import constants

from app import (
    app,
    wan_image,
    GPU_FUNCTION_A,
    host_media_volume,
    intermediate_volume,
    weights_volume,
)

_INTER = constants.INTERMEDIATE_ROOT


class WanError(Exception):
    pass


def _run_dir(run_id: str) -> str:
    return f"{_INTER}/{run_id}"


def _build_pipe():
    """Construct the WanS2V pipeline (loads ~44GB). Runs inside the snapshotted enter."""
    import sys

    if constants.WAN_REPO not in sys.path:
        sys.path.insert(0, constants.WAN_REPO)
    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS[constants.WAN_TASK]
    print(f"Wan: building WanS2V from {constants.WAN_CKPT_DIR} (snapshotted, one-time)…",
          flush=True)
    pipe = wan.WanS2V(
        config=cfg,
        checkpoint_dir=constants.WAN_CKPT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=constants.WAN_T5_CPU,
        init_on_cpu=True,
        convert_model_dtype=constants.WAN_CONVERT_DTYPE,
    )
    print("Wan: pipeline built.", flush=True)
    return pipe, cfg


def _load_lora_into_model(model, lora_path: str, strength: float):
    """
    Merge a LoRA (safetensors) into `model`'s Linear weights IN PLACE, format-tolerantly.

    Handles both key conventions (lora_A/lora_B and lora_down/lora_up), strips the usual
    prefixes (diffusion_model. / transformer. / model.), and matches each LoRA pair to a module
    by its dotted path (e.g. "blocks.0.self_attn.q"). Returns (matched, missed) so the caller can
    detect a key-format mismatch (matched==0) without running a generation. Weights are 2.2-S2V's
    own; the T2V LoRA only covers the shared DiT blocks (self_attn/cross_attn/ffn) — audio-path
    modules simply have no LoRA keys and stay stock.
    """
    import torch
    from safetensors.torch import load_file

    sd = load_file(lora_path)
    pairs: dict = {}
    for k, v in sd.items():
        kk = k
        for pref in ("diffusion_model.", "transformer.", "model."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        if kk.endswith(".lora_A.weight") or kk.endswith(".lora_down.weight"):
            base = kk.rsplit(".lora_", 1)[0]
            pairs.setdefault(base, {})["down"] = v
        elif kk.endswith(".lora_B.weight") or kk.endswith(".lora_up.weight"):
            base = kk.rsplit(".lora_", 1)[0]
            pairs.setdefault(base, {})["up"] = v
        elif kk.endswith(".alpha"):
            base = kk[: -len(".alpha")]
            pairs.setdefault(base, {})["alpha"] = float(v.reshape(-1)[0].item())

    named = dict(model.named_modules())
    matched = missed = 0
    for base, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            continue
        target = named.get(base)
        if target is None or not hasattr(target, "weight") or target.weight.dim() != 2:
            missed += 1
            continue
        down = parts["down"].to(torch.float32)   # [rank, in]
        up = parts["up"].to(torch.float32)        # [out, rank]
        rank = down.shape[0]
        alpha = parts.get("alpha", rank)
        scale = strength * (alpha / rank)
        delta = (up @ down) * scale               # [out, in]
        W = target.weight
        if tuple(delta.shape) != tuple(W.shape):
            missed += 1
            continue
        with torch.no_grad():
            W.data.add_(delta.to(W.dtype).to(W.device))
        matched += 1
    return matched, missed


def _maybe_apply_distill(pipe) -> bool:
    """Apply the cfg-step-distill LoRA to the DiT so it can run in ~4 steps. Returns True only if
    the LoRA actually merged; on any failure we log and return False so run() falls back to the
    stock step count (never run 4 steps on a NON-distilled model — that would be garbage)."""
    if not os.path.exists(constants.WAN_DISTILL_LORA_PATH):
        print(f"Wan distill: LoRA missing at {constants.WAN_DISTILL_LORA_PATH} — run "
              f"`modal run app.py::download_distill_lora`. Falling back to stock steps.", flush=True)
        return False
    # The DiT is WanS2V.noise_model (confirmed in speech2video.py); keep fallbacks in case a
    # future repo rev renames it.
    model = next((getattr(pipe, n) for n in ("noise_model", "model", "dit", "transformer")
                  if getattr(pipe, n, None) is not None), None)
    if model is None:
        print("Wan distill: could not find the DiT on the pipe (tried noise_model/model/dit/"
              "transformer) — cannot apply LoRA; stock steps.", flush=True)
        return False
    matched, missed = _load_lora_into_model(
        model, constants.WAN_DISTILL_LORA_PATH, constants.WAN_DISTILL_LORA_STRENGTH)
    if matched == 0:
        print(f"Wan distill: 0 modules matched (key-format mismatch, {missed} skipped) — "
              f"falling back to stock steps.", flush=True)
        return False
    print(f"Wan distill: LoRA merged into {matched} modules ({missed} skipped), "
          f"steps={constants.WAN_DISTILL_STEPS} guide={constants.WAN_DISTILL_GUIDE} "
          f"strength={constants.WAN_DISTILL_LORA_STRENGTH}", flush=True)
    return True


def _probe_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float((r.stdout or "0").strip())
    except Exception:  # noqa: BLE001
        return 0.0


def _probe_video(path):
    """Return (width, height, duration_seconds) via ffprobe; (None, None, None) on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        parts = [x for x in (r.stdout or "").split() if x]
        return int(parts[0]), int(parts[1]), round(float(parts[2]), 2)
    except Exception:  # noqa: BLE001
        return None, None, None


# Aspect-ratio aliases → width/height. Wan derives the OUTPUT aspect from the REFERENCE IMAGE
# (there's no explicit W×H arg to generate()), so to honor target_ratio we crop the reference to
# that aspect before generating. "reel" == 9:16 vertical for shorts/reels.
_RATIO_ALIASES = {
    "reel": 9 / 16, "portrait": 9 / 16, "vertical": 9 / 16, "9:16": 9 / 16,
    "landscape": 16 / 9, "horizontal": 16 / 9, "16:9": 16 / 9,
    "square": 1.0, "1:1": 1.0,
}


def _aspect_from_ratio(target_ratio):
    """Target width/height as a float, or None if unspecified/unparseable."""
    if not target_ratio:
        return None
    s = str(target_ratio).strip().lower()
    if s in _RATIO_ALIASES:
        return _RATIO_ALIASES[s]
    if ":" in s:
        try:
            w, h = (float(x) for x in s.split(":", 1))
            return w / h if w > 0 and h > 0 else None
        except Exception:  # noqa: BLE001
            return None
    return None


def _detect_face_box(pil_img):
    """(x, y, w, h) of the largest frontal face, or None. Uses OpenCV's bundled Haar cascade — no
    extra dependency or model download. Best-effort: any failure (incl. cv2 absent) returns None so
    the caller falls back to a plain center crop."""
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])  # largest
        return int(x), int(y), int(fw), int(fh)
    except Exception:  # noqa: BLE001
        return None


def _prep_reference_for_ratio(ref_path: str, target_ratio, out_path: str) -> str:
    """Crop-to-fill the reference image to target_ratio so the generated video comes out in that
    aspect, centering the crop on the host's FACE (falls back to center/top-bias if no face is
    found). Returns out_path, or the original ref_path when no reshape is needed/possible."""
    aspect = _aspect_from_ratio(target_ratio)
    if aspect is None:
        return ref_path
    try:
        from PIL import Image

        img = Image.open(ref_path).convert("RGB")
        w, h = img.size
        cur = w / h
        if abs(cur - aspect) < 0.02:
            return ref_path  # already the right shape

        face = _detect_face_box(img)
        print(f"Wan reshape: {w}x{h} (aspect {cur:.2f}) -> target {target_ratio} "
              f"(face={'centered' if face else 'none, center-crop'})", flush=True)
        if cur > aspect:                   # too wide → crop width (keep full height)
            new_w = max(1, int(round(h * aspect)))
            if face:                       # center the strip on the face's horizontal center
                fx, _fy, fw, _fh = face
                left = int(round(fx + fw / 2 - new_w / 2))
            else:
                left = (w - new_w) // 2
            left = max(0, min(left, w - new_w))
            box = (left, 0, left + new_w, h)
        else:                              # too tall → crop height (keep full width)
            new_h = max(1, int(round(w / aspect)))
            if face:                       # put the face ~40% down (headroom above, torso below)
                _fx, fy, _fw, fh = face
                top = int(round(fy + fh / 2 - new_h * 0.40))
            else:
                top = int((h - new_h) * 0.30)
            top = max(0, min(top, h - new_h))
            box = (0, top, w, top + new_h)
        img.crop(box).save(out_path)
        return out_path
    except Exception as e:  # noqa: BLE001 — reshape is an enhancement; never fail generation
        print(f"Wan reshape: failed ({e}); using original reference aspect.", flush=True)
        return ref_path


@app.cls(
    image=wan_image,
    gpu=GPU_FUNCTION_A,
    timeout=3600,
    volumes={
        constants.WEIGHTS_ROOT: weights_volume,
        constants.HOST_MEDIA_ROOT: host_media_volume,
        constants.INTERMEDIATE_ROOT: intermediate_volume,
    },
    enable_memory_snapshot=constants.WAN_ENABLE_SNAPSHOT,   # False: build fresh on GPU each cold
                                                            # start (snapshots break on Wan conv3d)
    scaledown_window=600,   # stay warm 10 min after a call → back-to-back requests skip the
                            # cold start (big win while iterating; small idle cost)
    min_containers=0,       # set to 1 for zero cold start during a demo (idle H100 cost)
)
class WanAvatar:
    @modal.enter()
    def load(self):
        # Build the pipeline on the GPU (fresh each cold start) and merge the 4-step distill LoRA.
        weights_volume.reload()
        self.pipe, self.cfg = _build_pipe()
        self.distill_ok = _maybe_apply_distill(self.pipe) if constants.WAN_USE_DISTILL else False

    @modal.method()
    def run(self, run_id: str, reference_image_path: str, target_ratio: str,
            prompt: str = None) -> dict:
        import sys

        host_media_volume.reload()
        intermediate_volume.reload()

        run_dir = _run_dir(run_id)
        os.makedirs(run_dir, exist_ok=True)

        audio_mp3 = f"{run_dir}/audio.mp3"
        if not os.path.exists(audio_mp3):
            raise WanError(f"Stage A audio missing at {audio_mp3} (orchestration bug)")
        if not reference_image_path or not os.path.exists(reference_image_path):
            raise WanError(f"studio reference image missing at {reference_image_path!r}")

        # Trim to the cost cap + convert to 16k mono wav.
        audio_wav = f"{run_dir}/audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_mp3, "-t", str(constants.WAN_MAX_SECONDS),
             "-ar", "16000", "-ac", "1", audio_wav],
            check=True, capture_output=True,
        )

        if constants.WAN_REPO not in sys.path:
            sys.path.insert(0, constants.WAN_REPO)
        from wan.utils.utils import save_video

        cfg = self.cfg
        max_area = constants.WAN_MAX_AREA  # ~480p budget; output aspect follows the ref image
        prompt = prompt or constants.WAN_DEFAULT_PROMPT
        out_path = f"{run_dir}/final_output.mp4"

        fps = cfg.sample_fps
        infer_frames = constants.WAN_INFER_FRAMES
        audio_dur = _probe_duration(audio_wav)
        num_repeat = max(1, math.ceil(audio_dur * fps / infer_frames)) if audio_dur else 1

        # Distilled path (LoRA merged at load) runs ~4 steps with CFG off; else stock steps.
        distill_on = getattr(self, "distill_ok", False)
        if distill_on:
            steps = constants.WAN_DISTILL_STEPS
            guide = constants.WAN_DISTILL_GUIDE
            shift = constants.WAN_DISTILL_SHIFT or cfg.sample_shift
        else:
            steps = constants.WAN_SAMPLE_STEPS or cfg.sample_steps
            guide = constants.WAN_GUIDE_SCALE or cfg.sample_guide_scale
            shift = cfg.sample_shift

        # Reshape the reference to the requested aspect (Wan follows the ref image's aspect).
        ref_for_gen = _prep_reference_for_ratio(
            reference_image_path, target_ratio, f"{run_dir}/ref_for_gen.png")
        reshaped = ref_for_gen != reference_image_path

        print(f"Wan2.2-S2V: run_id={run_id} area={max_area} target_ratio={target_ratio} "
              f"reshaped_ref={reshaped} audio={audio_dur:.1f}s num_repeat={num_repeat} "
              f"steps={steps} guide={guide} distill={distill_on}", flush=True)

        video = self.pipe.generate(
            input_prompt=prompt,
            ref_image_path=ref_for_gen,
            audio_path=audio_wav,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=num_repeat,
            pose_video=None,
            max_area=max_area,
            infer_frames=infer_frames,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=steps,
            guide_scale=guide,
            n_prompt="",
            seed=constants.WAN_BASE_SEED,
            offload_model=False,
            init_first_frame=False,
        )

        save_video(tensor=video[None], save_file=out_path, fps=fps, nrow=1,
                   normalize=True, value_range=(-1, 1))
        # Mux the FULL-QUALITY original audio (audio.mp3, 44.1kHz) into the final video — NOT the
        # 16kHz mono `audio_wav`, which exists only as Wan's lip-sync CONDITIONING input. Using the
        # 16k track for playback needlessly degrades the output. -t caps the length; -shortest
        # aligns the (chunk-rounded, slightly longer) video to the audio.
        muxed = f"{run_dir}/_final_muxed.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_path, "-i", audio_mp3,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-t", str(constants.WAN_MAX_SECONDS), "-shortest", muxed],
            check=True, capture_output=True,
        )
        os.replace(muxed, out_path)
        intermediate_volume.commit()

        w, h, dur = _probe_video(out_path)
        meta = {
            "run_id": run_id,
            "final_output_path": out_path,
            "resolution": f"{w}x{h}" if w else f"~{max_area}px",
            "duration_seconds": dur,
            "target_ratio": target_ratio,
            "backend": "wan2.2-s2v-14b",
            "steps": steps,
            "num_repeat": num_repeat,
            "distill": distill_on,
        }
        with open(f"{run_dir}/function_a_meta.json", "w") as f:
            json.dump(meta, f)
        print("Wan meta:", meta, flush=True)
        return meta
