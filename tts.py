"""
Stage A — ElevenLabs Text-to-Speech.  docs/02 §2, docs/03 §1.

Renders directed_text (tagged, via eleven_v3) or clean_text (untagged fallback, via
eleven_multilingual_v2) into MP3 bytes. Model selection is a decision, not a default: tags
require v3; untagged text uses multilingual_v2 (higher char limit, no tag support needed).

Pure Python (no Modal decorators); called in-process by the orchestrator on director_tts_image.
"""

import os

import constants
import director


class TTSError(Exception):
    stage = "tts"


class CharacterLimitError(TTSError):
    """directed_text/clean_text exceeds the applicable model's per-request character limit."""


def _voice_settings(stability_hint: str) -> dict:
    stability = constants.STABILITY_PRESETS.get(
        stability_hint, constants.STABILITY_PRESETS["natural"]
    )
    return {"stability": stability, **constants.TTS_VOICE_SETTINGS_DEFAULTS}


def select_model(
    directed_text: str,
    enable_director: bool,
    model_override: str | None,
) -> tuple[str, str]:
    """
    Decide (model_id, text_to_send). docs/02 §2 / docs/03 §1:
      - explicit override wins;
      - director disabled OR no tags present -> multilingual_v2 + clean_text;
      - tags present -> eleven_v3 + directed_text.
    Returns (model_id, "tagged"|"clean") — caller supplies the actual text for the mode.
    """
    if model_override:
        # Honor the override; send tagged text if tags exist, else clean.
        mode = "tagged" if director.has_tags(directed_text) else "clean"
        return model_override, mode
    if not enable_director or not director.has_tags(directed_text):
        return constants.TTS_MODEL_FALLBACK, "clean"
    return constants.TTS_MODEL_TAGGED, "tagged"


def _char_limit_for(model_id: str) -> int:
    if model_id == constants.TTS_MODEL_TAGGED:
        return constants.CHAR_LIMIT_V3
    if model_id == constants.TTS_MODEL_FALLBACK:
        return constants.CHAR_LIMIT_MULTILINGUAL_V2
    # Unknown override model: use the conservative v3 limit.
    return constants.CHAR_LIMIT_V3


def synthesize_fish(reference_id: str, text: str) -> tuple[bytes, dict]:
    """
    Fish Audio TTS. POST https://api.fish.audio/v1/tts with JSON + Bearer auth; the model goes in a
    'model' HEADER. Fish uses a `reference_id` (a Fish voice-model id), NOT an ElevenLabs voice_id,
    and ignores ElevenLabs [tags] — so we send the CLEAN text. Returns (mp3_bytes, metadata).
    """
    import requests

    api_key = os.environ.get("FISH_API_KEY")
    if not api_key:
        raise TTSError("FISH_API_KEY not present in environment (add it to .env).")
    reference_id = reference_id or constants.FISH_DEFAULT_REFERENCE_ID
    if not reference_id:
        raise TTSError(
            "Fish Audio needs a voice reference_id — pass fish_reference_id in the request "
            "(or set constants.FISH_DEFAULT_REFERENCE_ID). Get it from your Fish Audio playground."
        )
    if len(text) > constants.CHAR_LIMIT_FISH:
        raise CharacterLimitError(
            f"text length {len(text)} exceeds Fish limit of {constants.CHAR_LIMIT_FISH} chars."
        )

    body = {
        "text": text,
        "reference_id": reference_id,
        "format": constants.FISH_FORMAT,
        "mp3_bitrate": constants.FISH_MP3_BITRATE,
        "sample_rate": constants.FISH_SAMPLE_RATE,
        "chunk_length": constants.FISH_CHUNK_LENGTH,
        "normalize": True,
        "latency": "normal",
        "temperature": constants.FISH_TEMPERATURE,
        "top_p": constants.FISH_TOP_P,
        "prosody": {"speed": constants.FISH_SPEED, "volume": 0, "normalize_loudness": True},
    }
    resp = requests.post(
        f"{constants.FISH_API_BASE}/v1/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "model": constants.FISH_MODEL,
        },
        json=body,
        timeout=300,
    )
    if resp.status_code >= 400:
        detail = None
        try:
            detail = resp.json()
        except Exception:  # noqa: BLE001
            detail = {"raw": resp.text[:500]}
        raise TTSError(f"Fish Audio TTS returned {resp.status_code}: {detail}")

    meta = {
        "model_id": constants.FISH_MODEL,
        "provider": "fishaudio",
        "mode": "clean",
        "char_count": len(text),
        "reference_id": reference_id,
    }
    return resp.content, meta


def synthesize(
    voice_id: str,
    directed_text: str,
    clean_text: str,
    stability_hint: str,
    enable_director: bool = True,
    model_override: str | None = None,
    provider: str | None = None,
    fish_reference_id: str | None = None,
) -> tuple[bytes, dict]:
    """
    Render to (mp3_bytes, metadata). Dispatches on `provider`:
      - "fishaudio" -> Fish Audio (clean text + a Fish reference_id);
      - "elevenlabs" (default) -> ElevenLabs (tagged v3 / clean multilingual_v2, existing path).

    metadata includes {model_id, provider, char_count, ...}. Raises CharacterLimitError BEFORE
    sending if the text exceeds the model limit; TTSError on a non-2xx response.
    """
    provider = (provider or constants.TTS_PROVIDER_DEFAULT).lower()
    if provider in ("fishaudio", "fish"):
        return synthesize_fish(reference_id=fish_reference_id or voice_id, text=clean_text)

    import requests

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise TTSError("ELEVENLABS_API_KEY not present in environment (missing modal.Secret?)")

    model_id, mode = select_model(directed_text, enable_director, model_override)
    text = directed_text if mode == "tagged" else clean_text

    limit = _char_limit_for(model_id)
    if len(text) > limit:
        raise CharacterLimitError(
            f"text length {len(text)} exceeds {model_id} limit of {limit} chars; "
            f"reject or chunk before sending (docs/03 §1)."
        )

    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": _voice_settings(stability_hint),
        "apply_text_normalization": constants.TTS_APPLY_TEXT_NORMALIZATION,
    }

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        params={"output_format": constants.TTS_OUTPUT_FORMAT},
        json=body,
        timeout=300,
    )
    if resp.status_code >= 400:
        # Only the 422 body shape is confirmed (docs/02 §2 errors). Handle by status code,
        # surface whatever body came back for the orchestrator's structured error.
        detail = None
        try:
            detail = resp.json()
        except Exception:  # noqa: BLE001
            detail = {"raw": resp.text[:500]}
        raise TTSError(
            f"ElevenLabs TTS returned {resp.status_code}: {detail}"
        )

    meta = {
        "model_id": model_id,
        "provider": "elevenlabs",
        "mode": mode,
        "char_count": len(text),
        "stability": body["voice_settings"]["stability"],
    }
    return resp.content, meta
