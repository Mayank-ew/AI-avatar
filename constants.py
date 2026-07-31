"""
Central, tunable configuration for the EasyWebinar AI Avatar pipeline.

Everything here is a knob the docs flag as needing empirical validation or as a value that
should live in one place rather than scattered inline (docs/03 §3a, docs/06, docs/07). Keeping
them module-level means Phase 9's crop tuning and Phase 6's blend tuning happen by editing this
file, not by hunting through the stage code.
"""

# ---------------------------------------------------------------------------
# Stage 0 — LLM Audio Director (docs/02 §1)
# ---------------------------------------------------------------------------

# Which LLM backend Stage 0 calls. "openai" = GPT-4o (production default, per docs/02 §1).
# "groq" = a Groq-hosted OpenAI-compatible endpoint — set here for cheap/fast local testing.
# Groq's openai/gpt-oss-120b confirmed to support OpenAI Structured Outputs (strict json_schema
# mode) as of this writing, so director.py's guardrail/schema code works unchanged either way —
# only the client construction (api_key/base_url) and model id differ. Swap back to "openai"
# before any real production run (GPT-4o is the doc-specified, quality-validated choice).
DIRECTOR_PROVIDER = "groq"

# The model that supports OpenAI Structured Outputs (json_schema). gpt-4o-2024-08-06+ required.
DIRECTOR_MODEL = "gpt-4o-2024-08-06"
DIRECTOR_MODEL_GROQ = "openai/gpt-oss-120b"   # ~120B, Groq-hosted, supports strict json_schema
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Closed tag vocabulary. The director MUST NOT invent tags outside this set (docs/02 §1).
# Grouped for the system prompt; flattened set used by the guardrail to strip tags.
DIRECTOR_TAGS = {
    "emotion": [
        "excited", "nervous", "frustrated", "sorrowful", "calm", "sad", "angry",
        "happily", "curious", "sarcastic", "mischievously", "awe",
    ],
    "delivery": ["whispers", "shouts", "rushed", "drawn out", "dramatic tone"],
    "reactions": [
        "laughs", "laughs softly", "sighs", "exhales", "gasps", "clears throat",
    ],
    "pacing": ["pause", "hesitates"],
}

# Max retries on the content-preservation guardrail before hard-rejecting (docs/03 §1).
DIRECTOR_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Stage A — ElevenLabs TTS (docs/02 §2)
# ---------------------------------------------------------------------------

TTS_MODEL_TAGGED = "eleven_v3"                 # tags are effectively v3-exclusive
TTS_MODEL_FALLBACK = "eleven_multilingual_v2"  # non-tagged path (enable_director=false)

# Character limits per model. The v3 limit is CONFLICTING across sources (5000 vs 3000 post-GA,
# docs/02 §2 / docs/07). Start at the conservative 3000 and confirm empirically in Phase 1.
# Flip CHAR_LIMIT_V3 back to 5000 once the live API is confirmed to accept it.
CHAR_LIMIT_V3 = 3000
CHAR_LIMIT_MULTILINGUAL_V2 = 10000

# stability_hint -> voice_settings.stability numeric band. These numbers are HYPOTHESES to
# validate empirically (docs/02 §2 table, docs/07) — ElevenLabs documents named UI presets,
# not raw numbers.
STABILITY_PRESETS = {
    "creative": 0.30,
    "natural": 0.50,
}

TTS_VOICE_SETTINGS_DEFAULTS = {
    "similarity_boost": 0.75,
    "style": 0,
    "use_speaker_boost": True,
}

TTS_OUTPUT_FORMAT = "mp3_44100_128"
TTS_APPLY_TEXT_NORMALIZATION = "auto"

# Per-tier ElevenLabs concurrency cap (docs/02 §2a). Used to size the TTS semaphore if/when
# per-sentence chunking is added. Set to the dev account's tier.
ELEVENLABS_CONCURRENCY_CAP = 3  # Starter tier default; raise to match the real account tier.

# ---------------------------------------------------------------------------
# Fish Audio TTS — alternative provider, for A/B testing against ElevenLabs.
# ---------------------------------------------------------------------------
# POST https://api.fish.audio/v1/tts, JSON body, Bearer auth, model in a 'model' HEADER. Fish does
# NOT use ElevenLabs [tags]; it takes the CLEAN script + prosody/temperature for expressiveness, so
# the director stage is skipped for Fish. The voice is a `reference_id` (a Fish voice-model id from
# their playground) — DISTINCT from an ElevenLabs voice_id — supplied per-request (fish_reference_id)
# or via FISH_DEFAULT_REFERENCE_ID. Needs FISH_API_KEY in the environment (add it to .env).
TTS_PROVIDER_DEFAULT = "elevenlabs"     # "elevenlabs" | "fishaudio"; per-request override in payload
FISH_API_BASE = "https://api.fish.audio"
FISH_MODEL = "s2.1-pro-free"            # 'model' header: s1 | s2-pro | s2.1-pro | s2.1-pro-free
FISH_FORMAT = "mp3"
FISH_MP3_BITRATE = 128                  # 64 | 128 | 192
FISH_SAMPLE_RATE = 44100
FISH_CHUNK_LENGTH = 300                 # 100–300; Fish chunks long text internally
FISH_TEMPERATURE = 0.7                  # 0–1; higher = more expressive
FISH_TOP_P = 0.7
FISH_SPEED = 1.0                        # prosody.speed, 0.5–2.0
CHAR_LIMIT_FISH = 10000                 # generous guard; Fish handles long text via chunking
FISH_DEFAULT_REFERENCE_ID = "8081b96ed3184b9f96453a09e695fb9f"  # default Fish voice; overridable

# ---------------------------------------------------------------------------
# Reference Studio — reimagine a clean animation reference from the host's video.
# ---------------------------------------------------------------------------
# EchoMimicV2 (and any portrait animator) is only as good as its reference image: it wants a
# bright, front-facing, centered, half-body shot on a clean background. Real onboarding uploads
# are rarely like that (dark, side-lit, off-center, busy backgrounds). So at onboarding we take
# the best frame of the host's video and reimagine it into an idealized studio portrait of THE
# SAME PERSON via Google's Gemini 2.5 Flash Image ("nano-banana") — an identity-preserving image
# editor. That portrait becomes the stored animation reference. Runs ONCE per host (not per
# video), so it adds no per-generation latency/cost. Needs GEMINI_API_KEY in the environment.
ENABLE_STUDIO_REFERENCE = True

# nano-banana model id (Gemini 2.5 Flash Image). Swap to the "-preview" id if GA is unavailable.
NANO_BANANA_MODEL = "gemini-2.5-flash-image"

# How many frames to sample across the onboarding video when picking the best reference frame.
# Talking clips are mostly mid-word / blinking, so sample generously to find a clean frame.
STUDIO_BEST_FRAME_SAMPLES = 40

# Best-frame QUALITY GATES (a frame must pass these to be preferred; if none pass, the gates are
# relaxed progressively so we always return something). Tuned against real talking-head clips.
STUDIO_EYE_OPEN_MIN = 0.17     # eye-aspect-ratio floor; below this the eyes are shut/downcast
STUDIO_FRONTAL_MAX = 0.45      # max horizontal nose-offset (÷ inter-eye dist); higher = turned away
STUDIO_MIN_FACE_LUMA = 55      # mean face-region brightness floor (0-255); below = too dark
STUDIO_SHARP_CAP = 300.0       # cap sharpness's contribution so a huge close-up can't dominate

# Default scene if the host doesn't specify one. Overridable per onboarding / per re-roll.
DEFAULT_STUDIO_SCENE = (
    "seated at a modern podcast desk with a professional microphone in front of them, "
    "in a clean, softly-lit professional studio"
)

# Where the reimagined studio reference (and the source frame it came from) are stored, relative
# to the host's media folder.
STUDIO_REFERENCE_NAME = "reference_studio.png"
STUDIO_SOURCE_FRAME_NAME = "reference_source.png"

# Canonical aspect-ratio forms + aliases (used for per-aspect reference lookup/storage).
_RATIO_CANON = {
    "reel": "9:16", "portrait": "9:16", "vertical": "9:16", "9:16": "9:16",
    "landscape": "16:9", "horizontal": "16:9", "16:9": "16:9",
    "square": "1:1", "1:1": "1:1",
}


def canonical_ratio(ratio) -> str | None:
    """Map 'reel'/'portrait'/'9:16'/... to a canonical 'W:H' key; passthrough unknown W:H."""
    if not ratio:
        return None
    return _RATIO_CANON.get(str(ratio).strip().lower(), str(ratio).strip())


def ratio_slug(ratio) -> str:
    """Filename-safe form of a ratio, e.g. '9:16' -> '9x16'."""
    return (canonical_ratio(ratio) or "").replace(":", "x").replace("/", "x")


def studio_reference_path(host_id: str, ratio=None) -> str:
    """Default studio reference, or the per-aspect one (reference_studio_9x16.png) when ratio set."""
    if ratio:
        return f"{HOST_MEDIA_ROOT}/{host_id}/reference_studio_{ratio_slug(ratio)}.png"
    return f"{HOST_MEDIA_ROOT}/{host_id}/{STUDIO_REFERENCE_NAME}"

def studio_source_frame_path(host_id: str) -> str:
    return f"{HOST_MEDIA_ROOT}/{host_id}/{STUDIO_SOURCE_FRAME_NAME}"

# ---------------------------------------------------------------------------
# Reference face detection (used by reference_studio.py best-frame picker)
# ---------------------------------------------------------------------------

FACE_MIN_DET_SCORE = 0.5   # below this, treat as no reliable face detected

# ---------------------------------------------------------------------------
# Stage C — Wan2.2-S2V-14B animation (audio-driven). Replaces EchoMimicV2.
# ---------------------------------------------------------------------------
# Wan2.2-S2V (Alibaba, Apache-2.0) takes the reimagined studio portrait + the ElevenLabs voice
# track and produces a finished MP4 with the audio already muxed in, at the target resolution.
# It is a 14B video-diffusion model: high quality, but minutes per clip on a single H100 — which
# is why /generate is an async job (see main_generate.py). Weights (~49GB) live on the weights
# volume; inference runs from the cloned Wan2.2 repo's generate.py (see generate_stage.py).

WAN_REPO = "/workspace/Wan2.2"                       # git clone of github.com/Wan-Video/Wan2.2
WAN_HF_REPO = "Wan-AI/Wan2.2-S2V-14B"                # HF weights repo (~49GB)
WAN_CKPT_DIR = f"/vol/wan2.2-s2v-14b"               # where download_wan_weights() puts them
WAN_TASK = "s2v-14B"

# In-process warm-load flags (Function A builds WanS2V ONCE and reuses it). `offload_model=False`
# keeps the DiT resident on the H100 between calls — that's both the warm-load win AND a per-call
# speedup (offload shuttles the model CPU<->GPU every step). bf16 DiT via convert_model_dtype.
# If the logs show CUDA OOM at 80GB, set WAN_T5_CPU=True (moves the 11GB T5 encoder to CPU).
WAN_OFFLOAD_MODEL = False
WAN_CONVERT_DTYPE = True
WAN_T5_CPU = False

# Cold-start strategy. We build the pipeline fresh on the GPU each cold start (no snapshot).
# WHY NOT SNAPSHOTS: (a) the alpha GPU snapshot doesn't speed up storage loading and was the ~5-min
# path; (b) the CPU memory-snapshot + two-phase pattern (build on CPU in snap=True, move to GPU in
# snap=False) FAILS on this model — Wan's 3D-conv layers (VAE + DiT) crash when built on CPU and
# moved to CUDA ("aten::slow_conv3d_forward has no CUDA kernel"), and the only fix (rebuild them
# natively on GPU) defeats the point. So snapshots aren't viable here. Fresh-load is guaranteed
# correct and also removes the ~12-min snapshot-CREATION every redeploy used to pay. For zero cold
# start when needed (demos), set min_containers=1 (idle H100 cost). Left as a flag for clarity.
WAN_ENABLE_SNAPSHOT = False

# Sampling knobs. COST DRIVER: total diffusion ≈ num_repeat × steps × pixels, and num_repeat is
# ceil(audio_secs·fps / infer_frames) — so a 16s clip at 720p/20-steps = 80 passes ≈ 50 min.
# Keep these LOW. 10 steps (down from Wan's 40) + 480p keeps a single ~5s chunk to a few minutes.
WAN_SAMPLE_STEPS = 10
WAN_GUIDE_SCALE = None
WAN_BASE_SEED = 42
WAN_INFER_FRAMES = 80          # frames per generated clip chunk (num_repeat is derived from audio)

# Pixel budget passed to WanS2V.generate(max_area=...). Output aspect follows the reference image
# within this area. 480p (854x480 = 409,600) — ~16s/step on H100 → ~3 min for a single ~5s chunk
# at 10 steps. 2.3x fewer pixels than 720p (921,600), so ~2.3x faster PER STEP. Raise to 921600
# for max-quality 720p (but that's ~38s/step → ~6-7 min/chunk single-GPU; use 2x H100 seq-parallel
# to keep 720p under ~4 min).
WAN_MAX_AREA = 409600

# Wan takes a text prompt describing the VISUAL scene/motion (the words/voice come from the audio).
WAN_DEFAULT_PROMPT = (
    "A person speaking directly to the camera in a clean, well-lit professional studio, "
    "natural head movement and facial expressions, high quality, sharp focus"
)

# Cap on generated duration (seconds) — HARD cost guard. Each ~5s of audio = one more full
# diffusion pass (num_repeat). With 4-step distill a 30s clip is ~6 chunks ≈ ~4 min, so 30 is a
# reasonable test ceiling. Raise further only once per-second cost is acceptable.
WAN_MAX_SECONDS = 30

# ---------------------------------------------------------------------------
# 4-step distillation (SPEED) — a LoRA "turbo" on the SAME 2.2-S2V base model.
# ---------------------------------------------------------------------------
# The base model stays Wan2.2-S2V-14B (unchanged quality). This LoRA only lets it reach a good
# result in ~4 steps instead of 10-40 → ~2.5x fewer diffusion passes per chunk. There is NO
# official S2V distill LoRA; the 2.2 distill LoRAs are for the A14B *MoE* and don't fit S2V's
# *dense* backbone. S2V-14B rides the Wan2.1 dense-14B skeleton, so the Wan2.1-T2V-14B cfg-step
# -distill LoRA is the architecturally-compatible accelerator (community-verified on S2V).
# CROSS-TASK CAVEAT: it was trained on text-to-video, not audio-driven — so it may need strength
# tuning and can slightly soften lip-sync. It's a toggle: flip WAN_USE_DISTILL=False to instantly
# revert to the stock model (WAN_SAMPLE_STEPS / WAN_GUIDE_SCALE) with zero other changes.
WAN_USE_DISTILL = True

# HF source for the LoRA (~631MB). Fetched once by app.py::download_distill_lora into the weights
# volume. hf_hub_download preserves the "loras/" subfolder, so the on-disk path includes it.
WAN_DISTILL_LORA_REPO = "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v"
WAN_DISTILL_LORA_FILE = "loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
WAN_DISTILL_LORA_DIR = "/vol/distill-lora"                       # under WEIGHTS_ROOT (=/vol)
WAN_DISTILL_LORA_PATH = f"{WAN_DISTILL_LORA_DIR}/{WAN_DISTILL_LORA_FILE}"

# Distill sampling knobs (used ONLY when the LoRA actually loaded; else we fall back to stock).
# cfg-step-distill is trained for 4 steps with NO classifier-free guidance (guide=1.0).
WAN_DISTILL_STEPS = 4
WAN_DISTILL_GUIDE = 1.0
WAN_DISTILL_LORA_STRENGTH = 1.0    # official README merges at 1.0; raise toward ~1.5 if under-baked
WAN_DISTILL_SHIFT = None           # None = use the config's sample_shift; override to tune distill

# ---------------------------------------------------------------------------
# Storage / paths (docs/05 §0)
# ---------------------------------------------------------------------------

# Which ProfileStore backend the deployment uses. "modal_dict" is the Modal-native path (no
# external DB to provision — works immediately with just a Modal account) and is what this
# deployment uses; "postgres" is the alternate POC path (docs/05 §0 POC note) for when you want
# to eyeball host profiles in a SQL client during a demo — switch to it anytime by setting this
# back to "postgres" and filling in POSTGRES_CONN_STRING in .env.
PROFILE_STORE_BACKEND = "modal_dict"

HOST_MEDIA_ROOT = "/host_media"                 # host-media-vol mount point
WEIGHTS_ROOT = "/vol"                           # avatar-weights-vol mount point
INTERMEDIATE_ROOT = "/intermediate"             # shared A->B handoff volume mount point

def onboarding_video_path(host_id: str) -> str:
    return f"{HOST_MEDIA_ROOT}/{host_id}/onboarding_video.mp4"

def onboarding_audio_path(host_id: str) -> str:
    return f"{HOST_MEDIA_ROOT}/{host_id}/onboarding_audio.mp3"
