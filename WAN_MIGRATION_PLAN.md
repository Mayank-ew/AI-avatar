> **STATUS: COMPLETED.** This migration (EchoMimicV2 -> Wan2.2-S2V-14B) has shipped. Kept for history. Current system: `README.md`, `docs/01-architecture-overview.md`, `docs/02-models.md`, `OPS_RUNBOOK.md`.

# Migration Plan — Function A: EchoMimicV2 → Wan2.2-S2V

**Goal:** replace the animation core (EchoMimicV2) with **Wan2.2-S2V** (Alibaba, Apache-2.0,
audio-driven) to get HeyGen-tier quality, while keeping everything else we built. This is a
*plan* — implement in the phases below, gating on a benchmark before full productionization.

---

## 0. Why this touches more than one file

Wan2.2-S2V is a 14B video-diffusion model. On a single H100 it runs at roughly **2–3 minutes of
compute per second of output** (before optimization), so a 15–30s clip is **many minutes** — far
past any HTTP request timeout. That forces one architectural change beyond swapping the model:
**generation must become an async job** (submit → poll → fetch), not a synchronous request. Plan
accounts for this.

---

## 1. What stays vs. what changes

**Unchanged (all reused as-is):**
- Onboarding (`onboard.py`), voice clone (ElevenLabs), director (`director.py`, Groq).
- **Reference Studio** (`reference_studio.py`) — the nano-banana studio portrait is exactly the
  reference image Wan wants. No change.
- Profile store (`store.py`), constants pattern, frontend shell (`frontend.py`).
- Stage 0 (director) + Stage A (TTS) in `main_generate._orchestrate` — Wan consumes the same
  ElevenLabs `audio.wav`.

**Changes:**
- **Function A** (`generate_stage.py`): EchoMimicV2 → Wan2.2-S2V. New image, new weights, new
  inference call, warm model loading. This is the bulk of the work.
- **`app.py`**: new `wan_image`, weight download for Wan, GPU stays H100 (needs 80 GB).
- **`main_generate.py`**: becomes **async/job-based** (spawn + poll).
- **`constants.py`**: new Wan knobs; EchoMimicV2 constants deleted.
- **`frontend.py`**: generate panel polls a job instead of waiting on one long request.

**Removed COMPLETELY (decision: full removal — comparison outputs already captured):**
- **EchoMimicV2 — everything.** No `ANIMATION_BACKEND` flag, no fallback. Delete:
  - `echomimic_image` (and its `add_local_python_source`) from `app.py`
  - `health_echomimic` from `app.py`
  - EchoMimicV2 weight downloads from `download_weights` (denoising/motion/pose/reference unets,
    `sd-vae-ft-mse`, `sd-image-variations-diffusers`, `whisper/tiny.pt`) + their `verify_weights`
    entries — Wan bundles its own VAE + wav2vec2 audio encoder, so none are needed
  - the EchoMimicV2 repo clone + static ffmpeg-4.4 lines in the image
  - all `ECHOMIMIC_*` constants in `constants.py`
  - EchoMimicV2 pipeline loader, pose-tensor builder, demo-pose logic, `_result_to_frames` in
    `generate_stage.py` (file gets rewritten around Wan)
  - **Weights volume cleanup:** delete the EchoMimicV2 dirs from `avatar-weights-vol` (via
    `modal volume rm` or a one-off cleanup function) to reclaim space.
- **Restorer stack — REMOVED completely (decision):** it only existed to clean up EchoMimicV2's
  soft output; Wan's output needs no restoration. Delete:
  - `restoration_image` from `app.py`; `health_restoration`
  - GFPGAN + RestoreFormer++ weight downloads + `verify_weights` entries; wipe from the volume
  - `restore.py` (Function B: `run_function_b`, `run_restorer_check`, `load_restorer`,
    `assess_restorer_reliability`), `composite.py`, `temporal_blend.py`
  - `choose_preferred_restorer` + `extract_sample_frame` in `onboard.py`; the
    `preferred_restorer_override` form field / CLI arg
  - `preferred_restorer` field in `store.HostProfile` (+ Postgres DDL/SELECT/INSERT)
  - `VALID_RESTORERS`, `ENABLE_RESTORATION`, `ENABLE_TEMPORAL_BLEND`, `RESTORER_*`, `BLEND_*`,
    `COMPOSITE_*`, `FARNEBACK_PARAMS` constants
  - `g_restore` / `g_blend` / restorer-override controls in `frontend.py` (both panels)
  - Wan outputs the finished MP4 with audio muxed at the target size → **no Function B at all**.

---

## 2. Model facts (source of truth for implementation)

- **Weights:** `Wan-AI/Wan2.2-S2V-14B` — single HF repo, **~49 GB** (DiT 14B ~32.6 GB, UMT5-XXL
  text encoder 11.4 GB, VAE 0.5 GB, wav2vec2 audio encoder ~1.2 GB). No separate repos.
- **Code:** clone `github.com/Wan-Video/Wan2.2` (provides `generate.py` + the `wan` package).
- **Deps:** py3.10; `torch>=2.4 torchvision torchaudio` (cu124); `diffusers>=0.31`;
  `transformers>=4.49,<=4.51.3`; `accelerate>=1.1.1`; `flash_attn` (match wheel to torch/CUDA —
  build-from-source is the classic failure); `numpy<2`, `opencv-python`, `imageio[ffmpeg]`,
  `easydict ftfy tqdm`; plus `requirements_s2v.txt` (`openai-whisper librosa decord onnxruntime
  omegaconf conformer hydra-core lightning pyworld modelscope HyperPyYAML inflect wetext rich
  gdown matplotlib wget pyarrow GitPython`). System: **ffmpeg**.
- **Inputs:** `--image <ref.png>` + `--audio <drive.wav>` + `--prompt "<scene desc>"`.
- **Output:** MP4 with **audio muxed**; duration auto-matches audio (`--num_clip N` to cap).
- **Resolution:** `--size WIDTH*HEIGHT`, 480P/720P (e.g. `1024*704`, `704*1280`).
- **VRAM:** single-GPU needs **≥80 GB** with `--offload_model True --convert_model_dtype` →
  **H100 (80 GB) works**. Community FP8 repacks run ~24 GB but are ComfyUI-oriented (skip for now).
- **Reference CLI (headless):**
  ```bash
  python generate.py --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ \
    --offload_model True --convert_model_dtype \
    --prompt "A person speaking to camera in a professional studio" \
    --image ref.png --audio talk.wav --save_file out.mp4
  ```
  For a warm worker we **import the `wan` pipeline classes and load once** (see Phase 3), not
  shell out to `generate.py` per request (that reloads ~44 GB every call).

---

## 3. Aspect-ratio → Wan `--size` mapping (decide exact values in Phase 2)

| target_ratio | Wan `--size` (start at 480p, bump to 720p if speed allows) |
|---|---|
| 9:16 (reels) | `480*832` → `704*1280` |
| 16:9 (landscape) | `832*480` → `1280*704` |
| 1:1 (square) | `624*624` → `960*960` |

Wan derives final aspect from the reference image + `--size`; validate the exact supported sizes
against the repo's config in Phase 2 (only certain buckets are trained).

---

## 4. The async job change (required)

Today: `POST /generate` runs the whole pipeline synchronously and returns the video. With Wan
that request would run for many minutes and time out. New flow:

1. `POST /generate` → does director + TTS (fast), then **`.spawn()`** Function A (Wan) instead of
   `.remote()`, gets a Modal function-call id, stores `{job_id: {status, run_id, ...}}` in a
   `modal.Dict`, and **returns `{job_id}` immediately**.
2. `GET /status/{job_id}` → checks the spawned call (`FunctionCall.from_id(...).get(timeout=0)`
   or a status flag the worker writes to the Dict) → returns `pending | done | error` + the
   video (base64 or a volume path) when done.
3. **Frontend**: after submit, poll `/status/{job_id}` every few seconds, show a progress/"still
   rendering… (~N min)" state, then render the video when ready.

This also fixes the current UX where the tab hangs for the whole render.

---

## 5. Phased implementation

**Phase W1 — Image + weights (no pipeline yet)**
- Add `wan_image` to `app.py` (py3.10, torch 2.4 cu124, matched `flash_attn` wheel, clone Wan2.2
  repo, install both requirements files, ffmpeg). Add `"wan_stage"` (or reuse `generate_stage`)
  to `_ALL_SOURCES`.
- Add `download_wan_weights()` (snapshot_download `Wan-AI/Wan2.2-S2V-14B` → weights volume).
  Extend `verify_weights` with the Wan files.
- **Deploy + run download + a health check** that just imports `wan`, loads the model on H100,
  and reports VRAM/no-OOM. *Gate: image builds, weights present, model loads on one H100.*

**Phase W2 — Benchmark (measure, NOT a hard gate — decision: build all phases)**
- One-off function: ref image (existing `reference_studio.png`) + a ~5–8s sample `audio.wav`,
  `--size 480*832`, `--offload_model True --convert_model_dtype`, default steps.
- **Measure:** wall-clock, VRAM peak, output quality, $ (time × $0.001097/s). Try trimming steps
  and toggling offload (if it fits 80 GB without offload it's faster).
- We proceed to W3–W6 regardless (a bare benchmark isn't representative of final quality); this
  step just gives us the real per-clip speed/cost number to report. If it's wildly bad we'd
  flag LiveAvatar as an alternative, but the default is: keep building.

**Phase W3 — Function A rewrite (warm-loaded)**
- New `run_function_a` (Wan): reload volumes → resolve reference image (`profile.reference_image_path`)
  + `run_dir/audio.wav` + prompt (from scene/hint or default) + `--size` from `target_ratio` →
  run Wan → write `run_dir/final_output.mp4` (audio already muxed) → commit.
- Hold the Wan model in a module-global / `@modal.enter` cache (like we did for EchoMimicV2).
- Rewrite `generate_stage.py` around Wan; delete all EchoMimicV2 code in the same pass.

**Phase W4 — Orchestration + async**
- `main_generate`: director + TTS as today, then `.spawn()` Wan Function A; add `/status/{job_id}`;
  store job state in a `modal.Dict`. Bypass Function B (Wan output is final).

**Phase W5 — Frontend polling**
- Generate panel: submit → poll `/status` → show render progress → play result. Drop the
  now-irrelevant restoration/de-flicker toggles.

**Phase W6 — Clean up**
- Retire Function B from the default path (keep code). Document the EchoMimicV2↔Wan flag. Final
  end-to-end test + quality/cost writeup.

---

## 6. New constants (Phase W3)

```python
# EchoMimicV2 fully removed — no backend flag.
WAN_REPO = "/workspace/Wan2.2"
WAN_CKPT_DIR = f"{WEIGHTS_ROOT}/wan2.2-s2v-14b"
WAN_TASK = "s2v-14B"
WAN_OFFLOAD_MODEL = True
WAN_CONVERT_DTYPE = True
WAN_SAMPLE_STEPS = None            # tune in Phase W2
WAN_SIZE_BY_RATIO = {"9:16": "480*832", "16:9": "832*480", "1:1": "624*624"}  # 480p start
WAN_DEFAULT_PROMPT = "A person speaking directly to the camera in a clean, well-lit professional studio"
```

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `flash_attn` build fails | Use a prebuilt wheel matched to torch2.4/cu124/cp310; or Wan's Docker base |
| 49 GB download / cold load | One-time download to volume; hold model warm across calls |
| Single-H100 OOM | `--offload_model True --convert_model_dtype`; benchmark at 480p first |
| Too slow / too pricey | **Phase W2 gate**; fall back to LiveAvatar (real-time) or paid OmniHuman API |
| Wan `--size` buckets limited | Validate supported sizes from repo config in Phase W2 |
| Prompt drives visuals oddly | Keep a fixed, tested `WAN_DEFAULT_PROMPT`; expose optional override |

---

## 8. Decisions needed before implementing

**All resolved:**
1. ✅ **Async UX** — generate becomes submit→poll (video ready in minutes).
2. ✅ **Start at 480p** for the benchmark; bump to 720p later if speed/cost allow.
3. ✅ **Remove EchoMimicV2 completely** (comparison outputs already captured), incl. volume weights.
4. ✅ **Build all phases W1–W6** — benchmark is measured (for the cost number) but is **not** a
   hard stop; final judgment is on the real end-to-end output.
5. ✅ **Remove the entire restorer stack** (Function B / GFPGAN / RestoreFormer / composite /
   temporal_blend / `preferred_restorer`).
