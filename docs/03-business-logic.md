> **LEGACY / historical.** This document predates the Wan2.2-S2V migration and describes the original EchoMimicV2 + face-restoration design. For the current system, see `docs/01-architecture-overview.md`, `docs/02-models.md`, and `OPS_RUNBOOK.md`.

# Business Logic — How the Stages Actually Chain Together

This is the "how it actually works" doc. The original spec described five steps in prose;
this doc spells out the exact data contracts, crops, coordinate remaps, and edge cases between
them — the part that was glossed over and where a flaw-review pass found real gaps (see
[README.md](README.md) corrections #9 and #10).

## 0. Host onboarding contract (one-time, runs before any generation)

```
host uploads video + consent attestation (checkbox, required)
    │
    ▼  reject if consent not attested — do not proceed
ffmpeg: extract audio track from video
    │  (video is NOT assumed accepted directly by ElevenLabs' IVC endpoint —
    │   see 02-models.md §2a; this extraction is a required step, not optional)
    ▼
POST /v1/voices/add (ElevenLabs IVC) with extracted audio
    │
    ▼
{voice_id, requires_verification}
    │
    ▼  sample one representative frame from the uploaded video; run RestoreFormer++'s
    ▼  face-detection/alignment (not full enhance()) against it -> preferred_restorer
    ▼  ("restoreformer++" if reliable, else "gfpgan" — see 02-models.md §4a; a manual
    ▼   override bypassing this check should also exist)
    ▼  store uploaded video as base_video_ref (Volume/object storage)
    ▼  write {host_id, voice_id, base_video_ref, preferred_restorer, created_at}
       to metadata store
    ▼
host_id returned to caller
```

- **Consent attestation is a hard gate, not a soft prompt**: the onboarding request must carry
  an explicit boolean confirming the host attests they have the right to clone the voice being
  uploaded (their own). Reject the request outright if absent — do not proceed to the
  ElevenLabs call without it. This satisfies ElevenLabs' own consent-checkbox requirement, but
  is a *separate, lower bar* than the still-open legal question of whether ElevenLabs' ToS
  actually permits a host commercially redistributing their own cloned voice on personal social
  media — see [07-risks-and-open-questions.md](07-risks-and-open-questions.md). Passing the
  consent gate does not resolve that open question; both are enforced/tracked independently.
- **Audio extraction is mandatory, not optional**: `ffmpeg -i uploaded_video.mp4 -vn -acodec
  libmp3lame onboarding_audio.mp3` (or `.wav`) before calling `/v1/voices/add` — no confirmed
  ElevenLabs endpoint accepts a video container for voice cloning.
- **Single profile, replaceable**: a host has exactly one
  `{voice_id, base_video_ref, preferred_restorer}` at a time. Re-onboarding (the host uploads a
  new video later) overwrites the existing profile
  record — this is a deliberate scope decision, not a versioned history of past videos/clones.
  Whether the *old* ElevenLabs `voice_id` should be explicitly deleted on re-onboarding (to
  free its quota slot — see [07-risks-and-open-questions.md](07-risks-and-open-questions.md))
  or just left orphaned is a follow-on implementation decision; recommend explicit deletion to
  avoid silently accumulating orphaned voice slots against the account's quota.
  > **POC note:** it's fine for the POC to leave the old `voice_id` orphaned on re-onboarding
  > (skip the explicit-delete call) — a handful of demo hosts won't meaningfully dent quota.
  > Production should implement the explicit deletion described above before real hosts
  > re-onboard repeatedly.
- **What downstream generation actually consumes**: a `host_id` resolves to
  `{voice_id, base_video_ref, preferred_restorer}` via a single metadata-store lookup at the
  very start of the generation flow (see §1 below and
  [01-architecture-overview.md](01-architecture-overview.md)'s "Profile lookup" node) —
  `voice_id` feeds Stage A, `base_video_ref` feeds Stage B in place of a raw per-request video
  URL, and `preferred_restorer` feeds Stage D (§5 below) in place of any per-clip or per-frame
  restorer decision.
- **Caching Stage B's derived artifacts is a deferred optimization, not part of this design's
  first cut**: since `base_video_ref` is now fixed per host (not supplied fresh per request),
  the half-body reference crop and pose sequence Stage B derives from it *could* be computed
  once at onboarding time and cached, rather than recomputed on every generation request. This
  pipeline's first implementation recomputes per-request (simpler, no cache-invalidation logic
  needed on re-onboarding) — document caching as a follow-on once real usage/cost data justifies
  the added complexity, not something to build speculatively now.

## 1. Director → TTS contract

```
script_text (+ optional voice_character_hint)
    │
    ▼  Stage 0: GPT-4o, structured output
{ clean_text, directed_text, stability_hint }
    │
    ▼  validation guardrail (see below)
    ▼  character-limit check
    ▼  Stage A: ElevenLabs TTS
/tmp/audio.mp3
```

- **Validation guardrail**: strip all `[...]` tags, revert forced capitalization, and remove
  inserted ellipses/punctuation from `directed_text`; confirm the result equals `clean_text`
  (case/whitespace-normalized). If it doesn't match, the director altered content — reject and
  retry (or fall back to sending `clean_text` untagged) rather than silently shipping altered
  narration.
- **Character-limit check**: compare `len(directed_text)` against the applicable ElevenLabs
  model limit (5,000 for `eleven_v3` standard TTS — see [02-models.md](02-models.md) for the
  conflicting 3,000 figure to reverify) *before* sending. Reject or chunk, don't rely on the
  API to fail gracefully.
- **`stability_hint` → `voice_settings.stability`**: `"creative"` and `"natural"` map to the
  corresponding ElevenLabs preset tier. If the director inserted no tags at all (e.g. bypassed
  via `enable_director=false`), stability tier is irrelevant — use `eleven_multilingual_v2`
  instead of `eleven_v3` in that case (no tags to be responsive to).

## 2. Aspect-ratio math (Stage B, canvas formatting)

```python
def fit_to_aspect_ratio(frame, target_w_ratio, target_h_ratio):
    h, w = frame.shape[:2]
    target_aspect = target_w_ratio / target_h_ratio
    current_aspect = w / h

    if current_aspect > target_aspect:
        # source is wider than target -> crop width
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        # source is taller than target -> crop height
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h]

    h, w = frame.shape[:2]
    # CRITICAL: H264 requires even width/height, or ffmpeg encoding crashes
    w -= w % 2
    h -= h % 2
    return frame[:h, :w]
```

Worked examples (assuming a 1920×1080 source):
- `target_ratio="16:9"` → already 16:9, crop is a no-op beyond the even-dimension trim.
- `target_ratio="9:16"` → center-crop width to `1080 * 9/16 = 607.5` → `608` (rounded up would
  break evenness; always floor then re-apply the `w -= w % 2` trim) → output `608×1080`.
- `target_ratio="1:1"` → center-crop width to `1080` → output `1080×1080`.

This produces `Master_Background` — the aspect-ratio-locked frame sequence everything else
composites against.

## 3. Two distinct crops from Stage B — do not conflate them

The original spec's wording ("run InsightFace... crop the face and resize to 512×512 for
inference") reads as one crop feeding one model. In practice **two separate crops** are
needed, serving two different downstream consumers, plus a coordinate remap for pose:

### 3a. Reference-crop derivation for Stage C (EchoMimicV2)

InsightFace's `FaceAnalysis.get()` returns a **tight** face bounding box. EchoMimicV2's
reference image is a **half-body/chest-up** framing (confirmed from its own demo assets —
visible shoulders, torso, hands). Feeding the tight bbox straight in as-is produces a
face-filling frame with no body/hands for the DWPose conditioning to align against —
out-of-distribution relative to training, likely breaking animation quality, not just a
resolution nit.

```python
def derive_half_body_reference_crop(frame, face_bbox, expansion_ratio=3.5, vertical_bias=0.65):
    """
    face_bbox: (x0, y0, x1, y1) tight box from InsightFace.
    expansion_ratio: how many face-heights tall the output crop should be.
    vertical_bias: fraction of the expanded height that extends BELOW the chin
                   vs above the forehead (chest-up framing needs more room below).
    NOTE: expansion_ratio/vertical_bias are starting guesses, not validated constants —
    tune empirically against real base-video framings (see 06-development-testing-protocol.md).
    """
    fx0, fy0, fx1, fy1 = face_bbox
    face_h = fy1 - fy0
    face_cx = (fx0 + fx1) / 2

    crop_h = face_h * expansion_ratio
    top = fy0 - crop_h * (1 - vertical_bias)
    bottom = fy1 + crop_h * vertical_bias
    crop_h = bottom - top
    crop_w = crop_h  # square, before any letterboxing

    left = face_cx - crop_w / 2
    right = face_cx + crop_w / 2

    left, top, right, bottom = clamp_and_pad_to_square(frame, left, top, right, bottom)
    crop = frame[int(top):int(bottom), int(left):int(right)]

    # IMPORTANT: EchoMimicV2 resizes the reference image with a plain, non-aspect-preserving
    # PIL resize. Since the crop above is already square, resize is safe here — do not skip
    # the square-padding step above and rely on a stretch-resize to "fix" a non-square crop.
    return cv2.resize(crop, (768, 768), interpolation=cv2.INTER_LINEAR), (left, top, right, bottom)
```

**Face-selection rule** (what InsightFace returns is a list, with no built-in "pick one"
default):
- **Zero faces detected** → fail the request with a clear user-facing error ("no face detected
  in base video"), not a deep stack-trace failure inside a GPU Modal Function.
- **Multiple faces detected** → select the largest bbox by area (proxy for "the presenter,
  closest to camera"); document this as the default rule, with "most-central" as a documented
  alternative if the largest-bbox heuristic misfires in practice.

### 3b. Master_Background paste-back bbox (for Stage E)

Separately, record the bounding region in `Master_Background`'s own (arbitrary, possibly very
different) resolution where the final half-body frame will be pasted back at the very end.
This is unrelated to 3a's reference crop — different coordinate space, used only in Stage E.

### 3c. Pose coordinate remap

DWPose keypoints are extracted from a driving video (see §4 below for which video that is) in
that video's native coordinate space. Before EchoMimicV2 can use them, they must be remapped
into the **reference-crop's** 768×768 coordinate space using the same crop offset/scale factor
computed in 3a:

```python
def remap_pose_keypoints(keypoints_xy, crop_box, crop_size=768):
    left, top, right, bottom = crop_box
    scale_x = crop_size / (right - left)
    scale_y = crop_size / (bottom - top)
    return [((x - left) * scale_x, (y - top) * scale_y) for (x, y) in keypoints_xy]
```

Without this remap, pose-driven motion will be spatially misaligned with the reference image —
DWPose was extracted in the original video's coordinate space, not the cropped/resized one
EchoMimicV2 actually consumes.

## 4. Driving-video source — resolved by the onboarding design

Earlier drafts of this doc left open whether the DWPose pose sequence should be **(a)
self-driven** (extracted from the same video that supplies `Master_Background`) or **(b)
externally-driven** (a separate stock/gesture clip). The host-onboarding design in §0 resolves
this: since a host has exactly **one** stored `base_video_ref`, reused for every generation,
there is no separate per-request video to source pose from anyway — **pose is always
self-driven from `base_video_ref`**, the same video the reference crop (§3a) comes from. This
means the coordinate-remap in §3c is always computed against `base_video_ref`'s own coordinate
space, with no branch for an externally-driven case. Motion variety is bounded by whatever the
host does in their one onboarding clip — this is an accepted product tradeoff of the "single
profile, replaceable" scope decision (§0), not an oversight; supporting multiple/varied
driving clips per host is a documented future extension, not part of this design.

## 5. Face restoration — no hand-rolled bridge crop, no runtime restorer decision

Stage D calls RestoreFormer++/GFPGAN's own `enhance(..., has_aligned=False, paste_back=True)`
directly on the full 768×768 half-body frame from Stage C (see
[02-models.md](02-models.md) §4a for why). There is no manual "crop face → resize to 512" step
here — the library's internal `FaceRestoreHelper` performs detection, FFHQ alignment,
restoration, and inverse-affine paste-back into the half-body frame as one call. This is one
paste-back operation (face→half-body, library-internal) — keep it conceptually distinct from
the outer half-body→Master_Background paste-back in §7, so the two aren't confused when
diagramming or debugging.

*Which* restorer Stage D calls is not decided here at all — it's `HostProfile.preferred_restorer`,
read once at the profile-lookup step that fronts the whole generation flow (§1's diagram node,
[01-architecture-overview.md](01-architecture-overview.md)'s "Profile lookup" node), decided
once per host back at onboarding time (§0 above). Stage D itself contains no restorer-selection
logic, no fallback branching, and no per-frame or per-clip decision-making — it just reads the
one value the profile lookup handed it and calls that restorer, every frame, every generation,
for that host.

## 6. Temporal blending design (replaces BFVR-STC's chunking)

BFVR-STC's inference script had a hard-coded 24-frame cap requiring chunking and
re-concatenation, with visible-seam risk at chunk boundaries. RestoreFormer++/GFPGAN have no
such cap — restoration is per-frame and independently batchable. The temporal-blend pass that
replaces BFVR-STC's (non-commercial) approach:

1. Restore every frame independently (§5), no chunking needed.
2. Compute optical flow between adjacent restored frames.
3. Warp+blend over a 3-5 frame sliding window, weighting eye/mouth regions more heavily using
   the landmark masks §5's alignment step already produced (no fresh detection).
4. Output de-flickered half-body frames, same count/order as input.

**This is unproven, in-house code** — unlike BFVR-STC's chunk-seam risk (a known, bounded
quantity from a real upstream implementation), there's no reference implementation to validate
expected behavior against. Treat window size and landmark-weighting as tunable parameters to
validate visually (see [06-development-testing-protocol.md](06-development-testing-protocol.md)),
not settled constants.

## 7. Outer composite (Stage E) — feathering alone is not enough

```python
def composite_into_master_background(master_frame, half_body_frame, paste_bbox):
    x0, y0, x1, y1 = paste_bbox
    resized = cv2.resize(half_body_frame, (x1 - x0, y1 - y0))
    mask = build_feathered_mask(resized.shape, feather_px=15)
    # Feathering smooths the SPATIAL seam at the paste edge, but EchoMimicV2's output is
    # diffusion-generated — its background pixels will not pixel-match Master_Background's
    # real photographic environment even when sourced from the same footage, because
    # diffusion regenerates background texture/lighting/color per frame. Feathering does not
    # fix this; color/luminance matching (e.g. histogram matching or Poisson blending) is a
    # separate step required for a seamless result, and should be visually validated, not
    # assumed to be solved by feathering.
    matched = match_color_histogram(resized, reference=master_frame[y0:y1, x0:x1])
    master_frame[y0:y1, x0:x1] = alpha_blend(master_frame[y0:y1, x0:x1], matched, mask)
    return master_frame
```

## 8. Audio handling and final mux

The restoration/temporal-blend stage (§5-6) is video-only — frames in, frames out, no audio
concept. `/tmp/audio.mp3` (rendered by Stage A, from tag-directed `directed_text`) is carried
out-of-band through the entire video pipeline and muxed back in only at the very end:

```bash
ffmpeg -y -i /tmp/silent_output.mp4 -i /tmp/audio.mp3 \
  -c:v libx264 -c:a aac -pix_fmt yuv420p -aspect {target_ratio} \
  -shortest /tmp/final_output.mp4
```

`-pix_fmt yuv420p` and the explicit `-aspect` flag lock metadata so the file plays correctly
across web browsers; `-shortest` guards against audio/video length mismatches from any
frame-count truncation upstream.

## 9. GPU memory lifecycle

Sequential, not concurrent, VRAM occupancy:
- **Function A** loads EchoMimicV2's four checkpoints (VAE, reference UNet, denoising UNet,
  pose encoder) plus the Whisper audio encoder, runs inference, writes output frames to the
  shared Volume, and the container recycles (freeing GPU memory) — or explicitly
  `del model; torch.cuda.empty_cache()` between any sub-models if reusing a warm container
  across requests.
- **Function B** (separate container, separate GPU allocation) loads only
  RestoreFormer++/GFPGAN (lightweight GAN, no VAE/UNet diffusion stack) — there is no
  requirement for both stages' weights to be resident simultaneously, which is exactly the
  free offload boundary the Function A→B split provides.
- **Stage 0/A** (LLM + TTS) run on CPU-only infra with no GPU lifecycle concerns at all.
