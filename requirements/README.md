# `requirements/` — reference only

Modal builds every container image **inline in [`app.py`](../app.py)** (`onboarding_image`,
`director_tts_image`, `wan_image`, `studio_image`, `weights_image`, `frontend_image`) via
`.pip_install(...)`. Nothing in the codebase reads these `.txt` files — they are kept as a
readable record of each image's dependency set.

| File | Status | Corresponds to |
|---|---|---|
| `onboarding.txt` | current | `onboarding_image` (CPU, ffmpeg, yt-dlp) |
| `director_tts.txt` | current | `director_tts_image` (CPU — director + TTS + orchestration) |
| `echomimic.txt` | `LEGACY` | old Function A (EchoMimicV2) — replaced by `wan_image` |
| `restoration.txt` | `LEGACY` | old Function B (face restore / composite) — removed entirely |

If you change a `pip_install` list in `app.py`, update the matching file here too.
