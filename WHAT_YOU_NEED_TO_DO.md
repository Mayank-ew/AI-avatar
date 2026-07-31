# What You Need To Do — Action Guide (current / Wan2.2-S2V)

The pipeline is **built and demo-ready**. This is the short list of what needs *your* accounts,
keys, and judgment. Operational detail (every command) lives in
[`OPS_RUNBOOK.md`](OPS_RUNBOOK.md); architecture in [`docs/01`](docs/01-architecture-overview.md) /
[`docs/02`](docs/02-models.md).

> **Windows:** run `py -m modal ...` (not bare `modal ...`) to avoid the PATH issue.

## 1 · Accounts & keys (put them in `.env`)

| Key | For | Needed? |
|---|---|---|
| `MODAL_*` (auth via `py -m modal setup`) | serverless GPU/host | required |
| `ELEVENLABS_API_KEY` | TTS provider A + voice cloning (IVC) | required for ElevenLabs path |
| `FISH_API_KEY` | TTS provider B (`s2.1-pro`) | required for Fish path |
| `GROQ_API_KEY` | audio director (`gpt-oss-120b`) | required (ElevenLabs path) |
| `GEMINI_API_KEY` | nano-banana studio reference | optional — **needs PAID billing** (currently `limit:0`); references are made by hand otherwise |
| `APP_BEARER_TOKEN`, `MODAL_PROXY_KEY`, `MODAL_PROXY_SECRET` | endpoint auth | required |
| `ONBOARD_URL`, `GENERATE_URL` | panel → pipeline | required (from `modal deploy` output) |

Modal's free **Starter** plan includes GPU access (no separate request). Cloning a voice needs an
ElevenLabs tier that supports Instant Voice Clone; otherwise assign a **pre-made** voice with
`--voice-id` (or use Fish) and skip cloning.

## 2 · Deploy & weights (once) — see runbook §1

```bash
py -m modal deploy app.py
py -m modal run app.py::download_weights          # ~49GB Wan2.2-S2V-14B
py -m modal run app.py::download_distill_lora      # ~631MB 4-step distill LoRA
py -m modal run app.py::health_all
```

## 3 · Onboard + generate — see runbook §2–§3

Open the **control panel** URL from the deploy output. Onboard a host (upload/URL, or the manual
webinar path in the runbook), then generate from `host_id` + script + aspect + TTS provider. For a
**live demo**, follow the warm-up procedure in runbook §4 so you skip the ~2.5-min cold start.

## 4 · Judgment calls (only you can do these)

- **TTS A/B** — run the same script on ElevenLabs vs Fish; pick the voice that sounds most human.
- **Distill quality** — if lip-sync looks soft, raise `WAN_DISTILL_LORA_STRENGTH` (1.0→1.5) or set
  `WAN_USE_DISTILL=False` to compare against the stock model.
- **Reel framing** — if a 9:16 crop looks zoomed, register a waist-up per-aspect reference
  (runbook §2).
- **Cost vs quality** — 480p is the default; 720p (`WAN_MAX_AREA=921600`) is ~2.3× slower.

## 5 · Legal & compliance (before anything ships publicly)

1. **Likeness + voice consent.** You're recreating a real person's face and voice. Keep the
   onboarding consent gate, and get a retention / deletion-on-request policy reviewed.
2. **TTS ToS.** Confirm ElevenLabs / Fish Audio terms permit your commercial redistribution before
   public content.
3. **nano-banana / Gemini** image generation of real people has its own usage policy — the API is
   also billing-gated (`limit:0`) right now.

## 6 · Roadmap (not built yet)

- **Action/motion replication** — make an avatar copy a real subject's actual gestures from a
  driving video (Wan2.2-Animate, or Wan-S2V `pose_video`). Currently motion is *synthesized* from
  audio, not copied.
- **Automated per-aspect references** once Gemini billing is enabled (auto-reframe to 9:16/1:1).
