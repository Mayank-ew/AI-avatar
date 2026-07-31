"""
Phase 12 — End-to-end test harness.  docs/06 Phase 5, docs/08 Phase 12.

Exercises onboarding once, then generation (which never needs voice_id/video again), then a
second generation for the SAME host to prove the profile is reused. Run locally against a
deployed app, passing the two web endpoint URLs EXACTLY as `modal deploy` printed them (each
Modal web function gets its own full URL — don't try to reconstruct one from the other):

    python test.py \
        --onboard-url https://<workspace>--<app>-host-onboard.modal.run \
        --generate-url https://<workspace>--<app>-generate.modal.run \
        --app-token <app-level-bearer> \
        --modal-key <proxy-key> --modal-secret <proxy-secret> \
        --video ./test_presenter_clip.mp4

Edge cases it deliberately hits: missing consent (must fail at consent_check), unknown host_id
(must fail at profile_lookup). Decode the returned base64 video to disk for manual QA.
"""

import argparse
import base64
import json
import sys

import requests


def _headers(args, json_body=False):
    h = {}
    if args.modal_key and args.modal_secret:
        h["Modal-Key"] = args.modal_key
        h["Modal-Secret"] = args.modal_secret
    if args.app_token:
        h["Authorization"] = f"Bearer {args.app_token}"
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def onboard(args, consent="true"):
    url = f"{args.onboard_url.rstrip('/')}/host/onboard"
    with open(args.video, "rb") as f:
        resp = requests.post(
            url, headers=_headers(args),
            data={"host_id": args.host_id, "consent_attested": consent},
            files={"video": f},
        )
    print(f"[onboard consent={consent}] {resp.status_code}: {resp.text[:400]}")
    return resp


def generate(args, host_id, script, ratio="9:16", extra=None):
    url = f"{args.generate_url.rstrip('/')}/generate"
    payload = {"host_id": host_id, "script_text": script, "target_ratio": ratio}
    if extra:
        payload.update(extra)
    resp = requests.post(url, headers=_headers(args, json_body=True), json=payload)
    ct = resp.text[:400] if resp.status_code != 200 else "(200, video omitted from log)"
    print(f"[generate host={host_id} ratio={ratio}] {resp.status_code}: {ct}")
    return resp


def save_video(resp, out_path):
    data = resp.json()
    if data.get("status") != "success":
        print(f"  not saving — status={data.get('status')} stage={data.get('stage')}")
        return
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(data["video_base64"]))
    print(f"  saved {out_path} — {data['resolution']}, {data['duration_seconds']}s")
    print(f"  directed_text: {data['directed_text'][:200]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onboard-url", required=True,
                   help="full URL modal deploy printed for host_onboard, e.g. "
                        "https://<workspace>--easywebinar-avatar-pipeline-host-onboard.modal.run")
    p.add_argument("--generate-url", required=True,
                   help="full URL modal deploy printed for generate, e.g. "
                        "https://<workspace>--easywebinar-avatar-pipeline-generate.modal.run")
    p.add_argument("--video", required=True)
    p.add_argument("--host-id", default="test_host_1")
    p.add_argument("--app-token", default=None)
    p.add_argument("--modal-key", default=None)
    p.add_argument("--modal-secret", default=None)
    args = p.parse_args()

    # Edge case 1: missing consent must fail fast at consent_check (no side effects).
    r = onboard(args, consent="false")
    assert r.json().get("stage") == "consent_check", "consent gate did not fire"

    # Onboard for real.
    r = onboard(args, consent="true")
    r.raise_for_status()
    ob = r.json()
    assert ob["status"] == "success", ob
    assert ob["preferred_restorer"] in ("restoreformer++", "gfpgan"), ob
    print("voice_id:", ob["voice_id"], "| requires_verification:", ob["requires_verification"])

    # Edge case 2: unknown host must fail fast at profile_lookup.
    r = generate(args, "definitely_not_onboarded", "hello")
    assert r.json().get("stage") == "profile_lookup", "profile_lookup guard did not fire"

    # Generate #1.
    r = generate(args, args.host_id,
                 "Welcome to today's webinar on serverless AI pipelines. "
                 "We're excited to show you what's possible.")
    r.raise_for_status()
    save_video(r, "out_generation_1.mp4")

    # Generate #2 — same host, different script, NO re-onboard (proves profile reuse).
    r = generate(args, args.host_id,
                 "In this session we'll cover three things: setup, deployment, and scaling.",
                 ratio="16:9")
    r.raise_for_status()
    save_video(r, "out_generation_2.mp4")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    sys.exit(main())
