"""
Re-encode test_presenter_trimmed.mp4 -> test_presenter_final.mp4 as standard H.264 + AAC @ 720p,
using the ffmpeg bundled in imageio-ffmpeg (no PATH setup).

Why: the downloaded clip HAS audio, but it's VP9 video + Opus audio in an .mp4 — Windows' player
can't decode Opus-in-MP4 (so it plays silent), and the 4K VP9 video may not decode in the
pipeline's OpenCV reader. This produces a clean, universally-playable, pipeline-safe file.
"""

import subprocess

import imageio_ffmpeg

SRC = "test_presenter_trimmed.mp4"
OUT = "test_presenter_final.mp4"

ff = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using bundled ffmpeg: {ff}")

subprocess.run(
    [
        ff, "-y", "-i", SRC,
        "-vf", "scale=-2:720",          # downscale to 720p tall, keep aspect (even width)
        "-r", "25",                     # normalize frame rate
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",  # <-- transcodes Opus -> AAC so it actually plays
        "-movflags", "+faststart",
        OUT,
    ],
    check=True,
)
print(f"Done -> {OUT}")
