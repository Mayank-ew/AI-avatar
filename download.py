"""
Download + trim a YouTube clip WITH audio, using the ffmpeg binary bundled in the
`imageio-ffmpeg` pip package (no system ffmpeg / PATH setup needed).

Grabs a separate best-video + best-audio and merges them (that's what guarantees audio — the
earlier `-f 18` fallback picked a video-only stream), trimmed to 0:15 -> 1:45.

Run:  py -m pip install imageio-ffmpeg yt-dlp
      py download.py
"""

import subprocess
import sys

import imageio_ffmpeg

URL = "https://www.youtube.com/watch?v=zBh6i-ZU5gQ"
SECTION = "*0:15-1:45"   # start -> end (90s window)
OUT = "test_presenter_trimmed.mp4"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()  # downloads the bundled binary on first call
print(f"Using bundled ffmpeg: {ffmpeg}")

subprocess.run(
    [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg,
        "--download-sections", SECTION,
        "-f", "bv*+ba/b",                 # best video + best AUDIO, merged (fallback: best combined)
        "--merge-output-format", "mp4",
        "-o", OUT,
        URL,
    ],
    check=True,
)
print(f"Done -> {OUT}")
