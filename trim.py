"""
Throwaway helper: trim test_presenter_clip.mp4 -> test_presenter_trimmed.mp4 using the ffmpeg
binary bundled inside the `imageio-ffmpeg` pip package (no system ffmpeg / PATH needed).

Takes 90 seconds starting at 0:15 (so 0:15 -> 1:45). Adjust START/DURATION below if you want a
different window.

Run:  py -m pip install imageio-ffmpeg   (once)
      py trim.py
"""

import subprocess

import imageio_ffmpeg

START = "15"        # seconds (or "MM:SS")
DURATION = "90"     # seconds to keep
SRC = "test_presenter_clip.mp4"
OUT = "test_presenter_trimmed.mp4"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()  # downloads the bundled binary on first call
print(f"Using bundled ffmpeg: {ffmpeg}")

subprocess.run(
    [ffmpeg, "-y", "-ss", START, "-i", SRC, "-t", DURATION,
     "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", OUT],
    check=True,
)
print(f"Done -> {OUT}")
