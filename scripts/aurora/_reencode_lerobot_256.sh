#!/bin/bash
# Runs INSIDE lead.sif (has ffmpeg). Re-encodes every LeRobot camera mp4 under
# SRC_VIDEOS to 256x256 (square, matching the loaders' resize) under DST_VIDEOS,
# in parallel. Keyframe-per-frame (-g 1) so the video backends can seek cheaply.
# Resumable: skips outputs that already exist.
set -eo pipefail
: "${SRC_VIDEOS:?}"; : "${DST_VIDEOS:?}"; NUM_PAR="${NUM_PAR:-64}"
export SRC_VIDEOS DST_VIDEOS

find "$SRC_VIDEOS" -name '*.mp4' -print0 | xargs -0 -P "$NUM_PAR" -I{} bash -c '
  s="$1"
  rel="${s#$SRC_VIDEOS/}"
  d="$DST_VIDEOS/$rel"
  mkdir -p "$(dirname "$d")"
  if [ ! -f "$d" ]; then
    # With NUM_PAR parallel ffmpegs, libx264 auto-threads (~1.5x cores each) exhaust
    # the thread limit (EAGAIN). Force single-thread on BOTH the encoder (-threads 1
    # as an OUTPUT option) and libx264 itself (-x264-params threads=1), so total
    # threads ~= NUM_PAR. Write to .tmp then mv so a killed ffmpeg never leaves a
    # half file that the resume-skip would treat as done.
    t="$d.tmp.mp4"
    ffmpeg -y -loglevel error -i "$s" -vf scale=256:256 \
      -c:v libx264 -x264-params threads=1 -threads 1 -pix_fmt yuv420p -g 1 -an "$t" && mv "$t" "$d"
  fi
' _ {}

echo "reencode done: $(find "$DST_VIDEOS" -name '*.mp4' | wc -l) files -> ${DST_VIDEOS}"
