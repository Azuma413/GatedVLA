#!/bin/bash
# Runs INSIDE lead.sif. Proves the re-encode EAGAIN root cause (libx264 auto-threads
# x many parallel procs exhaust RLIMIT_NPROC) and that forcing single-thread fixes it.
# Reports: ulimit, threads spawned by OLD vs NEW ffmpeg, and OLD x32 vs NEW x32 success.
set -o pipefail
: "${SRC_VIDEOS:?}"; : "${OUT:?}"
mkdir -p "$OUT"; rm -f "$OUT"/*.mp4 "$OUT"/err_*.txt

echo "ulimit -u = $(ulimit -u)    nproc = $(nproc)"
V=$(find "$SRC_VIDEOS" -name '*.mp4' | head -1)
echo "test video: $V"

# max Threads of a pid while it runs
sample_threads(){
  local pid=$1 max=0 t
  while kill -0 "$pid" 2>/dev/null; do
    t=$(awk '/^Threads:/{print $2}' /proc/"$pid"/status 2>/dev/null)
    [ -n "$t" ] && [ "$t" -gt "$max" ] && max=$t
    sleep 0.03
  done
  echo "$max"
}

echo "=== [1] single OLD ffmpeg (default libx264 threads) ==="
ffmpeg -y -loglevel error -i "$V" -vf scale=256:256 -c:v libx264 -pix_fmt yuv420p -g 1 -an "$OUT/old.mp4" 2>"$OUT/err_old.txt" &
p=$!; m=$(sample_threads $p); wait $p && echo "OLD: ok, PEAK THREADS=$m" || { echo "OLD: FAILED"; cat "$OUT/err_old.txt"; }

echo "=== [2] single NEW ffmpeg (-threads 1 -x264-params threads=1) ==="
ffmpeg -y -loglevel error -i "$V" -vf scale=256:256 -c:v libx264 -x264-params threads=1 -threads 1 -pix_fmt yuv420p -g 1 -an "$OUT/new.mp4" 2>"$OUT/err_new.txt" &
p=$!; m=$(sample_threads $p); wait $p && echo "NEW: ok, PEAK THREADS=$m, out=$(stat -c%s "$OUT/new.mp4" 2>/dev/null)B" || { echo "NEW: FAILED"; cat "$OUT/err_new.txt"; }

mapfile -t VIDS < <(find "$SRC_VIDEOS" -name '*.mp4' | head -32)

run_parallel(){  # $1=label  $2..=extra ffmpeg args
  local label=$1; shift
  rm -f "$OUT"/p_*.mp4 "$OUT"/perr_*.txt
  local i
  for i in "${!VIDS[@]}"; do
    ffmpeg -y -loglevel error -i "${VIDS[$i]}" -vf scale=256:256 -c:v libx264 "$@" -pix_fmt yuv420p -g 1 -an "$OUT/p_$i.mp4" 2>"$OUT/perr_$i.txt" &
  done
  wait
  local ok eag
  ok=$(ls "$OUT"/p_*.mp4 2>/dev/null | wc -l)
  eag=$(grep -l "Resource temporarily" "$OUT"/perr_*.txt 2>/dev/null | wc -l)
  echo "$label x32: produced=$ok/32   EAGAIN=$eag"
}

echo "=== [3] OLD command x32 parallel (expect failures) ==="
run_parallel "OLD"

sleep 2
echo "=== [4] NEW command x32 parallel (expect all ok) ==="
run_parallel "NEW" -x264-params threads=1 -threads 1

echo "=== diag done ==="
