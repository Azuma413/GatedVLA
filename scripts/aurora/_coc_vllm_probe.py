#!/usr/bin/env python
"""Probe whether an Aurora-XPU-hosted vLLM OpenAI server accepts the exact
multimodal request shape the CoC autolabeler's remote QwenWrapper emits.

Tests, against http://127.0.0.1:${PORT}/v1 :
  A. stock autolabeler format: one video_url with comma-joined base64 JPEG frames
  B. fallback: a list of image_url data URIs
  C. guided_json (structured decoding) on the image-list request

Prints PASS/FAIL per case so we know what (if anything) to patch before the
full pipeline. Uses only stdlib + openai + Pillow (all in the frameworks module).
"""
import base64
import io
import os
import sys
import traceback

import openai
from PIL import Image

PORT = os.environ.get("PORT", "50000")
MODEL = os.environ.get("MODEL_NAME_SERVED", "Qwen/Qwen3-VL-2B-Instruct")
BASE_URL = f"http://127.0.0.1:{PORT}/v1"


def make_frame_b64(color) -> str:
    img = Image.new("RGB", (256, 144), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


FRAMES = [make_frame_b64(c) for c in [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]]

client = openai.Client(base_url=BASE_URL, api_key="EMPTY", timeout=120, max_retries=0)


def run_case(name, content, extra_body=None):
    print(f"\n===== CASE {name} =====", flush=True)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_completion_tokens=128,
            extra_body=extra_body,
        )
        txt = resp.choices[0].message.content
        print(f"PASS {name}: {txt[:300]!r}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return False


results = {}

# A. Stock autolabeler remote format: single video_url, comma-joined base64 JPEGs.
video_uri = "data:video/jpeg;base64," + ",".join(FRAMES)
results["A_joined_video_url"] = run_case(
    "A_joined_video_url",
    [
        {"type": "text", "text": "In one short sentence, what changes across the frames?"},
        {"type": "video_url", "video_url": {"url": video_uri}},
    ],
    extra_body={"mm_processor_kwargs": {"fps": [2]}},
)

# B. Fallback: list of image_url data URIs.
img_content = [{"type": "text", "text": "In one short sentence, what changes across these images?"}]
img_content += [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in FRAMES
]
results["B_image_list"] = run_case("B_image_list", img_content)

# C. Guided JSON (structured decoding) on the image-list request.
schema = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "num_frames": {"type": "integer"}},
    "required": ["summary", "num_frames"],
}
results["C_guided_json"] = run_case(
    "C_guided_json",
    img_content,
    extra_body={"guided_json": schema, "guided_decoding_backend": "outlines"},
)

print("\n===== SUMMARY =====", flush=True)
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)

# Exit non-zero only if the server was totally unreachable (all fail).
sys.exit(0 if any(results.values()) else 3)
