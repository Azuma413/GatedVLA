"""Alpamayo VLA model loading and inference.

Handles:
  - Model + processor download from HuggingFace (gated model).
  - CPU-only inference with SDPA attention (default).
  - Image preprocessing for multi-view camera inputs.
  - Output parsing: trajectory waypoints + reasoning trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)


@dataclass
class AlpamayoOutput:
    """Parsed output from a single Alpamayo inference call.

    Attributes:
        trajectory: List of ``[x, y]`` waypoints in the ego frame
                    (forward, left) — up to 64 points spanning 6.4 s.
        reasoning: Natural-language reasoning trace produced by the model.
        raw_text: The full raw decoded text before parsing.
    """

    trajectory: List[List[float]] = field(default_factory=list)
    reasoning: str = ""
    raw_text: str = ""


class AlpamayoInference:
    """Wrapper around the Alpamayo-R1-10B vision-language-action model.

    Usage::

        infer = AlpamayoInference(config)
        infer.load()
        out = infer.predict(camera_frames, ego_history)

    Args:
        config: Full parsed config dict (expects ``alpamayo`` section).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config.get("alpamayo", {})
        self._model_id: str = self._cfg.get("model_id", "nvidia/Alpamayo-R1-10B")
        self._device: str = self._cfg.get("device", "cpu")
        self._attn_impl: str = self._cfg.get("attn_implementation", "sdpa")
        self._num_samples: int = self._cfg.get("num_traj_samples", 1)

        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and load the model + processor onto the target device.

        Raises:
            RuntimeError: If the model cannot be loaded (e.g. OOM, auth issues).
        """
        logger.info(
            "Loading Alpamayo model '%s' on device='%s' (attn=%s)…",
            self._model_id,
            self._device,
            self._attn_impl,
        )

        try:
            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,  # CPU-safe
                "attn_implementation": self._attn_impl,
            }

            if self._device == "cpu":
                model_kwargs["device_map"] = "cpu"
            else:
                model_kwargs["device_map"] = "auto"

            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                **model_kwargs,
            )
            self._model.eval()
            logger.info("Model loaded successfully.")

        except Exception as exc:
            logger.error("Failed to load Alpamayo model: %s", exc)
            raise RuntimeError(f"Alpamayo model load failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        camera_frames: Dict[str, np.ndarray],
        ego_history: List[Dict[str, float]],
    ) -> AlpamayoOutput:
        """Run a single forward pass of the VLA model.

        Args:
            camera_frames: Dict mapping camera name → RGB numpy array ``(H, W, 3)``.
                           Expected keys: ``front_left``, ``front_wide``,
                           ``front_right``, ``front_tele``.
            ego_history: List of recent ego-state dicts (oldest first).

        Returns:
            An :class:`AlpamayoOutput` with trajectory and reasoning.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        images = self._prepare_images(camera_frames)
        ego_text = self._format_ego_history(ego_history)

        # Build the conversation / prompt in the format Alpamayo expects.
        prompt = self._build_prompt(len(images), ego_text)

        try:
            inputs = self._processor(
                text=prompt,
                images=images,
                return_tensors="pt",
            )

            # Move tensors to the correct device.
            device = self._device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )

            # Decode only the newly generated tokens.
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_len:]
            raw_text = self._processor.decode(generated_ids, skip_special_tokens=True)

            return self._parse_output(raw_text)

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during inference. Consider switching to CPU.")
            raise
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return AlpamayoOutput(
                trajectory=self._default_trajectory(),
                reasoning=f"[inference error: {exc}]",
                raw_text="",
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(num_images: int, ego_text: str) -> str:
        """Build the text prompt that accompanies the images.

        Alpamayo expects a multi-image prompt with driving-specific instructions.

        Args:
            num_images: Number of images being passed.
            ego_text: Formatted ego-motion history string.

        Returns:
            The text prompt string.
        """
        image_tokens = " ".join(["<image>"] * num_images)
        prompt = (
            f"{image_tokens}\n"
            "You are an autonomous driving model. You are given multi-view camera "
            "images from a vehicle and recent ego-motion history.\n\n"
            f"Ego-motion history:\n{ego_text}\n\n"
            "Predict the future trajectory of the ego vehicle as a sequence of "
            "(x, y) waypoints in the ego frame (x=forward, y=left) at 10 Hz for "
            "the next 6.4 seconds (64 waypoints). Also provide your reasoning.\n\n"
            "Output format:\n"
            "REASONING: <your reasoning>\n"
            "TRAJECTORY: [[x1,y1],[x2,y2],...]\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_images(camera_frames: Dict[str, np.ndarray]) -> List[Image.Image]:
        """Convert camera numpy arrays to PIL Images in canonical order.

        Args:
            camera_frames: Camera name → RGB numpy array.

        Returns:
            List of PIL Images in order: front_left, front_wide, front_right, front_tele.
        """
        order = ["front_left", "front_wide", "front_right", "front_tele"]
        images: List[Image.Image] = []
        for name in order:
            if name in camera_frames:
                img = Image.fromarray(camera_frames[name])
                images.append(img)
            else:
                logger.warning("Missing frame for '%s' — using black placeholder.", name)
                images.append(Image.new("RGB", (800, 600), (0, 0, 0)))
        return images

    # ------------------------------------------------------------------
    # Ego history formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_ego_history(ego_history: List[Dict[str, float]]) -> str:
        """Format the ego-motion history into a readable string.

        Args:
            ego_history: List of ego-state dicts.

        Returns:
            Multi-line string with one state per line.
        """
        if not ego_history:
            return "(no history yet)"
        lines: List[str] = []
        for i, state in enumerate(ego_history[-10:]):  # Last 10 states
            lines.append(
                f"  t-{len(ego_history) - 1 - i}: "
                f"pos=({state['x']:.1f}, {state['y']:.1f}), "
                f"speed={state['speed_kmh']:.1f} km/h, "
                f"heading={state['heading']:.1f}°"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(raw_text: str) -> AlpamayoOutput:
        """Parse the model's text output into trajectory + reasoning.

        Tries to find ``TRAJECTORY:`` and ``REASONING:`` markers.
        Falls back to default trajectory if parsing fails.

        Args:
            raw_text: The decoded model output string.

        Returns:
            Parsed :class:`AlpamayoOutput`.
        """
        import json as _json

        reasoning = ""
        trajectory: List[List[float]] = []

        # Extract reasoning
        if "REASONING:" in raw_text:
            parts = raw_text.split("REASONING:", 1)
            rest = parts[1]
            if "TRAJECTORY:" in rest:
                reasoning = rest.split("TRAJECTORY:", 1)[0].strip()
            else:
                reasoning = rest.strip()
        elif "reasoning" in raw_text.lower():
            reasoning = raw_text.strip()

        # Extract trajectory
        if "TRAJECTORY:" in raw_text:
            traj_part = raw_text.split("TRAJECTORY:", 1)[1].strip()
            # Find the JSON array
            bracket_start = traj_part.find("[")
            if bracket_start != -1:
                # Find matching closing bracket
                depth = 0
                end_idx = bracket_start
                for i, ch in enumerate(traj_part[bracket_start:], start=bracket_start):
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                try:
                    trajectory = _json.loads(traj_part[bracket_start:end_idx])
                except _json.JSONDecodeError:
                    logger.warning("Could not parse trajectory JSON; using default.")

        if not trajectory:
            trajectory = AlpamayoInference._default_trajectory()
            if not reasoning:
                reasoning = raw_text.strip() if raw_text.strip() else "[no output]"

        return AlpamayoOutput(
            trajectory=trajectory,
            reasoning=reasoning or "[no reasoning provided]",
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _default_trajectory() -> List[List[float]]:
        """Return a straight-ahead default trajectory (64 waypoints).

        Used as a fallback when the model fails to produce a valid trajectory.
        Assumes 10 Hz and ~30 km/h ≈ 8.3 m/s → 0.83 m per step.

        Returns:
            List of ``[x, y]`` pairs.
        """
        step_m = 0.83  # ~30 km/h at 10 Hz
        return [[round(step_m * (i + 1), 3), 0.0] for i in range(64)]
