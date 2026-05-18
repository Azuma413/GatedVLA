import logging
import sys
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger(__name__)


def _ensure_alpamayo_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    alpamayo_root = project_root / "3rd_party" / "alpamayo"
    alpamayo_src = alpamayo_root / "src"
    for path in (alpamayo_root, alpamayo_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


class AlpamayoInference:
    """Small online inference wrapper for SFT Alpamayo checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        torch_dtype: torch.dtype = torch.bfloat16,
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
    ) -> None:
        _ensure_alpamayo_on_path()

        from alpamayo_r1.processor.qwen_processor import QwenProcessor  # noqa: PLC0415
        from finetune.sft.models.sft_alpamayo_r1 import (  # noqa: PLC0415
            TrainableAlpamayoR1,
        )

        self.device = device
        self.torch_dtype = torch_dtype
        self.top_p = top_p
        self.temperature = temperature
        self.num_traj_samples = num_traj_samples
        self.max_generation_length = max_generation_length

        checkpoint_path = Path(checkpoint_path)
        LOG.info("Loading Alpamayo checkpoint from %s", checkpoint_path)
        try:
            self.model = TrainableAlpamayoR1.from_pretrained(
                str(checkpoint_path),
                dtype=torch_dtype,
            ).to(device)
        except TypeError:
            self.model = TrainableAlpamayoR1.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch_dtype,
            ).to(device)
        self.model.eval()

        qwen_processor = QwenProcessor(
            vlm_name_or_path=self.model.config.vlm_name_or_path,
            traj_vocab_size=self.model.config.traj_vocab_size,
            min_pixels=self.model.config.min_pixels,
            max_pixels=self.model.config.max_pixels,
        )
        self.preprocess_data = qwen_processor.get_preprocess_data_fn(
            num_tokens_per_history_traj=self.model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=self.model.config.tokens_per_future_traj,
            components_order=["image", "traj_history", "prompt", "traj_future"],
            components_prompt=["traj_future"],
            label_components=["traj_future"],
            generation_mode=True,
        )
        self.collate_fn = qwen_processor.collate_fn

    @torch.inference_mode()
    def predict_waypoints(self, sample_data: dict[str, Any]) -> torch.Tensor:
        """Return Alpamayo future waypoints as ``[1, 64, 2]`` in ego frame."""
        data = dict(sample_data)
        data["tokenized_data"] = self.preprocess_data(data=data)

        batched = self.collate_fn([data], padding_side="left")
        batched = _move_to_device(batched, self.device)

        with torch.autocast(self.device.type, dtype=self.torch_dtype):
            outputs = self.model.sample_trajectories_from_data(
                data=batched,
                top_p=self.top_p,
                temperature=self.temperature,
                num_traj_samples=self.num_traj_samples,
                max_generation_length=self.max_generation_length,
                return_extra=True,
            )

        pred_xyz = outputs[0]
        # [B, num_sets, num_samples, T, 3] -> [B, T, 2].
        return pred_xyz[:, 0, 0, :, :2].float()
