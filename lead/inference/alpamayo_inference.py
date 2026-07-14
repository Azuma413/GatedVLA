import logging
import os
import sys
import typing
from pathlib import Path
from typing import Any

import torch

LOG = logging.getLogger(__name__)

# Chat-template version drives which Alpamayo generation is loaded. "r1" is the
# original Alpamayo-R1 (v1) stack; "r1_5" is the navigation-conditioned 1.5 stack
# whose model/processor live in the alpamayo-recipes package.
_CHAT_TEMPLATE_VERSION = os.getenv("ALPAMAYO_CHAT_TEMPLATE_VERSION", "r1")
_IS_R1_5 = _CHAT_TEMPLATE_VERSION in ("r1_5", "r1.5")


def _ensure_alpamayo_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    third_party = project_root / "3rd_party"
    if _IS_R1_5:
        # alpamayo1_5_sft (recipes/), alpamayo processor+data (src/), alpamayo_r1 (alpamayo/src)
        paths = [
            third_party / "alpamayo-recipes" / "recipes",
            third_party / "alpamayo-recipes" / "src",
            third_party / "alpamayo" / "src",
        ]
    else:
        paths = [third_party / "alpamayo", third_party / "alpamayo" / "src"]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _ensure_self_shim() -> None:
    """The alpamayo1_5_sft package uses ``typing.Self`` (3.11+); LEAD runs on 3.10."""
    if not hasattr(typing, "Self"):
        try:
            from typing_extensions import Self  # noqa: PLC0415

            typing.Self = Self  # type: ignore[attr-defined]
        except ImportError:
            typing.Self = Any  # type: ignore[attr-defined]


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

        if _IS_R1_5:
            _ensure_self_shim()
            from alpamayo.processor.qwen_processor import QwenProcessor  # noqa: PLC0415
            from alpamayo1_5_sft.models.sft_alpamayo_r1 import (  # noqa: PLC0415
                TrainableAlpamayoR1,
            )
        else:
            from alpamayo_r1.processor.qwen_processor import QwenProcessor  # noqa: PLC0415
            from finetune.sft.models.sft_alpamayo_r1 import (  # noqa: PLC0415
                TrainableAlpamayoR1,
            )

        self.device = device
        self.torch_dtype = torch_dtype
        self.top_p = top_p
        # Env overrides let us reduce the frame-to-frame speed variance of the sampled
        # trajectory (single stochastic sample at temperature 0.6 lurches between "go" and
        # "stop"): lower temperature -> more deterministic; more samples -> median-aggregated.
        self.temperature = float(os.getenv("ALPAMAYO_TEMPERATURE", temperature))
        self.num_traj_samples = int(os.getenv("ALPAMAYO_NUM_SAMPLES", num_traj_samples))
        self.max_generation_length = max_generation_length

        checkpoint_path = Path(checkpoint_path)
        LOG.info("Loading Alpamayo (%s) checkpoint from %s", _CHAT_TEMPLATE_VERSION, checkpoint_path)

        # The 1.5 checkpoint records the VLM path from the training cluster, which
        # does not exist here. Patch the config to a local Qwen3-VL snapshot before
        # construction: the config __init__ builds the processor from vlm_name_or_path,
        # so a `from_pretrained(vlm_name_or_path=...)` kwarg would apply too late.
        from_pretrained_kwargs: dict[str, Any] = {}
        if _IS_R1_5:
            import json  # noqa: PLC0415

            from alpamayo_r1.config import AlpamayoR1Config  # noqa: PLC0415

            project_root = Path(__file__).resolve().parents[2]
            default_vlm = project_root / "outputs" / "models" / "Qwen3-VL-8B-Instruct"
            vlm_path = os.getenv("ALPAMAYO_VLM_PATH", str(default_vlm))
            cfg_dict = json.loads((checkpoint_path / "config.json").read_text())
            cfg_dict["vlm_name_or_path"] = vlm_path
            from_pretrained_kwargs["config"] = AlpamayoR1Config.from_dict(cfg_dict)

        try:
            self.model = TrainableAlpamayoR1.from_pretrained(
                str(checkpoint_path),
                dtype=torch_dtype,
                **from_pretrained_kwargs,
            ).to(device)
        except TypeError:
            self.model = TrainableAlpamayoR1.from_pretrained(
                str(checkpoint_path),
                torch_dtype=torch_dtype,
                **from_pretrained_kwargs,
            ).to(device)
        self.model.eval()

        if _IS_R1_5:
            qwen_processor = QwenProcessor(
                vlm_name_or_path=self.model.config.vlm_name_or_path,
                traj_vocab_size=self.model.config.traj_vocab_size,
                min_pixels=self.model.config.min_pixels,
                max_pixels=self.model.config.max_pixels,
                include_camera_ids=self.model.config.include_camera_ids,
                include_frame_nums=self.model.config.include_frame_nums,
                chat_template_version=_CHAT_TEMPLATE_VERSION,
            )
            components_order = ["image", "traj_history", "route", "prompt", "traj_future"]
        else:
            qwen_processor = QwenProcessor(
                vlm_name_or_path=self.model.config.vlm_name_or_path,
                traj_vocab_size=self.model.config.traj_vocab_size,
                min_pixels=self.model.config.min_pixels,
                max_pixels=self.model.config.max_pixels,
            )
            components_order = ["image", "traj_history", "prompt", "traj_future"]

        self.preprocess_data = qwen_processor.get_preprocess_data_fn(
            num_tokens_per_history_traj=self.model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=self.model.config.tokens_per_future_traj,
            components_order=components_order,
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
        # [B, num_sets, num_samples, T, 3] -> median over samples -> [B, T, 2].
        return pred_xyz[:, 0, :, :, :2].median(dim=1).values.float()
