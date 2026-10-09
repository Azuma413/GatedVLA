"""Smoke test for the QwenDrivePI framework on a tiny random Qwen-Drive-format checkpoint (CPU).

The checkpoint mimics Qwen/Qwen-Drive-1.0-4B: top-level ``model_type: qwen_drive`` with the
Qwen3.5 config under ``vlm_config`` and VLM weights stored with a ``vlm.`` prefix.
Tokenizer / processor files are fetched from the hub (weights are not).
"""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
    pytest.skip("transformers without Qwen3.5", allow_module_level=True)

from PIL import Image
from safetensors.torch import save_file

QWEN_DRIVE_ID = "Qwen/Qwen-Drive-1.0-4B"
PROCESSOR_FILES = [
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def _tiny_vlm_config(real_cfg: dict) -> dict:
    vlm = json.loads(json.dumps(real_cfg["vlm_config"]))
    text = vlm["text_config"]
    text.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    text["rope_parameters"]["mrope_section"] = [1, 1, 0]
    vlm["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    return vlm


@pytest.fixture(scope="module")
def tiny_qwen_drive_dir(tmp_path_factory):
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    out = tmp_path_factory.mktemp("Qwen-Drive-tiny")
    try:
        for name in PROCESSOR_FILES + ["config.json"]:
            src = hf_hub_download(QWEN_DRIVE_ID, name)
            (out / name).write_bytes(open(src, "rb").read())
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"cannot fetch {QWEN_DRIVE_ID} processor files: {e}")

    real_cfg = json.loads((out / "config.json").read_text())
    real_cfg["vlm_config"] = _tiny_vlm_config(real_cfg)
    (out / "config.json").write_text(json.dumps(real_cfg))

    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**real_cfg["vlm_config"]))
    state = {f"vlm.{k}": v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(out / "model.safetensors"))
    return out, model


def _make_cfg(base_vlm: str, eye_gate: dict = None, anchor: dict = None):
    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenDrivePI",
                "qwenvl": {"base_vlm": base_vlm, "attn_implementation": "sdpa"},
                "drive": {"current_image_pixels": 128 * 64, "history_image_pixels": 64 * 32},
                "eye_gate": eye_gate or {"enabled": False},
                "anchor": anchor or {"enabled": False},
            },
            "datasets": {"vla_data": {}},
            "trainer": {"repeated_diffusion_steps": 2},
        }
    )


def _sample(rng):
    frames = [Image.fromarray(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)) for _ in range(2)]
    return {
        "image": frames,
        "state": rng.uniform(-1, 1, (5, 10)).astype(np.float32),
        "action": rng.uniform(-1, 1, (10, 2)).astype(np.float32),
    }


def test_vlm_weights_loaded(tiny_qwen_drive_dir):
    from starVLA.model.modules.vlm.QwenDrive import load_qwen_drive_vlm

    path, ref = tiny_qwen_drive_dir
    loaded = load_qwen_drive_vlm(str(path), dtype=torch.float32)
    ref_sd, sd = ref.state_dict(), loaded.state_dict()
    for k in ref_sd:
        assert torch.equal(ref_sd[k], sd[k]), k


def test_forward_and_predict(tiny_qwen_drive_dir):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    model = build_framework(_make_cfg(str(path))).float()
    # DiT depth follows the VLM depth; interleaving -> projectors only on cross-attention blocks.
    assert len(model.action_model.model.transformer_blocks) == 4
    assert model.cross_layer_ids == [0, 2]
    assert model.vlm_layer_ids == [1, 3]

    rng = np.random.default_rng(0)
    batch = [_sample(rng), _sample(rng)]
    batch[1]["lang"] = "turn left at the intersection"
    loss = model(batch)["action_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    # proj_out_* (unused DiT output path) are frozen; lm_head / mtp are LM-only.
    # Everything else, incl. the last VLM layer, must receive gradients.
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".proj_out_" in n)
    allowed = ("qwen_vl_interface.model.lm_head", "qwen_vl_interface.model.mtp")
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not [n for n in unused if not n.startswith(allowed)], unused

    model.eval()
    out = model.predict_action([_sample(rng)])["normalized_actions"]
    assert out.shape == (1, 10, 2)
    assert np.all(np.abs(out) <= 1.0)


def test_eye_gate(tiny_qwen_drive_dir):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    model = build_framework(_make_cfg(str(path), {"enabled": True, "mode": "learned"})).float()
    rng = np.random.default_rng(0)
    batch = [_sample(rng), _sample(rng)]
    out = model(batch)
    for key in ("loss_gated", "loss_full", "loss_budget", "keep_view0", "keep_view1"):
        assert key in out, key
    # gate starts almost fully open (init_keep 0.99)
    assert abs(out["keep_view0"].item() - 0.99) < 1e-3 and abs(out["keep_view1"].item() - 0.99) < 1e-3
    out["action_loss"].backward()
    assert all(p.grad is not None for p in model.eye_gate.parameters())

    # keep maps are split per frame following the image grids (history frame has the smaller pixel budget)
    model.eval()
    pred = model.predict_action([_sample(rng)])
    maps = pred["keep_maps"][0]
    assert len(maps) == 2 and maps[0].size < maps[1].size
    assert np.allclose(maps[1], 0.99, atol=1e-3)


def test_eye_gate_open_matches_ungated(tiny_qwen_drive_dir):
    """A gate fixed fully open (uniform, keep = 1 -> bias 0) must reproduce the ungated model exactly."""
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    torch.manual_seed(0)
    plain = build_framework(_make_cfg(str(path))).float().eval()
    gated = build_framework(_make_cfg(str(path), {"enabled": True, "mode": "uniform", "budget": 1.0})).float().eval()
    gated.load_state_dict(plain.state_dict())
    sample = _sample(np.random.default_rng(1))
    torch.manual_seed(123)
    a = plain.predict_action([sample])["normalized_actions"]
    torch.manual_seed(123)
    b = gated.predict_action([sample])["normalized_actions"]
    assert np.allclose(a, b, atol=1e-5)
    # and closing it changes the prediction
    closed = build_framework(_make_cfg(str(path), {"enabled": True, "mode": "uniform", "budget": 1e-6})).float().eval()
    closed.load_state_dict(plain.state_dict())
    torch.manual_seed(123)
    c = closed.predict_action([sample])["normalized_actions"]
    assert not np.allclose(a, c, atol=1e-3)


@pytest.fixture(scope="module")
def tiny_anchors(tmp_path_factory):
    rng = np.random.default_rng(0)
    steps = np.arange(10)[:, None] * 0.1
    anchors = np.stack([np.concatenate([v * steps, c * (v * steps) ** 2], 1) for v in (2, 8, 15) for c in (-0.05, 0, 0.05)])
    path = tmp_path_factory.mktemp("anchors") / "anchors.npz"
    np.savez(path, anchors=anchors.astype(np.float32) + rng.normal(0, 0.01, anchors.shape).astype(np.float32),
             residual_std=np.full((10, 2), 0.3, np.float32),
             action_q01=np.array([0.0, -1.0], np.float32), action_q99=np.array([15.0, 1.0], np.float32))
    return str(path)


def _anchor_cfg(path, **kw):
    cfg = {"enabled": True, "path": path, "stage": 2, "reference_top_n": 3, "target_top_n": 3}
    cfg.update(kw)
    return cfg


def test_anchor_stage1(tiny_qwen_drive_dir, tiny_anchors):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    model = build_framework(_make_cfg(str(path), anchor=_anchor_cfg(tiny_anchors, stage=1))).float()
    rng = np.random.default_rng(0)
    out = model([_sample(rng), _sample(rng)])
    assert {"loss_anchor_cls", "anchor_top1_acc", "anchor_topn_recall"} <= set(out) and "loss_flow" not in out
    out["action_loss"].backward()
    assert model.anchor_query.grad is not None and model.anchor_classifier[-1].weight.grad is not None
    assert all(p.grad is None for p in model.action_model.parameters())
    model.eval()
    pred = model.predict_action([_sample(rng)])
    # stage 1 predicts the reference itself
    ref = pred["anchor_reference"][0]
    norm = 2 * (ref - np.array([0.0, -1.0])) / np.array([15.0, 2.0]) - 1
    assert np.allclose(pred["normalized_actions"][0], np.clip(norm, -1, 1), atol=1e-5)


@pytest.mark.parametrize("reference", ["soft", "hard"])
def test_anchor_stage2_gradient_to_logits(tiny_qwen_drive_dir, tiny_anchors, reference):
    """With the classification loss off, only the soft reference passes the flow loss gradient to the logits."""
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    cfg = _make_cfg(str(path), anchor=_anchor_cfg(tiny_anchors, reference=reference, cls_loss_weight=0.0))
    model = build_framework(cfg).float()
    rng = np.random.default_rng(0)
    out = model([_sample(rng), _sample(rng)])
    assert "loss_flow" in out and "anchor_weight_entropy" in out
    out["action_loss"].backward()
    grad = model.anchor_classifier[-1].weight.grad
    if reference == "soft":
        assert grad is not None and grad.abs().sum() > 0
    else:
        assert grad is None or grad.abs().sum() == 0
    assert model.anchor_embed[0].weight.grad is not None
    model.eval()
    pred = model.predict_action([_sample(rng)])
    assert pred["normalized_actions"].shape == (1, 10, 2) and pred["anchor_ids"].shape[1] == (3 if reference == "soft" else 1)


def test_anchor_with_eye_gate(tiny_qwen_drive_dir, tiny_anchors):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    cfg = _make_cfg(str(path), eye_gate={"enabled": True, "mode": "learned"}, anchor=_anchor_cfg(tiny_anchors))
    model = build_framework(cfg).float()
    rng = np.random.default_rng(0)
    out = model([_sample(rng), _sample(rng)])
    assert {"loss_anchor_cls", "loss_gated", "loss_full", "loss_budget"} <= set(out)
    out["action_loss"].backward()
    assert all(p.grad is not None for p in model.eye_gate.parameters())
    model.eval()
    pred = model.predict_action([_sample(rng)])
    assert "keep_maps" in pred and "anchor_reference" in pred
