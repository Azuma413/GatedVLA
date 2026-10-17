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


def _make_cfg(base_vlm: str, eye_gate: dict = None, anchor: dict = None, drive: dict = None):
    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenDrivePI",
                "qwenvl": {"base_vlm": base_vlm, "attn_implementation": "sdpa"},
                "drive": {"current_image_pixels": 128 * 64, "history_image_pixels": 64 * 32, **(drive or {})},
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
    assert np.all(np.isfinite(out))


def test_eye_gate(tiny_qwen_drive_dir):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    model = build_framework(_make_cfg(str(path), {"enabled": True, "mode": "learned"})).float()
    rng = np.random.default_rng(0)
    batch = [_sample(rng), _sample(rng)]
    out = model(batch)
    for key in ("loss_gated", "loss_full", "loss_budget", "keep_view0", "keep_view1"):
        assert key in out, key
    # gate starts around init_keep (0.2) and, with a random output layer, already depends on the input
    assert abs(out["keep_view0"].item() - 0.2) < 0.1 and abs(out["keep_view1"].item() - 0.2) < 0.1
    assert "mask_mi" not in out  # mi_loss_weight defaults to 0
    out["action_loss"].backward()
    assert all(p.grad is not None for p in model.eye_gate.parameters())

    # keep maps are split per frame following the image grids (history frame has the smaller pixel budget)
    model.eval()
    pred = model.predict_action([_sample(rng)])
    maps = pred["keep_maps"][0]
    assert len(maps) == 2 and maps[0].size < maps[1].size
    assert maps[1].std() > 0


def test_eye_gate_mutual_information(tiny_qwen_drive_dir):
    """I(m; x) = H(mean m) − mean H(m): ≥ 0, 0 for input-independent masks, and its gradient reaches the gate."""
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    model = build_framework(
        _make_cfg(str(path), {"enabled": True, "mode": "learned", "mi_loss_weight": 0.1})
    ).float()
    rng = np.random.default_rng(0)
    out = model([_sample(rng) for _ in range(3)])
    for key in ("mask_mi", "mask_marginal_entropy", "mask_cond_entropy"):
        assert key in out, key
    assert out["mask_mi"].item() >= 0
    assert torch.isclose(out["mask_mi"], out["mask_marginal_entropy"] - out["mask_cond_entropy"], atol=1e-6)

    # input-independent masks (uniform, or a fixed spatial pattern) give I = 0 with a zero gradient
    view_ids = torch.tensor([[-1, 0, 0, 1, 1, 1]] * 4)
    pattern = torch.tensor([1.0, 0.3, 0.9, 0.2, 0.5, 0.7])
    for fixed in (torch.full((4, 6), 0.2), pattern.repeat(4, 1)):
        keep = fixed.clone().requires_grad_(True)
        mi, _, _ = model._mask_mutual_information(keep, view_ids)
        mi.backward()
        assert abs(mi.item()) < 1e-6 and keep.grad.abs().max().item() < 1e-6
    # an input-dependent mask has I > 0
    keep = torch.tensor([[1.0, 0.1, 0.9, 0.9, 0.1, 0.5], [1.0, 0.9, 0.1, 0.1, 0.9, 0.5]] * 2)
    assert model._mask_mutual_information(keep, view_ids)[0].item() > 0.5


def test_eye_gate_logit_l2_without_budget(tiny_qwen_drive_dir):
    """Budget loss off, β·mean ℓ² on: the L2 gradient reaches a saturated gate (it bypasses the sigmoid)."""
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    cfg = {"enabled": True, "mode": "learned", "budget_loss_weight": 0.0, "mi_loss_weight": 0.1, "logit_l2_weight": 1e-3}
    model = build_framework(_make_cfg(str(path), cfg)).float()
    rng = np.random.default_rng(0)
    batch = [_sample(rng) for _ in range(2)]
    out = model(batch)
    assert "loss_budget" not in out and "gate_logit_rms" in out and "mask_mi" in out

    # saturate the gate fully closed (ℓ ≈ -30): only the L2 term still moves the output bias
    with torch.no_grad():
        model.eye_gate.fc2.weight.zero_()
        model.eye_gate.fc2.bias.fill_(-30.0)
    out = model(batch)
    assert abs(out["gate_logit_rms"].item() - 30.0) < 1e-3
    out["action_loss"].backward()
    assert model.eye_gate.fc2.bias.grad.item() < -1e-3  # descent raises the bias back towards 0


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


def test_text_first_visual_only_gate_distill(tiny_qwen_drive_dir):
    """[text, images] order, DiT cross-attention on visual tokens only, gate trained only by the distillation loss."""
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    gate = {"enabled": True, "mode": "learned", "parameterization": "budgeted", "budget_loss_weight": 0.0,
            "gated_loss": "distill", "gated_grad": "gate"}
    model = build_framework(_make_cfg(str(path), gate, drive={"text_first": True, "dit_visual_only": True})).float()
    rng = np.random.default_rng(0)
    batch = [_sample(rng), _sample(rng)]

    images, instructions, _ = model._prepare_inputs(batch)
    _, attn, gi = model._encode_vl_hidden_states(images, instructions)
    visual = gi["view_ids"] >= 0
    assert torch.equal(attn, attn & visual) and attn.any()  # the expert sees visual tokens only
    pos = torch.arange(visual.shape[1])[None].expand_as(visual)
    first_visual = pos[visual.cumsum(1) == 1].view(-1)
    assert gi["text_mask"].any(1).all()
    assert bool((pos[gi["text_mask"]].view(-1) < first_visual.repeat_interleave(gi["text_mask"].sum(1))).all())
    bias = model._cross_attention_bias(attn, model._gate_keep(gi, attn), torch.float32)
    assert torch.isinf(bias[~visual]).all() and torch.isfinite(bias[visual]).all()

    out = model(batch)
    for key in ("gate_loss", "loss_distill", "loss_gated", "loss_full", "keep_xsample_std"):
        assert key in out, key
    assert out["loss_distill"] > 0
    gate_params = list(model.eye_gate.parameters())
    vlm_params = [p for p in model.qwen_vl_interface.parameters() if p.requires_grad]
    out["gate_loss"].backward()  # gate path: reaches the gate, never the VLM / projectors
    assert all(p.grad is not None for p in gate_params)
    assert all(p.grad is None for p in vlm_params)
    assert all(p.grad is None for p in model.project_layers.parameters())
    gate_grads = [p.grad.clone() for p in gate_params]
    out["action_loss"].backward()  # full path: VLM + DiT, nothing on the gate
    assert any(p.grad is not None for p in vlm_params)
    assert all(torch.equal(p.grad, g) for p, g in zip(gate_params, gate_grads))

    model.eval()
    pred = model.predict_action([_sample(rng)])
    assert np.all(np.isfinite(pred["normalized_actions"]))


def test_gate_input_standardization():
    """A shared near-constant massive channel makes the plain gate input-independent; standardising fixes it."""
    from starVLA.model.framework.VLM4A.QwenDrivePI import ViewGate

    torch.manual_seed(0)
    width = 64

    def tokens(n):
        x = torch.randn(n, width)
        x[:, 0] = 45.0 + 0.1 * torch.randn(n)  # massive, near-constant channel shared by visual and text tokens
        return x

    h, text = tokens(200), tokens(20)
    spread = {}
    for standardize in (False, True):
        torch.manual_seed(1)
        gate = ViewGate(width, 32, init_keep=0.5, standardize=standardize).train()
        logits = gate(h, text)
        spread[standardize] = (logits.std() / logits.abs().mean()).item()
    assert spread[True] > 5 * spread[False], spread

    # running statistics: initialised from the first batch, then an EMA; used unchanged in eval
    gate = ViewGate(width, 32, init_keep=0.5).train()
    gate(h, text)
    assert bool(gate.std_h.initialized) and torch.allclose(gate.std_h.mean, h.mean(0), atol=1e-5)
    mean_before = gate.std_h.mean.clone()
    gate.eval()
    gate(tokens(50), text)
    assert torch.equal(gate.std_h.mean, mean_before)


def test_budgeted_keep():
    """Budget and contrast fixed by construction: exact mean, invariant to the score scale, soft, differentiable."""
    from starVLA.model.framework.VLM4A.QwenDrivePI import budgeted_keep

    torch.manual_seed(0)
    s = torch.randn(900, requires_grad=True)
    m = budgeted_keep(s, 0.2, 2.0)
    assert abs(m.mean().item() - 0.2) < 1e-4
    # scaling the scores (what saturates a plain sigmoid gate) does not change the mask
    assert torch.allclose(budgeted_keep(100 * s.detach(), 0.2, 2.0), m.detach(), atol=1e-5)
    # soft: the bulk of the tokens (|z| <= 2) stays away from 0/1; only the far tails approach it
    assert ((m > 0.005) & (m < 0.995)).float().mean().item() > 0.9
    (m * torch.randn(900)).sum().backward()
    assert (s.grad != 0).all()


def test_budgeted_keep_gradient_matches_finite_differences():
    """The bisection is not differentiated; the implicit-function term must make the gradient exact."""
    from starVLA.model.framework.VLM4A.QwenDrivePI import budgeted_keep

    torch.manual_seed(0)
    s = torch.randn(50, dtype=torch.float64, requires_grad=True)
    c = torch.randn(50, dtype=torch.float64)
    loss = lambda x: (budgeted_keep(x, 0.2, 2.0, iters=200) * c).sum()  # noqa: E731
    loss(s).backward()
    eps, num = 1e-6, torch.zeros(50, dtype=torch.float64)
    with torch.no_grad():
        for i in range(50):
            d = torch.zeros(50, dtype=torch.float64)
            d[i] = eps
            num[i] = (loss(s + d) - loss(s - d)) / (2 * eps)
    assert torch.allclose(s.grad, num, rtol=1e-4, atol=1e-7), (s.grad - num).abs().max()


def test_eye_gate_budgeted(tiny_qwen_drive_dir):
    from starVLA.model.framework.base_framework import build_framework

    path, _ = tiny_qwen_drive_dir
    cfg = {"enabled": True, "mode": "learned", "parameterization": "budgeted", "contrast": 2.0, "budget_loss_weight": 0.0}
    model = build_framework(_make_cfg(str(path), cfg)).float()
    rng = np.random.default_rng(0)
    out = model([_sample(rng), _sample(rng)])
    assert "loss_budget" not in out
    assert abs(out["keep_view0"].item() - 0.2) < 1e-3 and abs(out["keep_view1"].item() - 0.2) < 1e-3
    out["action_loss"].backward()
    assert model.eye_gate.fc1.weight.grad.abs().sum() > 0


def test_drive_prompt():
    from starVLA.dataloader.pai_av_drive import drive_prompt

    assert drive_prompt("predict trajectory", np.array([24.71, -3.94]), 8.26) == (
        "predict trajectory. speed: 8.3 m/s. target point: x 24.7 m, y -3.9 m"
    )
    assert drive_prompt("predict trajectory") == "predict trajectory"


def test_unicycle_round_trip_and_normalizer():
    """accel / curvature encoding of a smooth turn decodes back to the same positions; mean/std normaliser."""
    from starVLA.dataloader.pai_av_drive import Normalizer
    from starVLA.model.modules.action_space.unicycle import UnicycleCodec

    t = np.arange(1, 26) * 0.1
    v0, kappa = 8.0, 0.05  # constant speed on a 20 m radius arc
    yaw = kappa * v0 * t
    xy = np.stack([np.sin(yaw) / kappa, (1 - np.cos(yaw)) / kappa], -1)
    codec = UnicycleCodec(25)
    controls = codec.encode(xy, yaw, v0)
    assert np.allclose(controls[5:-5, 1], kappa, atol=2e-3) and np.allclose(controls[5:-5, 0], 0.0, atol=0.05)
    assert np.abs(codec.decode(controls[None], [v0])[0].numpy() - xy).max() < 0.05

    norm = Normalizer({"action": {"norm": "mean_std", "mean": [0.0, 0.0], "std": [1.0, 0.02]}})
    z = norm.normalize("action", controls)
    assert np.allclose(norm.unnormalize("action", z), controls, atol=1e-5)


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
    assert np.allclose(pred["normalized_actions"][0], norm, atol=1e-5)  # no clamp (action_clip off)


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


@pytest.fixture(scope="module")
def tiny_control_anchors(tmp_path_factory):
    rng = np.random.default_rng(0)
    anchors = np.stack([np.stack([np.full(10, a), np.full(10, k)], -1) for a in (-1.0, 0.0, 1.0) for k in (-0.05, 0.0, 0.05)])
    path = tmp_path_factory.mktemp("canchors") / "anchors.npz"
    np.savez(path, anchors=anchors.astype(np.float32) + rng.normal(0, 0.001, anchors.shape).astype(np.float32),
             residual_std=np.full((10, 2), 0.1, np.float32), space="control", dt=np.float32(0.1),
             action_mean=np.array([0.0, 0.0], np.float32), action_std=np.array([0.9, 0.025], np.float32))
    return str(path)


def test_anchor_control_space(tiny_qwen_drive_dir, tiny_control_anchors):
    """Control-space anchors: soft targets from metre distances after integrating from v0; stage 2 residual in
    control space; eval gets per-sample anchor trajectories."""
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.modules.action_space.unicycle import unicycle_rollout

    path, _ = tiny_qwen_drive_dir
    for stage in (1, 2):
        model = build_framework(_make_cfg(str(path), anchor=_anchor_cfg(tiny_control_anchors, stage=stage))).float()
        assert model.anchor_space == "control"
        rng = np.random.default_rng(0)
        batch = [dict(_sample(rng), v0=float(v)) for v in (3.0, 12.0)]
        out = model(batch)
        assert torch.isfinite(out["action_loss"]) and ("loss_flow" in out) == (stage == 2)
        out["action_loss"].backward()
        assert model.anchor_classifier[-1].weight.grad is not None
        model.eval()
        pred = model.predict_action([batch[1]])
        assert pred["anchor_xy"].shape == (1, 9, 11, 2) and pred["anchor_reference"].shape == (1, 11, 2)
        if stage == 1:  # prediction = reference controls; its rollout from v0 is the reported reference trajectory
            ctl = pred["normalized_actions"][0] * np.array([0.9, 0.025])
            xy = unicycle_rollout(torch.as_tensor(ctl, dtype=torch.float64), torch.tensor(12.0, dtype=torch.float64))
            assert np.allclose(xy.numpy(), pred["anchor_reference"][0, 1:], atol=1e-3)

    # nearest anchor in metres depends on the speed: same controls, different v0 -> distances scale with speed
    d_slow = model._anchor_distances(model.anchors[:1], torch.tensor([2.0]))
    d_fast = model._anchor_distances(model.anchors[:1], torch.tensor([20.0]))
    assert d_fast[0, 2] > d_slow[0, 2]
