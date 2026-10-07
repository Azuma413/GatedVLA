#!/usr/bin/env bash
# RoboTwin 2.0 を lerobot 環境に導入する。
#
# lerobot の docker/Dockerfile.benchmark.robotwin を非 Docker 化したもの。
# ただし実際に踏んだ差異を反映しているため、いくつか内容が異なる (下記コメント参照)。
#
# 前提: `uv sync` 済み。
# 使い方: bash scripts/setup_robotwin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RT="$ROOT/libs/RoboTwin"
PY="$ROOT/.venv/bin/python"
ROBOTWIN_SHA=0aeea2d669c0f8516f4d5785f0aa33ba812c14b4  # lerobot CI と同じ pin
CUROBO_REF=v0.7.8                                      # RoboTwin が v1 API (curobo.types.math) を使うため
CUDA_DIR=${CUDA_DIR:-/usr/local/cuda-12.9}
# sm_90 = H200, sm_120 = RTX PRO 6000 Blackwell。環境に合わせて変更する。
ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0;12.0"}

# HF キャッシュはホーム領域を圧迫するのでワークスペース側に置く
# (FastWAM/LingBot-VA 等は 1 モデルで数十 GB になる)
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "==> 1. RoboTwin clone ($ROBOTWIN_SHA)"
[ -d "$RT" ] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$RT"
git -C "$RT" checkout -q "$ROBOTWIN_SHA"

echo "==> 2. ランタイム依存"
# RoboTwin 自身の script/requirements.txt は torch==2.4.1 / scipy==1.10.1 を pin しており、
# そのまま入れると lerobot 側の torch を壊す。lerobot Dockerfile のサブセット + 別 pin を使う。
# num2words は SmolVLM プロセッサが実行時に要求する (lerobot の smolvla extra に含まれない)。
uv pip install \
  "sapien==3.0.0b1" "mplib==0.2.1" "transforms3d==0.4.2" "trimesh==4.4.3" \
  "open3d==0.19.0" "imageio==2.34.2" termcolor zarr pydantic h5py toppra num2words
# 注意: open3d 0.19 は numpy<2 を要求するため numpy が 1.26 系に落ちる。
# pytorch3d は入れない。RoboTwin では点群の farthest-point sampling (envs/camera/camera.py)
# にしか使われず、RGB 観測のみの用途では呼ばれない ("missing pytorch3d" と出るが無害)。

echo "==> 3. 上流パッチ"
"$PY" - <<'PYEOF'
import pathlib, re, sysconfig
sp = pathlib.Path(sysconfig.get_paths()["purelib"])

# (a) mplib==0.2.1: planner.py の壊れた `or collide` を削除
p = sp / "mplib" / "planner.py"
s = p.read_text()
if "or collide" in s:
    p.write_text(re.sub(r"\bor collide\b", "", s, count=1)); print("  [ok] mplib planner.py")
else:
    print("  [skip] mplib (適用済み)")

# (b) sapien==3.0.0b1: urdf_loader の encoding と .srdf 拡張子
#     lerobot Dockerfile の置換パターン ('with open(srdf_path) as f:') は
#     このバージョンのコードと一致せず encoding 修正が適用されない。
#     RoboTwin 公式 script/_install.sh の sed と同じ正規表現を使う。
p = sp / "sapien" / "wrapper" / "urdf_loader.py"
s = p.read_text()
s2 = s.replace('"srdf"', '".srdf"')
s2, n = re.subn(r'("r")\)( as)', r'\1, encoding="utf-8")\2', s2)
p.write_text(s2); print(f"  [ok] sapien urdf_loader.py (encoding x{n})")
PYEOF

echo "==> 4. CuRobo $CUROBO_REF (arch: $ARCH_LIST)"
if [ ! -d "$RT/envs/curobo" ]; then
  git clone --branch "$CUROBO_REF" --depth 1 https://github.com/NVlabs/curobo.git "$RT/envs/curobo"
fi
# CuRobo 0.7.8 は warp の旧 API `wp.torch.device_from_torch` を使うが、
# warp-lang >= 1.4 で warp.torch サブモジュールが廃止された (該当は world_mesh.py の 1 箇所のみ)。
# lerobot のドキュメントは warp-lang==1.3.1 へのダウングレードを指示しているが、
# 1.3.1 は Blackwell (sm_120) 未対応の可能性が高いため、こちらを書き換える。
"$PY" - <<'PYEOF'
import pathlib, os
p = pathlib.Path(os.environ["RT"]) / "envs/curobo/src/curobo/geom/sdf/world_mesh.py"
s = p.read_text()
if "wp.torch.device_from_torch(" in s:
    p.write_text(s.replace("wp.torch.device_from_torch(", "wp.device_from_torch(", 1))
    print("  [ok] curobo world_mesh.py (warp API)")
else:
    print("  [skip] curobo warp patch (適用済み)")
PYEOF
( cd "$RT/envs/curobo" \
  && PATH="$CUDA_DIR/bin:$PATH" CUDA_HOME="$CUDA_DIR" \
     TORCH_CUDA_ARCH_LIST="$ARCH_LIST" MAX_JOBS="${MAX_JOBS:-16}" \
     uv pip install -e . --no-build-isolation )

echo "==> 5. アセット (embodiments + objects, 約 5.3GB)"
# background_texture.zip (11GB) は demo_randomized (hard 設定) を使う段階で追加する。
"$PY" - <<'PYEOF'
import os, pathlib, zipfile
from huggingface_hub import hf_hub_download
assets = pathlib.Path(os.environ["RT"]) / "assets"; assets.mkdir(parents=True, exist_ok=True)
for f in ("embodiments.zip", "objects.zip"):
    if (assets / f[:-4]).exists():
        print(f"  [skip] {f}"); continue
    local = hf_hub_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
                            filename=f, local_dir=str(assets))
    with zipfile.ZipFile(local) as z: z.extractall(str(assets))
    pathlib.Path(local).unlink(); print(f"  [ok] {f}")
PYEOF

echo "==> 6. embodiment のパス設定"
( cd "$RT" && "$PY" script/update_embodiment_config_path.py >/dev/null )

cat <<MSG

セットアップ完了。

実行時の注意:
  * RoboTwin は相対パス ./assets/ を使うため、cwd を $RT にすること。
  * HF_HOME=$ROOT/.cache/huggingface を設定すること (ホーム領域の容量不足を避ける)。
  * PYTHONPATH に $RT と $ROOT/src を通すこと。
  * GPU は共有のため CUDA_VISIBLE_DEVICES を明示すること。

例:
  cd $RT && CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=$RT:$ROOT/src \\
    $PY -m grip_policy.eval_baseline --task beat_block_hammer \\
       --robotwin-root $RT --n-episodes 10
MSG
