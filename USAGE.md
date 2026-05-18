# CARLAデータ収集からAlpamayo SFTまで

この手順では、LEADでCARLAデータをAlpamayo/Physical AI AV互換の4カメラ構成で収集し、PAI風ローカル形式へexportしてからAlpamayo-R1をSFTします。CoCラベルは生成しません。

## 0. 環境の前提

LEADのデータ収集とPAI exportはリポジトリ直下のPython 3.10環境を使います。

```bash
cd /path/to/VLA4AD_Research
export LEAD_PROJECT_ROOT="$(pwd)"
source .venv/bin/activate
```

AlpamayoのSFTは `3rd_party/alpamayo/pyproject.toml` が要求するPython 3.12環境を使います。LEAD用の `.venv` ではなく、Alpamayoディレクトリ側で別途 `uv sync` してください。

## 1. 123D形式でCARLAデータを収集

### ローカルでCARLAを起動する場合

別ターミナルでCARLA 0.9.15を起動します。

```bash
cd /path/to/VLA4AD_Research
./3rd_party/CARLA_0915/CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -carla-rpc-port=2000
```

接続確認:

```bash
python3 - <<'PY'
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(5.0)
print(client.get_world().get_map().name)
PY
```

### SLURMで並列収集する場合

```bash
python3 slurm/data_collection/collect_data.py \
  --py123d \
  --export-pai \
  --root_folder data/ \
  --route_folder data/data_routes
```

`--py123d` はAlpamayo用4カメラ収集を使います。

- `camera_cross_left_120fov`
- `camera_front_wide_120fov`
- `camera_cross_right_120fov`
- `camera_front_tele_30fov`

123Dログは `data/carla_alpamayo_4c_py123d/` に保存され、`--export-pai` により `data/carla_alpamayo_pai/` が作られます。

### ローカルで短いルートを確認する場合

CARLAを起動してから:

```bash
export LEAD_PROJECT_ROOT="$(pwd)"
export PY123D_DATA_ROOT="${LEAD_PROJECT_ROOT}/data/carla_alpamayo_4c_py123d"
export LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false"

python3 -m lead \
  --expert \
  --py123d \
  --routes data/data_routes/lead/noScenarios/route_000391.xml
```

`short_route.xml` はCARLA接続とPy123D保存の確認には使えますが、clip長が短いためAlpamayoのデフォルト設定ではPAI export時またはloader側のsafe marginで除外されます。

## 2. 123DログをAlpamayo用PAI風形式へ変換

SLURMで `--export-pai` を使わなかった場合は手動で実行します。

```bash
python3 scripts/export_py123d_to_pai.py \
  --logs-root data/carla_alpamayo_4c_py123d \
  --output-dir data/carla_alpamayo_pai \
  --split train \
  --overwrite
```

出力には `clip_index.parquet`, `features.csv`, `metadata/feature_presence.parquet`, camera zip, egomotion zip が含まれます。4つのAlpamayoカメラが揃っていないログはエラーになります。

## 3. Alpamayo-R1のcheckpointを用意

```bash
huggingface-cli login
huggingface-cli download nvidia/Alpamayo-R1-10B \
  --local-dir outputs/checkpoints/Alpamayo-R1-10B
```

## 4. AlpamayoをSFT

Alpamayoディレクトリへ移動します。

```bash
cd 3rd_party/alpamayo
uv sync
```

Stage 1: VLM本体のSFT。画像・履歴軌道・プロンプトから未来軌道を離散trajectory tokenとして出力するよう学習。

```bash
uv run torchrun --nproc_per_node 8 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage1 \
  data.train_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.val_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.train_dataset.chunk_ids=0-99 \
  data.val_dataset.chunk_ids=99-100 \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false \
  model.checkpoint_path=../../outputs/checkpoints/Alpamayo-R1-10B \
  paths.output_dir=../../outputs/alpamayo_sft/stage1
```

Stage 2: VLMをfreezeしてAction Expertを学習。

```bash
uv run torchrun --nproc_per_node 8 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage2 \
  data.train_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.val_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.train_dataset.chunk_ids=0-99 \
  data.val_dataset.chunk_ids=99-100 \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false \
  model.pretrained_model_name_or_path=../../outputs/checkpoints/Alpamayo-R1-10B \
  model.stage1_vlm_checkpoint_path=../../outputs/alpamayo_sft/stage1/checkpoint-XXXX \
  paths.output_dir=../../outputs/alpamayo_sft/stage2
```

`checkpoint-XXXX` はStage 1で保存されたcheckpointディレクトリに置き換えてください。GPU数やchunk範囲は収集量と環境に合わせて調整します。

### ローカルでSFTを短く確認する場合

データとcheckpointが揃っている状態で、まず1 stepだけ通すと設定とデータローダの疎通を確認できます。

```bash
uv run torchrun --nproc_per_node 3 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage1 \
  data.train_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.val_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.train_dataset.chunk_ids=0-1 \
  data.val_dataset.chunk_ids=0-1 \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false \
  model.checkpoint_path=../../outputs/checkpoints/Alpamayo-R1-10B \
  paths.output_dir=../../outputs/alpamayo_sft_smoke/stage1 \
  +trainer.max_steps=1 \
  trainer.num_train_epochs=1 \
  trainer.dataloader_num_workers=0
```

Stage 2も1 stepだけ確認する場合:

```bash
uv run torchrun --nproc_per_node 3 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs \
  --config-name sft_stage2 \
  data.train_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.val_dataset.local_dir=../../data/carla_alpamayo_pai \
  data.train_dataset.chunk_ids=0-1 \
  data.val_dataset.chunk_ids=0-1 \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false \
  model.pretrained_model_name_or_path=../../outputs/checkpoints/Alpamayo-R1-10B \
  model.stage1_vlm_checkpoint_path=../../outputs/alpamayo_sft_smoke/stage1/checkpoint-1 \
  paths.output_dir=../../outputs/alpamayo_sft_smoke/stage2 \
  +trainer.max_steps=1 \
  trainer.num_train_epochs=1 \
  trainer.dataloader_num_workers=0
```
