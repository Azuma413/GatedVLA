# オンプレサーバでの残りデータ収集手順

オンプレGPUサーバで、残り **567ルート** を高速化済みの PyAV-light 設定で収集する手順です。

現在の残りルートは [missing_manifest/remaining_routes.txt](missing_manifest/remaining_routes.txt) に凍結済みです。このリストは、既存収集分、今回の成功分、失敗分、途中停止分を整理した後の「まだ収集が必要なルート」です。

この手順では、PyAV/CoC/学習に必要な4カメラ、egomotion、navigation_commandを残し、不要なLiDAR、box detection書き出し、traffic light書き出し、perturbated sensorsを無効化します。

---

## 前提条件

| 項目 | 内容 |
|------|------|
| GPU | NVIDIA RTX A6000 / RTX 6000 Ada など、CARLAを動かせるGPU |
| CARLA | 0.9.15（バイナリ展開済み） |
| Python環境 | `uv run` で起動可能な `.venv` |
| リポジトリ | このリポジトリを `git clone` 済み |
| ルートmanifest | `missing_manifest/remaining_routes.txt` |

CARLAバイナリが標準パスにない場合は、実行時に `CARLA_BIN=/path/to/CarlaUE4-Linux-Shipping` を追加してください。

---

## 0. 環境確認

```bash
cd /path/to/VLA4AD_Research

# Python環境が壊れていないか確認
uv run python -c "import lead; print('lead OK')"

# 壊れている場合だけ同期し直す
uv sync

# CARLAバイナリ確認
ls 3rd_party/CARLA_0915/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping

# GPU確認
nvidia-smi

# 残りroute数確認
wc -l missing_manifest/remaining_routes.txt
```

期待値:

```text
567 missing_manifest/remaining_routes.txt
```

---

## 1. ドライラン

実行前に、batch生成と見積もりだけ確認します。

```bash
CHUNK=1 \
MISSING_LIST=missing_manifest/remaining_routes.txt \
uv run bash scripts/local/collect_missing_local.sh
```

`LAUNCH=1` を付けない限り、CARLAは起動しません。

---

## 2. 収集実行

### 3 GPUを使う場合

```bash
GPU_IDS=0,1,2 N_WORKERS=3 N_GPUS=3 CHUNK=1 MAX_RETRIES=1 CLIENT_TIMEOUT=21600 \
EXTRA_LEAD_EXPERT_CONFIG="py123d_rgb_only_smoke=true perturbate_sensors=false py123d_disable_lidar=true py123d_skip_box_detections=true py123d_skip_traffic_lights=true" \
MISSING_LIST=missing_manifest/remaining_routes.txt \
LAUNCH=1 uv run bash scripts/local/collect_missing_local.sh
```

### GPU 1,2 だけを使う場合

```bash
GPU_IDS=1,2 N_WORKERS=2 N_GPUS=2 CHUNK=1 MAX_RETRIES=1 CLIENT_TIMEOUT=21600 \
EXTRA_LEAD_EXPERT_CONFIG="py123d_rgb_only_smoke=true perturbate_sensors=false py123d_disable_lidar=true py123d_skip_box_detections=true py123d_skip_traffic_lights=true" \
MISSING_LIST=missing_manifest/remaining_routes.txt \
LAUNCH=1 uv run bash scripts/local/collect_missing_local.sh
```

### GPU 1枚だけを使う場合

```bash
GPU_IDS=0 N_WORKERS=1 N_GPUS=1 CHUNK=1 MAX_RETRIES=1 CLIENT_TIMEOUT=21600 \
EXTRA_LEAD_EXPERT_CONFIG="py123d_rgb_only_smoke=true perturbate_sensors=false py123d_disable_lidar=true py123d_skip_box_detections=true py123d_skip_traffic_lights=true" \
MISSING_LIST=missing_manifest/remaining_routes.txt \
LAUNCH=1 uv run bash scripts/local/collect_missing_local.sh
```

設定の意味:

- `CHUNK=1`: 1 CARLAセッション1 route。失敗切り分けと再収集を簡単にする。
- `MAX_RETRIES=1`: その場で粘らず、失敗routeを後で再queueする。
- `CLIENT_TIMEOUT=21600`: 重いTown12/Town13/Town15 route用に6時間まで許容する。
- `py123d_rgb_only_smoke=true`: 4カメラ中心の軽量Py123D収集にする。
- `perturbate_sensors=false`: perturbated cameraを収集しない。
- `py123d_disable_lidar=true`: LiDARを収集しない。
- `py123d_skip_box_detections=true`: Py123D box detectionを書き出さない。
- `py123d_skip_traffic_lights=true`: Py123D traffic light detectionを書き出さない。

この設定でも、PyAV/PAI変換に必要な `sync.arrow`、4カメラ、`egomotion`、`navigation_command` は残ります。

---

## 3. ログ確認

実行すると `logs/local_collection/<timestamp>/` が作成されます。

```bash
WORK=logs/local_collection/<timestamp>

# 全体summary
tail -f ${WORK}/summary.jsonl

# workerログ
tail -f ${WORK}/worker_00.log
tail -f ${WORK}/worker_01.log

# CARLA / clientログ
tail -f ${WORK}/server_00.log
tail -f ${WORK}/client_00.log

# GPU使用状況
nvidia-smi
```

成功/失敗の集計:

```bash
uv run python - <<'PYCODE'
import json
from collections import Counter
from pathlib import Path

work = Path("logs/local_collection/<timestamp>")
c = Counter()
reasons = Counter()
for line in (work / "summary.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    c[row.get("status")] += 1
    if row.get("status") != "ok":
        reasons[row.get("reason", "")] += 1
print(dict(c))
print(dict(reasons))
PYCODE
```

Py123D実ログ数の確認:

```bash
find data/carla_alpamayo_4c_py123d/logs/carla_train -name sync.arrow | wc -l
```

---

## 4. 中断

`screen` で起動している場合は、対象sessionとCARLA/leaderboardプロセスを停止してください。

```bash
screen -ls
ps -eo pid,ppid,stat,etime,cmd | grep -E 'collect_|CarlaUE4|leaderboard|python -m lead|timeout 21600' | grep -v grep
```

停止する場合は、対象PIDを確認してから `kill -9` します。

```bash
kill -9 <pid> <pid> ...
screen -wipe
```

停止後の確認:

```bash
pgrep -af CarlaUE4
pgrep -af leaderboard_evaluator_local
pgrep -af collect_missing_local
screen -ls
```

---

## 5. 再開

同じ `WORK` を指定して再実行すると、`checkpoint_endpoint.json` と `--resume` により成功済みbatchはスキップされます。

```bash
WORK=logs/local_collection/<timestamp> \
GPU_IDS=0,1,2 N_WORKERS=3 N_GPUS=3 CHUNK=1 MAX_RETRIES=1 CLIENT_TIMEOUT=21600 \
EXTRA_LEAD_EXPERT_CONFIG="py123d_rgb_only_smoke=true perturbate_sensors=false py123d_disable_lidar=true py123d_skip_box_detections=true py123d_skip_traffic_lights=true" \
MISSING_LIST=missing_manifest/remaining_routes.txt \
LAUNCH=1 uv run bash scripts/local/collect_missing_local.sh
```

別サーバへ移す場合は、前サーバの `summary.jsonl` を基準に成功routeを除外した新しい `remaining_routes.txt` を作ってから実行してください。単に同じmanifestを別サーバで再実行すると、前サーバで成功したrouteも再実行対象になります。

---

## 6. 完了判定

`summary.jsonl` に `ok` 以外が残っていなければ、そのmanifest分は完了です。

```bash
uv run python - <<'PYCODE'
import json
from collections import Counter
from pathlib import Path

work = Path("logs/local_collection/<timestamp>")
c = Counter()
for line in (work / "summary.jsonl").read_text().splitlines():
    if line.strip():
        c[json.loads(line).get("status")] += 1
print(dict(c))
PYCODE
```

失敗がある場合:

```bash
cat ${WORK}/failed_batches.txt 2>/dev/null
```

`traffic_manager_bind` は再実行で通ることが多いです。`Agent timed out` はroute固有に重いか、agentが詰まっている可能性があります。完了までには、失敗routeだけを再manifest化して回収します。

---

## 7. PyAV / CoC 確認

今回のPyAV-light設定で収集した成功routeは、実データから以下を確認済みです。

- PyAV/PAI変換可能
- 4カメラあり
- `navigation_command.parquet` あり
- `egomotion` 生成可能
- `features.csv`, `clip_index.parquet`, `feature_presence.parquet` 生成可能
- CoC autolabelerが使う `camera_front_wide_120fov` zipを生成可能

変換例:

```bash
uv run python scripts/export_py123d_to_pai.py \
  --logs-root data/carla_alpamayo_4c_py123d/logs/carla_train/<clip_dir> \
  --output-dir /tmp/pyav_verify_one_clip \
  --overwrite \
  --max-logs 1
```

CoCについては、形式としては対応しています。ただしCoCのmeta-action/keyframe pipelineは短いclipを落とすため、目安として17.5秒未満のclipはCoCラベリング対象から外れる可能性があります。

---

## スクリプト構成

| ファイル | 役割 |
|--------|------|
| [scripts/local/collect_missing_local.sh](scripts/local/collect_missing_local.sh) | manifest検証、batch生成、並列収集起動 |
| [scripts/local/collect_local_parallel.sh](scripts/local/collect_local_parallel.sh) | N_WORKERS並列のCARLA+client実行 |
| [scripts/tools/validate_route_manifest.py](scripts/tools/validate_route_manifest.py) | 不正routeの事前除外 |
| [scripts/tools/build_route_batches.py](scripts/tools/build_route_batches.py) | route XMLをtown別batch XMLに変換 |
| [scripts/export_py123d_to_pai.py](scripts/export_py123d_to_pai.py) | Py123D ArrowログをPAI/PyAV形式へ変換 |

### ログ構成

```text
logs/local_collection/<timestamp>/
  batch_manifest.txt
  routes_filtered.txt
  routes_filtered.raw.txt
  invalid_routes_lt2_waypoints.txt
  route_validation_summary.txt
  summary.jsonl
  failed_batches.txt
  worker_00.log
  server_00.log
  client_00.log
  carla_saved_00/
  batches/
```
