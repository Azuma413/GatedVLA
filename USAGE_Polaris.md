# CARLAデータ収集からAlpamayo SFTまで

この手順では、LEADでCARLAデータをAlpamayo/Physical AI AV互換の4カメラ構成で収集し、PAI風ローカル形式へexportしてからAlpamayo-R1をSFTします。CoCラベルは生成しません。

## 重要: Polarisで収集を動かすための必須条件（2026-06-18 検証）

Polaris計算ノード上のCARLAに別ノードから接続し、LEAD py123d形式で収集できることを `route_000391`(Town03) で確認済み（cross-node: サーバ `x3001c0s1b1n0`、クライアント `x3001c0s7b0n0`、`summary {"status":"ok"}`、4カメラ+LiDAR+ego_state+sync+navigation_command を生成）。

そこに至るまでに、以下2点が「マップ切り替え/収集が落ちる」原因だったので必ず守ること。

1. **CARLAサーバの起動フラグは最小限にする。**
   動く: `CarlaUE4 -RenderOffScreen -quality-level=Epic -carla-rpc-port=<port> -graphicsadapter=<gpu>`
   `-vulkan` `-nosound` `-unattended` `-NoSplash` `-log` `-stdout` `-FullStdOutLogOutput` の**いずれか**を付けると、複雑マップで `client.get_trafficmanager()` がクライアント側 `libcarla`(`carla::geom::Math::GetForwardVector`, TrafficManagerのInMemoryMap構築中)で **SIGSEGV** する。`-graphicsadapter` は安全（多GPU用に必要）。`collect_py123d_*.sh` は修正済み。

2. **`_Opt`（最適化）マップを使わない。**
   `Town03_Opt` 等の `_Opt` バリアントは、最小起動にしても `get_trafficmanager()` が同じ箇所でSIGSEGVする（クライアント種別・torch有無に無関係）。非`_Opt`の `Town03` 等は正常。`USE_OPTIMIZED_MAPS` と opt変換ロジック(`prepare_route_map_variant.py` 等)は削除済みで、ルートXMLの `town` をそのまま使う。

補足:
- 起動時のマップ引数（`CarlaUE4 <map>` や `/Game/Carla/Maps/<map>`）は**このパッケージでは無視**され、常に `DefaultEngine.ini` の `ServerDefaultMap`(=Town10HD_Opt) で起動する。マップ切替はクライアントの `client.load_world(<town>)` で行う（最小起動なら正常に動く）。
- `_setup_simulation` では TrafficManager を作らず、`_load_and_wait_for_world` 内で `load_world()`→`apply_settings(synchronous)`→`get_trafficmanager()` の順に**load_world後**にTMを作る（`3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py` 修正済み）。
- `containers/carla.sif` は 2026-06-18 に AdditionalMaps 入りで再ビルド済み。`Town01-07 / Town10HD / Town11-13 / Town15` を含み、**Town06/07/15 は load_world+TM+tick 動作確認済み**。再ビルドは `scripts/polaris/build_carla_am.pbs`(carla_am.sif を作り差し替え)。carla.def の要点: AdditionalMaps の正しい配布元は CARLA release bucket(`carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz`、github releases は404)、展開は `tar --no-same-owner`(fakerootでchown不可)。maps無しSIFのバックアップ: `containers/carla.sif.bak_town01-05_10hd`。
- **1ルート毎のCARLA再起動を避ける**: `scripts/tools/build_route_batches.py` で**Town別に複数ルートを1つのXMLへ結合**し、1サーバに渡す。評価器がルートをループし、同一Townなので2本目以降は load_world をスキップ（サーバ再利用・TM有効のまま）。スループットは約390ms/step(Epic+4カメラ+LiDAR≈2.5fps)なので長尺ルートは15-20分。PBS walltimeをルート長に合わせる(prodは最大24h)か、`-quality-level=Low`で高速化。

## 0. 環境の前提
Polarisでは計算ノード上でApptainer imageを作成します。

```bash
cd /eagle/RobustViT/hiratsuka/VLA4AD_Research
qsub scripts/polaris/build_apptainer.pbs
```

CARLA imageだけ作り直す場合:

```bash
qsub -v FORCE_CARLA=1 scripts/polaris/build_apptainer.pbs
```

`carla.def` は `carlasim/carla:0.9.15` をベースに `containers/carla.sif` を作ります。build時にALCF proxyを通じて `AdditionalMaps_0.9.15.tar.gz` を自動ダウンロードし、Town06・Town07・Town15などを追加します。ダウンロードに失敗しても build は続行しますが、その場合これらのマップは使えません（ビルドログの WARNING を確認してください）。AdditionalMapsのダウンロードをスキップしたい場合:

```bash
qsub -v FORCE_CARLA=1,ADDITIONAL_MAPS_SKIP=1 scripts/polaris/build_apptainer.pbs
```

build後は、短いroute収集に進む前に必ず次の `load_world` matrix を通してください。

```bash
qsub scripts/polaris/carla_load_world_matrix.pbs
tail -f logs/polaris_carla_load_world_matrix.out
```

結果は job id ごとに以下へ残ります。

```text
logs/polaris_collection/<jobid>/server_XX_<town>.log
logs/polaris_collection/<jobid>/probe_XX_<town>.log
logs/polaris_collection/<jobid>/summary.jsonl
```

デフォルトでは `Town01`, `Town01_Opt`, ..., `Town07_Opt`, `Town10HD`, `Town10HD_Opt`, `Town15` を確認します。各townで以下が通ることを acceptance とします。

```text
client.load_world("<town>")
get_world().get_map().name
RGB sensor spawn
30 tick以上の synchronous tick
```

特に `Town03` / `Town03_Opt` が `Signal 11` や timeout になる SIF は、大規模収集に使わないでください。対象townを絞る場合:

```bash
qsub -v TOWNS="Town03 Town03_Opt Town10HD_Opt Town15" scripts/polaris/carla_load_world_matrix.pbs
```

## 1. 123D形式でCARLAデータを収集

### 計算ノードでCARLAを起動する場合

CARLAサーバはlogin nodeではなく、PBSで確保した計算ノード上でApptainer経由で起動します。短い確認だけならinteractive jobを取り、その中でサーバを起動します。

```bash
qsub -I \
  -l select=1:ngpus=1 \
  -l walltime=0:30:00 \
  -q debug \
  -A RobustViT \
  -l filesystems=home:eagle
```

計算ノードに入ったら:

```bash
cd /eagle/RobustViT/hiratsuka/VLA4AD_Research

ml use /soft/modulefiles
ml spack-pe-base
ml apptainer

source scripts/polaris/carla_apptainer_common.sh
prepare_host_graphics_overlay

mkdir -p logs/carla_saved_manual
export CARLA_PORT=2000

apptainer exec --nv --writable-tmpfs \
  --contain \
  --bind "$(pwd)/logs/carla_saved_manual:/opt/carla/CarlaUE4/Saved" \
  --bind "$(pwd)/logs/carla_saved_manual:/home/carla/CarlaUE4/Saved" \
  $(carla_graphics_bind_args) \
  $(carla_graphics_env_args) \
  --env CARLA_RPC_PORT="${CARLA_PORT}" \
  --env CARLA_PORT="${CARLA_PORT}" \
  --env CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  containers/carla.sif \
  /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
    /Game/Carla/Maps/Town03 \
    -vulkan \
    -RenderOffScreen \
    -nosound \
    -quality-level=Low \
    -carla-rpc-port="${CARLA_PORT}" \
    -graphicsadapter=0 \
    -unattended \
    -NoSplash \
    -log \
    -stdout \
    -FullStdOutLogOutput
```

`/Game/Carla/Maps/Town03` は下の短い確認で使う `data/data_routes/lead/noScenarios/route_000391.xml` のtownです。他のrouteを使う場合は、そのXMLの `town` に合わせて `/Game/Carla/Maps/<town>` を指定します。サーバに別ノードから接続する場合は、計算ノード上で `hostname -f` を確認し、クライアント側でそのhost名と `CARLA_PORT` を指定します。

注意: 起動引数でマップを渡しても無視され、初期worldは `Carla/Maps/Town10HD_Opt`（DefaultEngine.iniのServerDefaultMap）で立ち上がります。route実行時はleaderboard evaluator側の `client.load_world(<town>)` でroute XMLのtownへ切り替わります（最小起動なら正常）。

### 短いルートを確認する場合（1ルート）

CARLAサーバをnode 0、データ収集プロセスをnode 1で起動する2ノードPBS jobを使います。`USAGE.md` のローカル確認と同じ `route_000391.xml` を使い、クライアント側では `lead-python -m lead --expert --py123d` を実行します。

```bash
qsub scripts/polaris/collect_py123d_short_route.pbs
tail -f logs/polaris_collect_py123d_short_route.out
```

デフォルトでは以下を使います。route XML の `town`（例 `Town03`）をそのまま使い、クライアントが `client.load_world("Town03")` でマップを切り替えます。`_Opt` 変換は行いません（冒頭の必須条件を参照）。

```bash
ROUTE=data/data_routes/lead/noScenarios/route_000391.xml
OUTPUT_ROOT=data
CARLA_SERVER_MODE=host-vulkan
BASE_PORT=20000
SCENARIO_ID=0
```

別のrouteや出力先を使う場合:

```bash
qsub -v ROUTE=data/data_routes/lead/noScenarios/route_000391.xml,OUTPUT_ROOT=data \
  scripts/polaris/collect_py123d_short_route.pbs
```

このjobの内部では、server rankがCARLAのhost/portを `logs/polaris_collection/$PBS_JOBID/server_ready/scenario_00.env` に書き、client rankがそれを読んで次と同等のコマンドを実行します。

```bash
export LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research
export PY123D_DATA_ROOT=/workspace/VLA4AD_Research/data/carla_alpamayo_4c_py123d
export LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false"

lead-python -u -m lead \
  --expert \
  --py123d \
  --host "${SERVER_HOST}" \
  --port "${CARLA_PORT}" \
  --traffic-manager-port "${TM_PORT}" \
  --timeout 600 \
  --output-dir /workspace/VLA4AD_Research/data/outputs \
  --routes /workspace/VLA4AD_Research/data/data_routes/lead/noScenarios/route_000391.xml
```

成功時は `data/carla_alpamayo_4c_py123d/logs/carla_train/.../` 以下に、4カメラとLiDARなどのArrow/Parquetファイルが生成されます。

```text
camera.ftcam_l0.arrow
camera.ftcam_f0.arrow
camera.ftcam_r0.arrow
camera.ftcam_tele_f0.arrow
lidar.lidar_top.arrow
ego_state_se3.arrow
sync.arrow
navigation_command.parquet
```

ジョブごとの詳細ログは以下に残ります。

```bash
sed -n '1,240p' logs/polaris_collection/<PBS_JOBID>/server_00.log
sed -n '1,240p' logs/polaris_collection/<PBS_JOBID>/client_00.log
```

標準ログ配置は次です。

```text
logs/polaris_collection/<jobid>/server_00.log
logs/polaris_collection/<jobid>/client_00.log
logs/polaris_collection/<jobid>/server_ready/scenario_00.env
logs/polaris_collection/<jobid>/summary.jsonl
logs/polaris_collection/<jobid>/failed_routes.txt
data/carla_alpamayo_4c_py123d/...
```

### 並列収集（複数ルートを同時実行）

`collect_py123d_parallel.pbs` は N 個のserver+clientペアを別々のノードペアで同時に走らせます。各ペアが独立した1ルートを処理します。

**ルートリストファイルを準備する:**

```bash
# 例: lead/noScenarios の Town15 ルートを最初の8本だけリストアップ
grep -rl 'town="Town15"' data/data_routes/lead/noScenarios/ | sort | head -8 \
  > logs/routes_town15_8.txt
```

**4ペア（8ノード）で実行:**

```bash
qsub -v ROUTES_FILE=logs/routes_town15_8.txt \
  scripts/polaris/collect_py123d_parallel.pbs
tail -f logs/polaris_collect_py123d_parallel.out
```

**8ペア（16ノード）に増やす場合:**

```bash
qsub -l select=16:ngpus=4 \
  -v N_PAIRS=8,ROUTES_FILE=logs/routes_town15_8.txt \
  scripts/polaris/collect_py123d_parallel.pbs
```

内部では `mpiexec -n $((N_PAIRS * 2)) -ppn 1` を使い、rank 2k がシナリオ k のサーバ、rank 2k+1 がクライアントになります。各シナリオのログは以下に残ります。

```text
logs/polaris_collection/<jobid>/server_00.log   # シナリオ0サーバ
logs/polaris_collection/<jobid>/client_00.log   # シナリオ0クライアント
logs/polaris_collection/<jobid>/server_01.log   # シナリオ1サーバ
logs/polaris_collection/<jobid>/client_01.log   ...
logs/polaris_collection/<jobid>/routes_manifest.txt  # 各シナリオのルートパス
logs/polaris_collection/<jobid>/summary.jsonl        # 成功/失敗サマリ
logs/polaris_collection/<jobid>/failed_routes.txt    # 失敗ルート一覧
```

portは scenario id から固定でずらします。

```text
rpc_port = BASE_PORT + scenario_id * 10
tm_port  = rpc_port + 500
```

同一ノードで複数CARLAを走らせる拡張では、CARLA streaming port も scenario id から固定でずらしてください。CARLAが `-carla-streaming-port` を受ける構成にした場合は、`streaming_port = rpc_port + 1` のように予約し、scenario間で10 port幅を共有しない設計にします。

route XML の `town` はそのまま使います（`_Opt` 変換なし）。サーバはデフォルトマップで起動し、クライアントが `client.load_world(<town>)` で切り替えます。

```text
route town="Town03"  -> client.load_world("Town03")  -> /Game/Carla/Maps/Town03
route town="Town04"  -> client.load_world("Town04")  -> /Game/Carla/Maps/Town04
route town="Town06"  -> client.load_world("Town06")   (要: AdditionalMaps入り carla.sif)
```

`CARLA_SERVER_MODE` は `scripts/polaris/carla_apptainer_common.sh` を通した `host-vulkan` を標準にします。

確認結果:

- **2026-06-18**: short_route(`route_000391`, Town03) で cross-node 収集成功（`summary {"status":"ok"}`、4カメラ+LiDAR+ego_state+sync+navigation_command を生成、サーバとクライアントは別ノード）。決め手は「CARLA最小起動」+「`_Opt`不使用」（冒頭の必須条件）。
- 旧2026-06-11の所見（`Town03` が `Signal 11`、`Town03_Opt` は成功）は**旧起動フラグ下の現象**で、最小起動では逆転（`Town03` 成功、`Town03_Opt` がTMでSIGSEGV）。`_Opt` は使わないこと。
- 現 `containers/carla.sif`(6/10ビルド) は `Town01-05 / Town10HD` のみ。`Town06/07/15` を使うには `carla.def`(AdditionalMaps自動DL) で `carla.sif` を再ビルドし、matrix で各townの `load_world`+TM を確認してから収集に使うこと。

### スケール前の段階テスト

大規模化する前に、本番と同じ sensor / Py123D 設定で短いrouteを通します。smoke用の軽量設定ではなく、`--py123d`、4 camera Arrow、LiDAR Arrow、ego state、sync、navigation commandが出ることを確認します。

```text
1 server + 1 client  (collect_py123d_short_route.pbs)
2 pairs (4 nodes)    (collect_py123d_parallel.pbs N_PAIRS=2)
4 pairs (8 nodes)    (collect_py123d_parallel.pbs N_PAIRS=4, デフォルト)
8 pairs (16 nodes)   (collect_py123d_parallel.pbs N_PAIRS=8)
```

各段階で `summary.jsonl` を集計し、失敗率、平均route時間、GPU memory、I/O throughputを記録します。route単位の失敗は `signal_11`, `timeout`, `sensor_no_frame`, `client_exit_<code>` に分類し、成功済みrouteはresume時にskipしてください。

## 2. 123DログをAlpamayo用PAI風形式へ変換

`scripts/export_py123d_to_pai.py` を `lead.sif` の中で実行します。GPUは不要で、login nodeでも動きますが、Polarisのlogin nodeはプロセス数(RLIMIT_NPROC)が厳しく、numpy/OpenBLASやOpenCVがスレッドを大量に作ろうとして失敗します。**必ず `OPENBLAS_NUM_THREADS=1` などスレッド数を1に絞って**実行してください（OpenCVの `WorkerThread ... res = 11` 警告は出ますが致命的ではありません）。

```bash
cd /eagle/RobustViT/hiratsuka/VLA4AD_Research
ml use /soft/modulefiles; ml spack-pe-base; ml apptainer

apptainer exec \
  --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
  -B "$(pwd):/workspace/VLA4AD_Research" \
  containers/lead.sif \
  lead-python scripts/export_py123d_to_pai.py \
    --logs-root data/carla_alpamayo_4c_py123d \
    --output-dir data/carla_alpamayo_pai \
    --split train \
    --overwrite
```

出力には `clip_index.parquet`, `features.csv`, `metadata/feature_presence.parquet`, `camera/<4カメラ>/`, `labels/egomotion`, `labels/navigation_command` が含まれます。4つのAlpamayoカメラが揃っていないログはエラーになります。

確認結果(2026-06-22): 上の§1で収集した6 clip(`data/carla_alpamayo_4c_py123d/logs/carla_train/...`)を `lead.sif` でexportし、`data/carla_alpamayo_pai/`(`Done. Exported 6 clips.`、全clipが `chunk=0 / split=train`)を生成。次の§4 SFTはこのchunk 0をそのまま使います。

## 3. Alpamayo-R1のcheckpointを用意

gatedモデルなので、HuggingFaceで `nvidia/Alpamayo-R1-10B` のアクセス権を取得しておきます。Polarisではトークンを `.env`(`HF_TOKEN=...`、リポジトリ直下、git管理外)に置き、ALCF proxy経由で `lead.sif` 同梱の `hf` CLIでダウンロードします（約21GB / safetensors 5 shard）。

```bash
cd /eagle/RobustViT/hiratsuka/VLA4AD_Research
ml use /soft/modulefiles; ml spack-pe-base; ml apptainer
HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | cut -d= -f2-)

apptainer exec \
  --env HTTP_PROXY=http://proxy.alcf.anl.gov:3128 --env HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 \
  --env http_proxy=http://proxy.alcf.anl.gov:3128 --env https_proxy=http://proxy.alcf.anl.gov:3128 \
  --env HF_TOKEN="${HF_TOKEN}" \
  -B "$(pwd):/workspace/VLA4AD_Research" \
  containers/lead.sif \
  hf download nvidia/Alpamayo-R1-10B \
    --local-dir /workspace/VLA4AD_Research/outputs/checkpoints/Alpamayo-R1-10B
```

加えて、SFT/評価時にVLMの **processor/tokenizer** として `Qwen/Qwen3-VL-8B-Instruct` が必要です（重みではなく設定ファイルのみ。`model.checkpoint_path` の重みとは別物）。実行ノードがオフラインでも動くよう、これも事前にリポジトリ内のHFキャッシュへ落としておきます。

```bash
mkdir -p outputs/hf_home
apptainer exec \
  --env HTTP_PROXY=http://proxy.alcf.anl.gov:3128 --env HTTPS_PROXY=http://proxy.alcf.anl.gov:3128 \
  --env http_proxy=http://proxy.alcf.anl.gov:3128 --env https_proxy=http://proxy.alcf.anl.gov:3128 \
  --env HF_TOKEN="${HF_TOKEN}" --env HF_HOME=/workspace/VLA4AD_Research/outputs/hf_home \
  -B "$(pwd):/workspace/VLA4AD_Research" \
  containers/lead.sif \
  hf download Qwen/Qwen3-VL-8B-Instruct --exclude "*.safetensors" "*.bin" "*.pth" "*.gguf"
```

確認結果(2026-06-22): `outputs/checkpoints/Alpamayo-R1-10B`(21GB, `model-0000{1..5}-of-00005.safetensors` + `config.json` + `model.safetensors.index.json`)と `outputs/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct`(processor/tokenizer)を取得済み。

## 4. AlpamayoをSFT

AlpamayoのSFTはPython 3.12 + torch 2.8 + flash-attn + deepspeed + vllm + cosmos-rl が必要で、`lead.sif`(Python 3.10)とは別環境です。Polarisでは専用のApptainer image `containers/alpamayo.sif` を `alpamayo.def` からビルドして使います（`lead.def` と同じく、依存だけ `uv sync --no-install-project` で入れ、`alpamayo_r1` / `finetune` のソースは実行時に `PYTHONPATH`=bind mountで解決）。

```bash
cd /eagle/RobustViT/hiratsuka/VLA4AD_Research
qsub scripts/polaris/build_alpamayo.pbs
tail -f logs/polaris_build_alpamayo.out
```

`alpamayo.def` はALCF proxyを使い、A100(sm_80)向けに `TORCH_CUDA_ARCH_LIST=8.0` を固定してflash-attnを入れます（prebuilt wheelが取れれば数分で完了）。ビルド後 `%test` で `torch 2.8.0+cu128 / flash_attn 2.8.3 / transformers 4.57.1 / deepspeed 0.18.2` が確認できます。

### Polarisハードウェアに合わせた必須調整（4× A100 40GB）

FINETUNE_SFTは8× H100 80GBで検証されています。Polaris計算ノードは1ノード4× A100 **40GB** なので、本番configそのままでは載りません。`scripts/polaris/sft_alpamayo_short.pbs` は次のPolaris調整込みで本番の `train_hf`(`sft_stage1`/`sft_stage2`)を回します（**smoke用の軽量モデル/データではなく**、本物の10B checkpoint・§2のPAIデータ・本番configを使い、ステップ数だけ1に絞っています）。

1. **Stage 1 は ZeRO-2 + optimizer CPU offload**(`finetune/sft/configs/deepspeed/zero2_offload.json`)。本番は ZeRO-2 だがそのままだとAdam状態で40GBを超える。**ZeRO-3 は使えない**: AlpamayoのVLM重み読み込みが素の `load_state_dict` で、ZeRO-3 の遅延初期化(size-0 placeholder)と衝突して `size mismatch ... current model is torch.Size([0])` で落ちる。offloadでoptimizer/masterをCPUへ逃がせばZeRO-2のまま載る。
2. **Stage 2 も ZeRO-2 + offload**。本番の `deepspeed: null`(DDP)はfrozen 10B VLMを各GPUに複製してOOMする。offloadでexpertのoptimizerをCPUへ。
3. **`action_in_proj.py` の dtype 修正**(`3rd_party/alpamayo/src/alpamayo_r1/models/action_in_proj.py`)。action expertはFourier特徴をfp32で作るが、DeepSpeed(autocastなし)ではbf16 Linearにfp32が入り `mat1 and mat2 must have the same dtype` で落ちる。encoder直前で `x = x.to(next(self.encoder.parameters()).dtype)` とモデルのcompute dtypeへ揃える（DDP+autocast時は暗黙に行われる変換で、挙動は不変）。
4. `gradient_checkpointing` は `TrainableAlpamayoR1` が非対応なので使わない。fragmentation対策に `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付与。

Qwen processorはオフライン参照（`HF_HOME=outputs/hf_home`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`）。W&Bは `report_to=none` で無効です。

### SFTを短く確認する場合（Stage 1 → Stage 2、各1 step）

```bash
qsub scripts/polaris/sft_alpamayo_short.pbs
tail -f logs/polaris_sft_alpamayo_short.out
```

既定値:

```bash
STAGE=all            # all | 1 | 2
NPROC=4              # 1ノード4GPU
PAI_DIR=.../data/carla_alpamayo_pai
CKPT=.../outputs/checkpoints/Alpamayo-R1-10B
OUT_ROOT=.../outputs/alpamayo_sft_polaris
CHUNK_IDS=0-1        # §2で出した chunk 0
MAX_STEPS=1
```

Stage 1だけ / Stage 2だけを回すには `qsub -v STAGE=1 ...` / `qsub -v STAGE=2 ...`。Stage 2はStage 1の `stage1/checkpoint-${MAX_STEPS}` を読みます。内部のコマンドは本番と同じ `alpamayo-torchrun --nproc_per_node 4 -m finetune.sft.train_hf --config-name sft_stage{1,2} ...`(+ 上記Polaris調整 + `+trainer.max_steps=1 trainer.save_steps=1`)です。

成功すると次が生成されます。

```text
outputs/alpamayo_sft_polaris/stage1/checkpoint-1/   (model-0000x-of-0000y.safetensors, trainer_state.json, ...)
outputs/alpamayo_sft_polaris/stage2/checkpoint-1/   (同上)
```

確認結果(2026-06-22): 4× A100で Stage 1(`epoch 1.0`)→ Stage 2(`train_loss 2.484`, `train_runtime 95.7s`)を各1 stepで通し、両 `checkpoint-1` を保存。ログ末尾は `=== SFT verification done ===`。

## 5. AlpamayoをBench2Drive評価し入力動画を保存

closed-loop評価(`lead -m lead --alpamayo --bench2drive`)は **CARLA(`carla.sif`) と Alpamayo推論を1プロセスで**動かします。Alpamayo推論コード(`lead/inference/alpamayo_inference.py`)は `3rd_party/alpamayo` を `sys.path` に足して `alpamayo_r1` / `finetune.sft.models` をimportするので、CARLA python APIを持つ `lead.sif`(Python 3.10)の中で動きます（§4の学習用 `alpamayo.sif` ではなく `lead.sif`）。データ収集と同じ「node 0=CARLAサーバ / node 1=クライアント」の2ノード構成を使います。

`scripts/polaris/eval_alpamayo_b2d_short.pbs`(= `eval_alpamayo_b2d_short.sh`)が一式を行います。USAGE.md §5と同じく `LEAD_CLOSED_LOOP_CONFIG='produce_input_video=true produce_frame_frequency=1'`、`ALPAMAYO_INFERENCE_INTERVAL=10` で、4カメラを横連結した入力動画 `{route_id}_input.mp4` を出します。

Polaris固有の注意:

- **`lead.sif` には flash_attn が入っていない**(torch 2.7.1)。SFT checkpointの `config.json` は `attn_implementation=flash_attention_2` なので、スクリプトが評価前に **checkpointの `config.json` を `sdpa` へ書き換え**ます(`ATTN_IMPL=sdpa`)。本番の推論コードは無改造のまま動きます。
- processorはオフライン参照(`HF_HOME=outputs/hf_home`, `HF_HUB_OFFLINE=1`)。
- 入力動画の `.mp4` は **routeが正常終了したとき(agentのcleanup)に finalize**されます。途中で `qdel` すると `moov atom not found` の壊れたmp4になります。短い確認では、routeを数秒で完走させて確実にfinalizeさせるため、長いTown12 route `23687.xml` の先頭数waypointだけにした `23687_short.xml` を使います（評価経路・agent・コードは本番のまま、1エピソードを短くしただけ）。

```bash
qsub -v ROUTE=data/benchmark_routes/bench2drive/23687_short.xml,OUTPUT_ROOT=outputs/alpamayo_b2d_video_short \
  scripts/polaris/eval_alpamayo_b2d_short.pbs
tail -f logs/polaris_eval_alpamayo_b2d_short.out
```

既定の `CHECKPOINT=outputs/alpamayo_sft_polaris/stage2/checkpoint-1` を評価します。本番フルrouteを流すなら `ROUTE=data/benchmark_routes/bench2drive/23687.xml` とし、walltimeを延ばしてroute完走を待ちます（途中killはmp4を壊すので避ける）。

出力例:

```text
outputs/alpamayo_b2d_video_short/23687_input.mp4
outputs/alpamayo_b2d_video_short/metric_info.json
outputs/alpamayo_b2d_video_short/infractions.json
outputs/alpamayo_b2d_video_short/checkpoint_endpoint.json
```

確認結果(2026-06-22): CARLA(Town12) + `lead.sif` のAlpamayo agent(sdpaでcheckpointロード、`Agent setup time≈52s`)でcross-node closed-loopが起動し、agentが毎frameでtrajectoryを出して `23687_short.xml` を完走(`Alpamayo Bench2Drive eval finished`)。finalize済みの4カメラ入力動画 `23687_input.mp4`(mpeg4, 3200×600 = 800px×4カメラ横連結, 1251 frame, 62.5s)と `metric_info.json` / `infractions.json`(`RouteCompletionTest` 等) / `checkpoint_endpoint.json` を生成。決め手は「`lead.sif` 内でsdpaロード」+「scenario付きの短縮routeで完走させてmp4をfinalize」。
