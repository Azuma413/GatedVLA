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

未確認です。上のsmokeでPy123D Arrowは生成済みですが、PAI風形式へのexportはまだ通していません。

## 3. Alpamayo-R1のcheckpointを用意

未確認です。

## 4. AlpamayoをSFT

未確認です。

### SFTを短く確認する場合

未確認です。

## 5. AlpamayoをBench2Drive評価し入力動画を保存

未確認です。
