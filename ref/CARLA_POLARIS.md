# CARLA On Polaris

このドキュメントは、このリポジトリで CARLA 0.9.15 を ALCF Polaris 上で起動し、別ノードから CARLA API で接続し、画像や動画を保存するための実装メモです。実際に動作確認したスクリプトを基準にしています。

## 関連ファイル

- サーバ用コンテナ定義: [`carla_server.def`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/carla_server.def)
- Polaris 用サーバ起動スクリプト: [`carla_server_polaris.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/carla_server_polaris.sh)
- Python 3.7 クライアント環境構築: [`setup_py37_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/setup_py37_polaris_qsub.sh)
- 2 ノードの疎通確認ジョブ: [`run_capture_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/run_capture_polaris_qsub.sh)
- 単発画像を保存するクライアント例: [`client_capture.py`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/client_capture.py)
- 8 カメラ動画クライアント例: [`client_multicam_video.py`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/client_multicam_video.py)
- 3 ノード 8 シナリオ並列ジョブ: [`run_multiscenario_multicam_video_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/run_multiscenario_multicam_video_polaris_qsub.sh)
- 3 ノード用 MPI ドライバ: [`capture_multiscenario_multicam_mpi_driver.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/capture_multiscenario_multicam_mpi_driver.sh)

## 全体像

このリポジトリでの基本構成は次の通りです。

1. compute node 上で `apptainer exec --nv` により `carla_server.sif` を起動する
2. CARLA の `rpc_port` を外部に公開し、別ノードから TCP で到達確認する
3. Python クライアントから `carla.Client(host, port)` で接続する
4. 車両、カメラ、Traffic Manager を生成し、画像または動画を保存する

ジョブ構成は用途ごとに分かれています。

- 1 ノード: サーバとクライアントを同一ノードで動かす
- 2 ノード: node 0 でサーバ、node 1 でクライアント
- 3 ノード: node 0-1 で計 8 台のサーバ、node 2 で 8 本のクライアント

## コンテナ

CARLA サーバ用の SIF は [`carla_server.def`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/carla_server.def) から生成します。

```bash
apptainer build carla_server.sif carla_server.def
```

この定義では、コンテナ内に `carla-server` というラッパーコマンドを用意しています。主な役割は次の通りです。

- `CarlaUE4.sh` の実体を `/opt/carla` から見つける
- `-graphicsadapter` を `CARLA_GRAPHICS_ADAPTER` または `CUDA_VISIBLE_DEVICES` から決める
- Vulkan ICD が見えるときは `-vulkan` を自動付与する
- `Saved/Logs` を作成して、headless 実行しやすい環境変数を整える

## サーバ起動方法

Polaris 向けの実行エントリポイントは [`carla_server_polaris.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/carla_server_polaris.sh) です。

特徴は次の通りです。

- `apptainer exec --nv --contain --writable-tmpfs` を使う
- NVIDIA GLX / EGL / Vulkan ICD を明示 bind する
- `Saved` ディレクトリをホスト側へ bind してログを残す
- `CARLA_RPC_PORT` と `CARLA_GRAPHICS_ADAPTER` を外から上書きできる

単体起動の最小例です。

```bash
bash carla_server_polaris.sh -carla-rpc-port=20000 -graphicsadapter=0
```

環境変数で起動条件を渡す例です。

```bash
export CARLA_RPC_PORT=20020
export CARLA_GRAPHICS_ADAPTER=2
export CARLA_SAVED_BASE=$PWD/logs/carla_saved
bash carla_server_polaris.sh
```

実行ログは `CARLA_SAVED_BASE/$JOBID/...` 配下の `Saved/Logs` に保存されます。

## Python クライアント環境

CARLA 0.9.15 の Python wheel に合わせて、クライアントは Python 3.7 を使っています。構築スクリプトは [`setup_py37_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/setup_py37_polaris_qsub.sh) です。

```bash
qsub tools/carla_multinode_test/setup_py37_polaris_qsub.sh
```

作成される環境は [`py37`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/py37) です。入るパッケージは次の通りです。

- `carla-0.9.15-cp37-...whl`
- `numpy==1.21.6`
- `imageio==2.31.2`
- `imageio-ffmpeg==0.4.9`

OpenBLAS が login node で過剰にスレッドを立てないよう、ジョブ内では次を設定しています。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## ノード間通信の考え方

この実装では、ノード間通信はすべて TCP と共有ファイルで同期しています。

1. PBS が割り当てたノード一覧を `PBS_NODEFILE` から読む
2. サーバ rank は `hostname -f` と使用ポートをファイルに書く
3. クライアント rank はそのファイルを読み、`/dev/tcp/host/port` または Python の socket で接続確認する
4. 接続可能になったら CARLA API を叩く

2 ノード版では [`run_capture_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/run_capture_polaris_qsub.sh) が node 0 を server、node 1 を client に割り当てます。

3 ノード版では [`capture_multiscenario_multicam_mpi_driver.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/capture_multiscenario_multicam_mpi_driver.sh) が rank ごとに役割を分けます。

- rank 0: node 0 上で GPU 0-3 に 4 サーバを起動
- rank 1: node 1 上で GPU 0-3 に 4 サーバを起動
- rank 2: node 2 上で 8 クライアントを並列起動

サーバの接続情報は `JOB_ROOT/server_ready/scenario_XX.env` に書き出しています。内容は次のような形式です。

```bash
scenario_id=3
server_host=x3209c0s19b1n0.hsn.cm.polaris.alcf.anl.gov
gpu_id=3
rpc_port=20030
tm_port=20530
```

## ポート設計

単一サーバでは `CARLA_RPC_PORT` をそのまま使います。複数サーバでは衝突を避けるため、シナリオごとにポートをずらしています。

```text
rpc_port = BASE_SERVER_PORT + scenario_id * 10
tm_port  = rpc_port + 500
```

たとえば `BASE_SERVER_PORT=20000` の場合は次のようになります。

- scenario 0: `rpc_port=20000`, `tm_port=20500`
- scenario 1: `rpc_port=20010`, `tm_port=20510`
- scenario 7: `rpc_port=20070`, `tm_port=20570`

このずらし方にしておくと、今後 streaming port や追加の内部サービスを使う場合も整理しやすいです。

## 最小クライアント例

単発画像の最小例は [`client_capture.py`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/client_capture.py) です。流れは次の通りです。

1. socket で `host:port` の疎通確認
2. `carla.Client` でサーバに接続
3. 車両を 1 台 spawn
4. RGB カメラを 1 台 attach
5. 車両を前進させ、速度が出たフレームを保存

最小例を抜き出すと次のようになります。

```python
import carla
import socket
import time

def wait_for_port(host, port, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            sock.connect((host, port))
            return
        except Exception:
            time.sleep(2)
        finally:
            try:
                sock.close()
            except Exception:
                pass
    raise RuntimeError("port not ready")

wait_for_port(host, port)
client = carla.Client(host, port)
client.set_timeout(60.0)
world = client.get_world()

bp = world.get_blueprint_library()
vehicle_bp = bp.filter("vehicle.tesla.model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

camera_bp = bp.find("sensor.camera.rgb")
camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=2.4)),
    attach_to=vehicle,
)
```

## マルチカメラ動画クライアント

8 カメラ動画の例は [`client_multicam_video.py`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/client_multicam_video.py) です。

このスクリプトでは次を行っています。

- world の ready を待つ
- synchronous mode を有効にする
- Traffic Manager を別ポートで立てる
- 車両を spawn して autopilot を有効化する
- 8 つの yaw に RGB カメラを付ける
- フレームごとに 3x3 グリッドへ並べて MP4 を保存する

使っているカメラ角度は次です。

```text
300   0   60
270   -   90
240 180  120
```

`scenario_id` と `scenario_count` を渡すと、spawn point をずらして 8 シナリオを散らせます。

## 2 ノードの疎通確認

サーバ 1 台とクライアント 1 台の確認は [`run_capture_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/run_capture_polaris_qsub.sh) を使います。

```bash
qsub tools/carla_multinode_test/run_capture_polaris_qsub.sh
```

このジョブでは次を確認できます。

- node 0 で CARLA サーバが立ち上がる
- node 1 から `rpc_port` に到達できる
- CARLA API で車両とカメラを生成できる
- 運転中の画像を 1 枚保存できる

## 3 ノード 8 シナリオ並列実行

8 本同時生成のジョブは [`run_multiscenario_multicam_video_polaris_qsub.sh`](/eagle/RobustViT/ntakada/gitrepo/simlingo_multiview/tools/carla_multinode_test/run_multiscenario_multicam_video_polaris_qsub.sh) です。

```bash
qsub tools/carla_multinode_test/run_multiscenario_multicam_video_polaris_qsub.sh
```

このジョブは `preemptable` キューを使います。Polaris の `debug` キューは最大 2 ノードなので、3 ノード構成は `debug` には載りません。

実行の流れは次の通りです。

1. PBS が 3 ノードを確保する
2. `mpiexec -n 3 -ppn 1` で rank 0, 1, 2 を 1 ノードずつ割り当てる
3. rank 0, 1 がそれぞれ 4 サーバずつ起動する
4. rank 2 が `server_ready/scenario_XX.env` を読んで 8 クライアントを並列起動する
5. 各クライアントが `multicam_drive.mp4` と `multicam_preview.png` を保存する
6. すべての出力が揃ったら `client.done` を作り、サーバ側が終了する

出力先は次です。

- ジョブルート: `logs/carla_multicam_video_8scenarios/$PBS_JOBID`
- シナリオ別動画: `logs/carla_multicam_video_8scenarios/$PBS_JOBID/scenarios/scenario_XX/multicam_drive.mp4`
- シナリオ別プレビュー: `logs/carla_multicam_video_8scenarios/$PBS_JOBID/scenarios/scenario_XX/multicam_preview.png`
- 実行サマリ: `logs/carla_multicam_video_8scenarios/$PBS_JOBID/summary.log`

## ログ確認

ジョブ状態の確認です。

```bash
qstat -fx <jobid>
```

単体サーバの標準出力です。

```bash
sed -n '1,200p' logs/carla_server_polaris.out
sed -n '1,200p' logs/carla_server_polaris.err
```

3 ノード並列ジョブの summary です。

```bash
sed -n '1,240p' logs/carla_multicam_video_8scenarios/<jobid>/summary.log
```

各シナリオのクライアントログです。

```bash
sed -n '1,240p' logs/carla_multicam_video_8scenarios/<jobid>/scenarios/scenario_00/client.stdout.log
sed -n '1,240p' logs/carla_multicam_video_8scenarios/<jobid>/scenarios/scenario_00/client.stderr.log
```

## 実装上の注意

- `--fakeroot` はこの CARLA コンテナでは使わない
- `-graphicsadapter` を明示した方が Polaris では安定する
- `Saved` はホスト側へ bind してログを残す
- `Traffic Manager` は CARLA RPC ポートと別ポートにする
- login node はプロセス制限が厳しいので、重い処理は `qsub` で投げる
- 3 ノード以上を使うときは `preemptable` キュー前提で考える

## 既知の改善余地

- `summary.log` 生成は `printf >> file` ベースなので、さらに厳密にするならロックを入れてもよい
- 8 シナリオの定義はいまは spawn point ベースなので、将来的に route / weather / actor density を外部設定化できる
- 生成後に 8 本を一覧できる contact sheet や merged movie を作る後処理を追加してもよい
