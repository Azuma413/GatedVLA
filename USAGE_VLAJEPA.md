# VLA-JEPA: CARLA/PAIデータでの学習（Aurora XPU）とBench2Drive評価（オンプレ）

`USAGE_Polaris.md` で収集した CARLA 4カメラ(Alpamayo構成)の 123D データを使って
`3rd_party/VLA-JEPA` を学習し、Bench2Drive でクローズドループ評価するための導線です。

- **学習**: Aurora の計算ノード(Intel XPU)。VLA-JEPA 本体のコード(`train_starvla.py` / `gr00t_lerobot`
  dataloader / DiT action head / V-JEPA2 world model)を使い、**mpiexec + xccl による DDP 分散**で回します。
- **評価**: オンプレの推論サーバ(NVIDIA)。VLA-JEPA の推論を **VLA-JEPA 自身の websocket サーバ**
  (`deployment/model_server/server_policy.py`)で動かし、**LEAD のエージェント/PIDコントローラ**
  (`lead/inference/vlajepa_agent.py`)がクライアントとして接続します。

## アーキテクチャ概要

```
[学習: Aurora XPU / frameworks module + venv / mpiexec + xccl]
  data/carla_alpamayo_4c_py123d (123D logs)
    └─(export_py123d_to_pai.py)→ data/carla_alpamayo_pai (PAI)
        └─(pai_to_lerobot.py)→ data/lerobot/carla_lerobot (LeRobot v2.1 + modality.json)
            └─(train_starvla.py, XPU DDP)→ outputs/vlajepa_carla/carla_ft/{final_model,checkpoints}/pytorch_model.pt

[評価: オンプレ NVIDIA]
  CARLA(carla.sif) ── LEAD vlajepa_agent (lead.sif) ──websocket──> VLA-JEPA policy server
                          ▲ unicycleロールアウト→ego waypoints → PID   │ 予測 [accel,curvature]×64 (正規化)
                          └────────────────────────────────────────────┘
```

## 重要: Aurora XPU で「コンテナは使えない/venv を使う」(2026-07-08 検証)

当初 `vlajepa.sif`(Intel IPEX ベース)で学習を試みましたが、**汎用 IPEX コンテナは Aurora の XPU を認識できません**
(`torch.xpu.device_count()==0`)。コンテナ同梱の Level-Zero ドライバがホストの agama カーネルドライバと不整合で、
ホストの `libze_intel_gpu` をバインドするとクラッシュします。**ホストの `frameworks` module は XPU 12基を正常認識**
するため、**学習は `frameworks` module + venv でネイティブ実行**します。`vlajepa.sif` は評価サーバ側(オンプレ NVIDIA)で
のみ利用可能です。

## 重要: Aurora XPU 分散は mpiexec + xccl (torchrun ではない)

`accelerate launch`/torchrun の TCP rendezvous は Aurora の oneCCL と衝突して SIGSEGV します。ALCF 公式手順どおり
**`mpiexec --pmi=pmix --envall` 起動 + `mpi4py` で rank/size + `xccl` backend** を使います。`train_starvla.py` に
`VLAJEPA_XPU_MPI=1` のブートストラップ(全 import より前で `set_device`→`init_process_group("xccl")`)を追加済みで、
accelerate はその既存プロセスグループを再利用します。

## VLA-JEPA 本体への変更点(すべて XPU/運転対応の最小パッチ)

| ファイル | 変更 |
| --- | --- |
| `starVLA/training/train_starvla.py` | 先頭に mpiexec+xccl ブートストラップ(`VLAJEPA_XPU_MPI`); DeepSpeed を env ゲート(`VLAJEPA_USE_DEEPSPEED`, 既定オフ=DDP); DDP `find_unused_parameters=True`(凍結V-JEPA2 encoder対策); autocast device を `VLAJEPA_AMP_DEVICE` で切替 |
| `starVLA/model/framework/VLA_JEPA.py` | autocast device 切替; action model 入力を明示 fp32 化(XPU は fp32 autocast 非対応) |
| `starVLA/model/modules/vlm/QWen3.py` | `attn_implementation` を config から(=`sdpa`); `device_map="cuda"` 撤去(CPUロード) |
| `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | `pytorch3d` を遅延 import(XPUビルド無し・運転では未使用) |
| `starVLA/dataloader/__init__.py` | `delete_pause_frame` を config から渡す |
| `starVLA/dataloader/gr00t_lerobot/{data_config,mixtures,embodiment_tags}.py` | `carla_ego` データ設定/mixture/embodiment を登録 |
| `starVLA/config/aurora_xpu.yaml` | (参考)非DeepSpeed の XPU accelerate 設定 |
| `scripts/config/vlajepa_carla_ft.yaml` | 運転用 config(action_dim=2, horizon=64, sdpa, with_state=false, delete_pause_frame=false) |

## リポジトリ側の追加ファイル

| ファイル | 役割 |
| --- | --- |
| `scripts/pai_to_lerobot.py` | PAI → LeRobot v2.1(+modality.json)変換。行動=unicycle制御[accel,curvature](Alpamayo 1.5) |
| `scripts/aurora/setup_venv_vlajepa.sh` | frameworks module 上に不足依存だけ重ねる venv 作成 |
| `scripts/aurora/prepare_data_vlajepa.pbs` | 123D→PAI→LeRobot(lead.sif, 計算ノード) |
| `scripts/aurora/train_vlajepa_carla_venv.pbs` | **Aurora XPU 学習(mpiexec+xccl DDP)** |
| `vlajepa.def`, `scripts/aurora/build_vlajepa.pbs` | (評価サーバ用)IPEX コンテナ。※Aurora XPU 学習には使えない |
| `lead/inference/vlajepa_client.py`, `vlajepa_agent.py` | 評価: websocket クライアント + LEAD SENSORS エージェント |
| `lead/__main__.py` | `--vlajepa` フラグ |
| `scripts/eval/vlajepa_server.sh`, `eval_vlajepa_bench2drive.sh` | 評価サーバ起動 + LEAD評価(オンプレ) |

## 設計上の決定

- **行動表現(Alpamayo 1.5準拠)**: 各フレームの `action` = ユニサイクル制御 `[accel, curvature]`(m/s², 1/m)。
  Alpamayo 1.5 の `UnicycleAccelCurvatureActionSpace`(`3rd_party/alpamayo1.5`)で軌道から算出する。未来 `H=64`
  (≈6.4秒 @10Hz)を予測し、評価時に LEAD 側で `action_to_traj`(現在speedを初期値)にロールアウトして自車 waypoint
  列に戻す。**accel と curvature はスケールが大きく異なる(std ≈3.5 対 ≈0.02)ため、各次元を mean/std 正規化する**
  (未正規化だと curvature が flow-matching 損失で埋没し旋回を学習できない)。推論側は server の `norm_stats` で
  un-normalize してからロールアウトする。
- **カメラ**: LeRobot video_key 順は `[front_wide, cross_left, front_tele, cross_right]`。VLM に4枚、V-JEPA2 に
  index 0,2(front_wide, front_tele)。評価時 LEAD の `rgb_1..4=[cross_left,front_wide,cross_right,front_tele]` なので
  エージェントは `[rgb_2, rgb_1, rgb_4, rgb_3]` の順で送る。
- **言語命令**: PAI の `navigation_command`(`command_text`)を annotation(task)に使用。

---

## 0. 前提

- `USAGE_Polaris.md` §1 で収集した 123D ログ(`data/carla_alpamayo_4c_py123d/logs/carla_train/...`)。
- Aurora ログインノードは user namespace 無効なので **apptainer を使う処理(データ変換)は計算ノード(PBS)で実行**。
- プロジェクト `RobustViT`、queue `debug`、`filesystems=home:flare`。

## 1. 事前学習済みモデルを用意（ログインノード, proxy経由）

```bash
module use /soft/modulefiles; module load frameworks
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
hf download Qwen/Qwen3-VL-2B-Instruct    --local-dir outputs/models/Qwen3-VL-2B-Instruct
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir outputs/models/vjepa2-vitl-fpc64-256
```

## 2. 学習用 venv を作成（ログインノード, proxy経由, 1回のみ）

Aurora の `frameworks/2025.3.1`(torch 2.10+xpu / transformers 4.57 / accelerate / deepspeed 済)の上に、
不足依存(diffusers, timm, decord, numpydantic, av, qwen-vl-utils, tensorboard, albumentations)だけを venv に足します。
torch/torchvision は module のものを維持(再インストールすると XPU が壊れる)。

```bash
bash scripts/aurora/setup_venv_vlajepa.sh   # outputs/vlajepa_venv を作成
```

## 3. データ変換（123D → PAI → LeRobot v2.1, 計算ノード）

`lead.sif`(CPU)で実行。検証時は少数クリップだけ変換(`MAX_LOGS`, 既定8; 全件は `MAX_LOGS=0`)。

```bash
qsub scripts/aurora/prepare_data_vlajepa.pbs
tail -f logs/aurora_prepare_data_vlajepa.out
```

出力 `data/lerobot/carla_lerobot/`: `meta/{info,modality,episodes,tasks}` と
`data/chunk-000/episode_*.parquet`(action=[accel,curvature]) と `videos/chunk-000/observation.images.<cam>/episode_*.mp4`。

## 4. 学習（Aurora XPU, mpiexec + xccl DDP）

```bash
# 短い動作確認(2タイル, 5ステップ)
qsub -v NPROC=2 scripts/aurora/train_vlajepa_carla_venv.pbs
tail -f logs/aurora_train_vlajepa_carla_venv.out

# 1ノードフル(12タイル)
qsub -v NPROC=12,PPN=12,MAX_STEPS=30000 scripts/aurora/train_vlajepa_carla_venv.pbs

# 複数ノード(例: 4ノード=48タイル)
qsub -l select=4 -v NPROC=48,PPN=12,MAX_STEPS=30000 scripts/aurora/train_vlajepa_carla_venv.pbs
```

内部は `mpiexec --pmi=pmix --envall -n ${NPROC} -ppn ${PPN} python -u train_starvla.py --config_yaml
scripts/config/vlajepa_carla_ft.yaml --<overrides>`。`NPROC` は総タイル数、`PPN`(≤12)はノード内タイル数。
`PPN=12` のとき CPU affinity(`--cpu-bind`)を自動適用。モデル/データパスは yaml のコンテナパスを実パスへ上書き。

環境: `VLAJEPA_AMP_DEVICE=xpu`, `VLAJEPA_XPU_MPI=1`, `HF_HUB_OFFLINE=1`。成果物:

```text
outputs/vlajepa_carla/carla_ft/config.yaml
outputs/vlajepa_carla/carla_ft/dataset_statistics.json
outputs/vlajepa_carla/carla_ft/checkpoints/steps_${N}_pytorch_model.pt
outputs/vlajepa_carla/carla_ft/final_model/pytorch_model.pt
```

> DeepSpeed(ZeRO)は既定オフ(DDP)。1タイルにモデル+optimizerが載らない場合のみ `VLAJEPA_USE_DEEPSPEED=1` で有効化
> (Aurora XPU での DeepSpeed 初期化は別途要検証)。

## 5. 評価（オンプレ NVIDIA サーバ, Bench2Drive）

> 評価はオンプレ環境で実施(本リポジトリでの動作確認は未実施, 実装のみ)。LEAD 側には `pip install websockets msgpack` が必要。

**(a) VLA-JEPA 推論サーバ**(VLA-JEPA/CUDA 環境):

```bash
CKPT_PATH=outputs/vlajepa_carla/carla_ft/final_model/pytorch_model.pt PORT=10093 \
  bash scripts/eval/vlajepa_server.sh
```

`server_policy.py` が `baseframework.from_pretrained(CKPT_PATH)` でモデルを復元(同じ run dir の `config.yaml`,
`dataset_statistics.json` を参照)し、`predict_action` を websocket 提供。

**(b) CARLA 起動 → (c) LEAD 評価**(`lead.sif`):

```bash
VLAJEPA_SERVER_HOST=<host> VLAJEPA_SERVER_PORT=10093 CARLA_HOST=<host> CARLA_PORT=2000 \
ROUTE=data/benchmark_routes/bench2drive/23687.xml OUTPUT_ROOT=outputs/vlajepa_b2d_eval \
  bash scripts/eval/eval_vlajepa_bench2drive.sh
```

`python -m lead --checkpoint vlajepa --vlajepa --bench2drive --routes ...` を実行。`vlajepa_agent.py` が毎
`VLAJEPA_INFERENCE_INTERVAL` ステップで4カメラ(各224²にリサイズ)+nav命令を送り、返る正規化 `[accel,curvature]×64`
を server の `norm_stats` で un-normalize → `action_to_traj`(現在speed起点)でロールアウトして `execute_waypoints`
(LEAD PID)へ。出力: `23687_input.mp4`, `metric_info.json`, `infractions.json`。

---

## 動作確認状況（2026-07-08）

- [x] `py_compile` / `[accel,curvature]`↔`action_to_traj` 往復(実CARLAクリップで誤差≈0.44m/6.4s・40m, 旋回y符号一致) / PAI→LeRobot 変換
- [x] Qwen3-VL-2B / V-JEPA2 ダウンロード
- [x] データ変換(実データ8クリップ→8 episodes/2437 frames)
- [x] frameworks venv 構築(XPU torch 維持)
- [x] 実データで単一XPU forward 完走(`action_loss≈1.3`)
- [x] xccl 分散(mpiexec, 4rank all_reduce 正)
- [x] **2タイル DDP 分散学習が5ステップ完走 + checkpoint 保存**(`final_model/pytorch_model.pt`, 6.8GB)
- [ ] 12タイル/複数ノードのスケール確認(NPROC/PPN を上げるだけ)
- [ ] 評価(オンプレ, 未実施 — 実装のみ)
- [ ] コンテナでの XPU 学習は Aurora では不可(venv 方式を使用)
