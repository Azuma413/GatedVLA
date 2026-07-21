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
| `scripts/config/vlajepa_carla_ft.yaml` | 運転用 config(action_dim=2, horizon=64, sdpa, with_state=false, delete_pause_frame=false, frame_stride=12, train_by_epochs+epochs=3, per_device_batch_size=10, freeze_modules=vj_encoder) |

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
# PAI が既にある場合は SKIP_EXPORT=1 で Step 2 のみ。全4,459クリップで約3.5時間かかるので
# debug キュー(walltime 1h)ではなく capacity で流す。--overwrite が出力先を rmtree する。
qsub -q capacity -l walltime=6:00:00 -v SKIP_EXPORT=1 scripts/aurora/prepare_data_vlajepa.pbs
tail -f logs/aurora_prepare_data_vlajepa.out
```

出力 `data/lerobot/carla_lerobot/`: `meta/{info,modality,episodes,tasks}` と
`data/chunk-000/episode_*.parquet`(action=[accel,curvature]) と `videos/chunk-000/observation.images.<cam>/episode_*.mp4`。

**学習が読むのは `data/lerobot` ではなく `data/lerobot_256`**(256²動画。800x600 をデコードしてから縮小するのが
I/O ボトルネックだったため)。`data/lerobot` を作り直したら必ずこれも流す:

```bash
qsub scripts/aurora/reencode_lerobot_256.pbs   # meta をコピー(info.json を256にパッチ)+ parquet を symlink
```

parquet は symlink なので action は自動で伝播するが、**`meta/` はコピーなので再実行しないと古いまま**。
動画は不変なので resumable でスキップされる。`meta/steps_*.pkl`(step index キャッシュ)が残っていたら消す。

## 4. 学習（Aurora XPU, mpiexec + xccl DDP）

**2段階**。Stage 1 で world model(`vj_predictor` + QwenVL)を CARLA 動画に事前学習し、Stage 2 が
その checkpoint を読んで action head を学習する。凍結された `vj_encoder`(facebook/vjepa2-vitl)自体は
どちらの段でも学習しない。

### 4.1 Stage 1: V-JEPA2 world model 事前学習 — **完了済み(2026-07-12)、再実行不要**

```bash
# 実行済み: carla_full_256/clips 全4,459クリップ, BS=1 x 12タイル, 30,000 steps, 7h20m
qsub -v NPROC=12,PPN=12,MAX_STEPS=30000,VIDEO_DIR=... scripts/aurora/train_vjepa_video_carla_venv.pbs
```

成果物 `outputs/vlajepa_carla_video_full/carla_vjepa_pretrain/final_model/pytorch_model.pt` (6.4GB)。
`action` を使わず `wm_loss` のみ・`video_datasets` 分岐(LeRobot 非経由・`DistributedSampler(shuffle=True)` あり)・
`BS=1` で走ったため、**§6 のバグはいずれも Stage 1 には影響していない**(1 は LeRobot 非経由、2 は B>1 のみ、
3 は sampler あり、4 は action 不使用)。

### 4.2 Stage 2: action head の学習

```bash
# 短い動作確認(2タイル, 5ステップ)。MAX_STEPS を渡すと epoch モードが無効になる
qsub -v NPROC=2,MAX_STEPS=5 scripts/aurora/train_vlajepa_carla_venv.pbs
tail -f logs/aurora_train_vlajepa_carla_venv.out

# 1ノードフル(12タイル): config の trainer.epochs 分だけ回す
qsub -q capacity -l walltime=12:00:00 -v NPROC=12,PPN=12 scripts/aurora/train_vlajepa_carla_venv.pbs

# epoch 数を変える(例: 20 epoch)
qsub -q capacity -l walltime=24:00:00 -v NPROC=12,PPN=12,EPOCHS=20 scripts/aurora/train_vlajepa_carla_venv.pbs
```

**`PRETRAINED_CKPT` は §4.1 の Stage 1 checkpoint を既定で読む**(PBS 内で `${REPO_DIR}` から解決、
存在しなければ即エラー終了)。これを渡し忘れると Stage 1 の `vj_predictor` を捨ててランダム初期化から
学習してしまう(2026-07-13 の17時間の学習はこれをやっていた)。意図的にスクラッチから始める場合のみ
`-v PRETRAINED_CKPT=` を空で渡す。

`DATA_ROOT` の既定は **`data/lerobot_256`**(256²動画; §3 の `data/lerobot` はその生成元)。
`BS`(既定10) x タイル数 x `GRAD_ACCUM`(既定1) = global batch。既定は 10x12x1 = **120**。

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

## 6. 学習が壊れていた原因（2026-07-17 に発見・修正、実機検証はこれから）

2026-07-13 の17時間の学習で Policy が学習できなかった原因は4つ重なっていた。いずれも無言で失敗する。

| # | バグ | 症状 | 修正 |
| --- | --- | --- | --- |
| 1 | `pai_to_lerobot.py` が全4,459 episode を `chunk-000` に書くのに `info.json` に `chunks_size: 1000` と書いていた | ローダは `episode_index // chunks_size` でパス解決するため episode 1000 以降が存在しない `chunk-001`〜`004` を指し、例外を `except` が握り潰して **3,459 episode (77.6%) を無言で破棄**(step index の実測: 4,459 中 1,000 trajectory のみ) | `chunks_size: max(total_episodes, 1)`。取りこぼしは例外で落とすように変更 |
| 2 | `VLA_JEPA.py` の `torch.chunk(video_embeddings, chunks=V, dim=0)` | 行は `[b*V+v]` 順なのに連続ブロックで分割するため**異なるサンプルの view を結合**。形状は正しいので無言。**B=1 でのみ正しい**が実際の学習は BS=10 だった → `wm_loss` がゴミ | `view(B,V,L,D).permute(0,2,1,3).reshape(B,L,V*D)` |
| 3 | `dataloader/__init__.py` の `lerobot_datasets` 分岐だけ `shuffle=True` がコメントアウト・sampler 無し | バッチ10件が同一 episode の連続10フレーム(1秒)。相関の極めて高いバッチ | `shuffle=True`(accelerate が shard する) |
| 4 | action が未正規化 | accel(std≈3.5) に対し curvature(std≈0.02) が flow-matching 損失に埋没し旋回を学習しない | `mean_std` 正規化(§設計上の決定) |

その他、同時に判明した点:

- `_get_all_steps` が step index キャッシュのファイル名をハードコードしており(上流が `# @BUG` と明記)、
  config ハッシュを無視して壊れた index を無言でロードし続けていた → ハードコードを撤去し
  `frame_stride`/`num_trajectories` を鍵に追加。
- `repeated_diffusion_steps` を `framework.action_model.*` に置いていたが、コードは `trainer.*` を読むため
  **既定値4が黙って使われていた** → `trainer.*` へ移動。
- `freeze_modules` はオプティマイザから除外するだけで `requires_grad=False` にしない → `vj_encoder.requires_grad_(False)`
  を明示。これで DDP の reducer バケットからも外れる。
- Stage 2 が `PRETRAINED_CKPT` を渡しておらず Stage 1 を捨てていた → PBS の既定値にし、存在しなければエラー終了。
- `epoch` メトリクスが optimizer step を micro-batch 数で割っており grad accum 分だけ膨張していた。

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
