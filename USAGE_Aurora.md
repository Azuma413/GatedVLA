# Aurora: CARLA 123Dログを全件PAI形式へ変換

`USAGE_Polaris.md` §2 と同じ「123D → PAI」変換を、**Aurora で `carla_train` の全クリップ
(4566 clip)** に対して行う手順です。`USAGE_Polaris.md` は 1 プロセスで少数 clip を変換していましたが、
全件だと 1 プロセスでは ~33 時間かかるため、Aurora の 1 計算ノード上で**複数シャードに分割して並列変換**し、
最後に 1 つの PAI データセットへマージします。

対象:

```
data/carla_alpamayo_4c_py123d/logs/carla_train   (4566 clip, 各 clip に sync.arrow + 4カメラ + ego_state + navigation_command.parquet)
        └─(export_py123d_to_pai.py, N シャード並列)→ data/carla_alpamayo_pai_shards/shard_XXXX
                └─(merge_pai_shards.py)→ data/carla_alpamayo_pai   (単一の PAI データセット)
```

> PAI → LeRobot 変換や学習は `USAGE_VLAJEPA.md` を参照。本ドキュメントは **PAI への全件変換のみ**を扱います。

## 重要な前提・制約（2026-07-08 検証）

- **ログインノードでは apptainer が使えない**（user namespace 無効）。`lead.sif` を使う変換は
  必ず **PBS の計算ノード**で実行する（`USAGE_VLAJEPA.md` と同じ）。
- **1 プロセスの変換は I/O バウンドで ~26 秒/clip**（`lead.sif` 内、24 clip を 10:27 で変換、CPU は 1 コアの
  84% しか使わず、Arrow を Lustre から mmap 読みするページフォルトが支配的）。全 4566 clip では ~33 時間。
  → 1 ノードで **N 個のシングルスレッド変換プロセスを並列**に走らせる（Aurora の計算ノードは ~104 物理コア、
  I/O バウンドなのでコア数以上のオーバーサブスクライブも可）。
- **キューの選び方**: 本番キュー（`small`/`medium`/`large`）は **最小 256 ノード**必要で、1 ノードジョブには使えない。
  1 ノードを長時間確保できるのは **`capacity` キュー（1–16 ノード、walltime 最大 168h）**。`debug` は
  1h・最大 2 ノードなので検証用にのみ使う。
- 出力サイズは **~23 MB/clip**（4566 clip で ~105 GB）。flare に十分な空きがあることを確認する
  （`df -h /lus/flare/projects/RobustViT`）。マージは**同一ファイルシステム内の rename（移動）**なので
  データを二重に持たない（シャード出力→最終出力へ move）。

## リポジトリ側の変更・追加

| ファイル | 変更/役割 |
| --- | --- |
| `scripts/export_py123d_to_pai.py` | `--num-shards` / `--shard-index` を追加。`find_log_dirs` のソート済みリストを N 等分し、担当シャードだけを変換（各シャードは chunk 番号 0 から開始）。既存の単一プロセス動作は不変。 |
| `scripts/merge_pai_shards.py`(新規) | 各シャード dir の chunk zip を**グローバル chunk オフセット**を付けて最終 dir へ move し、`clip_index` / `feature_presence` を連結、`features.csv` を再生成。member 名は clip_id 前置で一意なので衝突しない。 |
| `scripts/aurora/export_pai_full.pbs`(新規) | `capacity` キュー・1 ノードで N シャードを 1 コンテナ内のバックグラウンドプロセスとして並列実行→マージ。 |

**設計のポイント**: PAI の消費側（`pai_to_lerobot.py`）は clip ごとに `clip_index.parquet` の `chunk` 列で
zip を特定し、member 名を `{clip_id}.{...}` で構成する。chunk 番号の連続性やサイズには依存しない。よって
シャードごとに chunk を 0 から振り、マージ時にオフセットを足すだけで単一プロセス出力と等価な PAI になる
（chunk 番号だけは異なるが consumer に影響しない）。

## 手順

### 0. 事前確認

```bash
cd /lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research
# clip 数
find data/carla_alpamayo_4c_py123d/logs/carla_train -maxdepth 2 -name sync.arrow | wc -l   # -> 4566
# 空き容量（>~105 GB 必要）
df -h /lus/flare/projects/RobustViT
```

### 1.（任意・推奨）小規模で shard+merge を検証

`debug` キューで少数 clip を「1 プロセス serial」と「N シャード+マージ」の両方で変換し、
clip 集合・feature_presence・各 clip の zip member が一致することを確認する。

```bash
qsub scripts/aurora/_test_shard_merge.pbs
tail -f logs/aurora_test_shard_merge.out    # 末尾に "RESULT: PASS"
```

### 2. 全件変換（capacity キュー、1 ノード）

```bash
qsub scripts/aurora/export_pai_full.pbs
tail -f logs/aurora_export_pai_full.out
```

既定値（`qsub -v KEY=VAL,...` で上書き可）:

```bash
NUM_SHARDS=48    # 並列シャード数。4566/48 ≈ 95 clip/シャード ≈ 41 分
PAI_DIR=data/carla_alpamayo_pai
SHARD_ROOT=data/carla_alpamayo_pai_shards
LOGS_ROOT=data/carla_alpamayo_4c_py123d
MAX_LOGS=0       # 0=全件。>0 で総 clip 数を絞れる（検証用）
```

ジョブは 2 段階:

1. **Step 1（変換）**: 1 つの `lead.sif` コンテナ内で `NUM_SHARDS` 個の
   `export_py123d_to_pai.py --num-shards N --shard-index i` をバックグラウンド起動し `wait`。
   各シャードのログは `data/carla_alpamayo_pai_shards/shard_XXXX.log`、出力は同 `shard_XXXX/`。
   失敗/空シャードは `clip_index.parquet` が無いので Step 2 が報告する（当該 shard-index だけ再実行可能）。
2. **Step 2（マージ）**: `merge_pai_shards.py` が全シャードを `data/carla_alpamayo_pai` へ統合。

成功すると次が生成される（`USAGE_Polaris.md` §2 と同じ PAI レイアウト）:

```text
data/carla_alpamayo_pai/clip_index.parquet
data/carla_alpamayo_pai/features.csv
data/carla_alpamayo_pai/metadata/feature_presence.parquet
data/carla_alpamayo_pai/camera/{camera_cross_left_120fov,camera_front_wide_120fov,camera_cross_right_120fov,camera_front_tele_30fov}/*.chunk_XXXX.zip
data/carla_alpamayo_pai/labels/egomotion/*.chunk_XXXX.zip
data/carla_alpamayo_pai/labels/navigation_command/*.chunk_XXXX.zip
```

### 3. 後始末

マージ後、シャードの zip は最終 dir へ move 済み。残った空ディレクトリは削除してよい:

```bash
rm -rf data/carla_alpamayo_pai_shards
```

## トラブルシュート

- **Step 2 が `no clip_index.parquet ... skipping` を多数出す**: そのシャードの
  `data/carla_alpamayo_pai_shards/shard_XXXX.log` を確認。log 出力先の親ディレクトリが無いと
  リダイレクトに失敗して変換が起動しない（`SHARD_ROOT` は PBS が `mkdir -p` する）。
- **walltime が足りない**: `NUM_SHARDS` を増やす（例 64/96）か、`qsub -l walltime=...` を延ばす。
  `capacity` は最大 168h。
- **`OMP_NUM_THREADS already has value [1], will not forward [208]`** / **`gocryptfs not found`**:
  apptainer の無害な警告。

## 動作確認状況（2026-07-08）

- [x] 1 プロセスの変換速度計測: 24 clip を 10:27（~26 s/clip、CPU 1 コアの 84%、I/O バウンド）。
- [x] 小規模 shard+merge 検証（`_test_shard_merge.pbs`, 12 clip / 3 シャード）: serial 変換と
  clip 集合・feature_presence・各 clip の zip member が一致（`RESULT: PASS`）。各シャードの
  local chunk 0 → global chunk 0/1/2 へ正しくオフセットされることを確認。
- [x] **全 4566 clip の PAI 変換完了**（`export_pai_full.pbs`, `NUM_SHARDS=48`, ジョブ `8657187`）。
  48 シャード全て正常終了（エラー0）。**4459 clip を変換**、107 clip は「clip が短すぎる」（Alpamayo の
  history 1.6s + future 6.4s ≈ 8s マージンに満たない）ため export がスキップ（`export_py123d_to_pai.py`
  本来の挙動、Polaris の 1 プロセス版と同一）。4459 + 107 = 4566 で全件説明可能。
  出力 `data/carla_alpamayo_pai`（4459 clip / 48 chunk）。
  - 実測: 48 並列で **wall ~1h35**（1 プロセス換算 ~33h から短縮。I/O 競合で実効並列度は 48 より低い）。
  - スキップされた 107 clip を確認: `grep -h Skipping data/carla_alpamayo_pai_shards/shard_*.log`
- [x] マージ後データの整合性チェック（`scripts/_check_pai_integrity.py`, ジョブ `8657709`, `RESULT: PASS`）:
  4459 clip / clip_id 全て一意 / chunk 0..47（48 distinct）/ 全 clip で 6 feature（egomotion,
  navigation_command, 4 カメラ）が 100% present / ランダム 30 clip の zip member が全て解決。

---

# Aurora で CoC（Chain-of-Causation）ラベルを付与

`data/carla_alpamayo_pai`（上で作った PAI データセット）に、NVlabs
`3rd_party/alpamayo-coc-autolabeler` で CoC ラベルを付与する手順です。Polaris 版
（`scripts/polaris/coc_autolabel_short.*`）は CoC の VLM 段を **NVIDIA GPU + CUDA vLLM**
でローカル実行していましたが、**Aurora は Intel XPU（NVIDIA GPU なし）**なので、VLM 段は
**Aurora XPU 上で動く vLLM の OpenAI 互換サーバ**に対して autolabeler を**リモートクライアント**
として実行します。

```
[Aurora 1 計算ノード]
  frameworks module (native) : vLLM OpenAI サーバ (XPU, Qwen3-VL, :50000)
        ▲ 127.0.0.1:50000 (ノード内で host/コンテナはネットワーク共有)
  coc_autolabeler.sif (apptainer) : meta-action → keyframe → video 抽出 (CPU)
                                     → CoC ラベリング (VLMCoCRemoteAgent, リモート)
```

CoC pipeline は 4 段:
`meta-action`(trajdata, CPU) → `keyframe`(CPU) → `video 抽出`(CPU) → `CoC ラベリング`(VLM)。
最初の 3 段は Polaris と同一で `coc_autolabeler.sif` 内 CPU 実行。4 段目だけ Aurora 固有です。

## 重要な前提・制約（2026-07-09 検証）

- **vLLM は frameworks module（vLLM 0.15.0 / torch 2.10+xpu）で XPU 対応**。`vllm serve` 相当を
  ネイティブ起動し **OpenAI 互換サーバ**として使う（`USAGE_VLAJEPA.md` と同じく学習系は container 不可）。
  **fp8/awq は XPU 非対応**なので `--dtype float16` の**非量子化モデル**のみ。
- **モデルは Qwen3-VL 系に限られる**。autolabeler 既定の `qwen3.5_27b_fp8` / `qwen3.5_35b`
  （オリジナルが使う大型モデル）は、(1) fp8、(2) **arch（`Qwen3_5*ForConditionalGeneration`）が
  frameworks の vLLM 0.15.0 に未登録**、の二重の理由で **Aurora では起動不可**。vLLM 0.15 が対応する
  Qwen3 系 VL arch は `Qwen3VLForConditionalGeneration`（dense 2B/8B/32B）と
  `Qwen3VLMoeForConditionalGeneration`（30B-A3B）。
- **本番は `Qwen3-VL-8B-Instruct`（1 タイル, TP 不要）**。スモークは `Qwen3-VL-2B-Instruct`。
  **32B/30B は TP≥2 が必須（62GB fp16 > 1 タイル 64GB）だが、vLLM 0.15 XPU の Ray テンソル並列が
  worker 初期化でハングする**（"Connected to Ray cluster" 以降 progress ログ無しで walltime まで停止。
  2 タイル/12 タイル両方で再現。`external_launcher` は SPMD でオンラインサーバに使えない）。
  よって現状 **1 タイルに載る 8B が実用上の最良**。32B を使うにはこの XPU Ray TP ハングの解消
  （ALCF サポート案件）が必要。
- **初回 modelinfo inspection が SIGSEGV する（ALCF 既知バグ）**。`ModelConfig` の validation error として
  現れる。ALCF の WA スクリプト `scripts/aurora/_wa/vllm_build_all_modelinfo_caches.py --arch <arch>` で
  `VLLM_CACHE_ROOT` に cache を**事前生成**すれば回避（`VLLM_CACHE_ROOT` を変えるたびに 1 回）。
- **autolabeler の remote video 形式は stock vLLM で復号できない**。remote QwenWrapper は
  `data:video/jpeg;base64,<frame1,frame2,...>`（NVIDIA 内部サーバ用の連結形式）を送るが、stock vLLM は
  これを 1 本の動画コンテナとみなして失敗する。**フレームを `image_url` のリストで送る**よう
  `qwen.py` を修正済み（Qwen3-VL が画像列として正しく処理。構造化出力 `guided_json` も XPU vLLM で動作確認済み）。
- **trajdata は `data_dir` 内の全 clip を先に前処理してから** `--scene_list` 絞り込みを適用する。よって
  **少数 clip の検証には subset PAI が必要**（`_make_pai_subset.py`）。さらに一部 clip は ego 姿勢の
  SLERP 補間が `Interpolation times must be within [...]` で落ち、**1 個でも壊れた clip があると
  meta-action 全体が巻き添えで落ちる**。query 時刻を範囲内に clamp する **pai_dataset.py パッチを
  bind-mount** して回避（clip は落とさず全件残す）。
- **キュー**: サーバ起動＋TP は 1 ノードで完結。検証は `debug`（1h・最大2ノード・**Q 投入数に上限**あり）。
  全件の CoC は計算量が大きく（下記）`capacity`（1–16 ノード, 最大168h）で**複数サーバ/複数ノードに分割**する。

## リポジトリ側の変更・追加

| ファイル | 変更/役割 |
| --- | --- |
| `3rd_party/alpamayo-coc-autolabeler/.../vlm_wrappers/qwen.py` | remote video を `image_url` リスト送信に修正; `qwen3_vl_8b` / `qwen3_vl_32b`（非量子化）を登録; remote の video-only `mm_processor_kwargs` を除去 |
| `.../vlm_wrappers/factory.py` | `qwen3_vl_8b` / `qwen3_vl_32b` を wrapper registry に追加 |
| `scripts/aurora/coc_autolabel.pbs`(新規) | XPU で vLLM サーバ起動（modelinfo cache 事前生成 + TP>1 は Ray）→ `coc_autolabeler.sif` で CoC pipeline。`COC_MODEL`/`STAGE`/`PAI_DIR`/`CLIP_LIST` で制御。CPU 段（meta/keyframe/video/prep）ではサーバを起動しない |
| `scripts/aurora/coc_autolabel.sh`(新規) | container 内の 4 段実行。stage 4 は `VLMCoCRemoteAgent` + `server_ip=127.0.0.1`。`SEGMENT_JSON`/`RESUME_DIR`/`WORK` を env 化、hydra run dir を worker 毎に一意化 |
| `scripts/aurora/coc_autolabel_parallel.pbs`(新規) | **1 ノード=12 タイルをフル活用**: `ZE_AFFINITY_MASK=i` で 8B サーバを 12 個（:50000..:50011）+ clip 分割した 12 client。prep 未実施なら自動で先に実行。resumable |
| `scripts/aurora/_split_segment_json.py`(新規) | keyframe segment json を clip 単位で N 分割（meta-action 構造は保持） |
| `.../model_clients/chat_utils_local_server.py` | サーバ接続 port を env `COC_LABELING_MODEL_SERVER_PORT`(既定 50000) から取得（多サーバ用） |
| `scripts/aurora/_make_pai_subset.py`(新規) | 検証用に PAI を指定 clip だけの subset へ切り出し（各 chunk zip を該当 member だけで再作成） |
| `scripts/aurora/_wa/vllm_build_all_modelinfo_caches.py` | ALCF の modelinfo cache 事前生成 WA（外部取得） |
| `scripts/aurora/_wa/pai_dataset.patched.py` | trajdata pai loader の SLERP clamp パッチ（bind-mount 用） |

## 0. 事前準備（ログインノード, proxy 経由）

```bash
cd /lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research
module use /soft/modulefiles; module load frameworks
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
# 本番 8B（≈17GB, 1タイル）とスモーク 2B（VLA-JEPA 手順で取得済みなら不要）
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir outputs/models/Qwen3-VL-8B-Instruct
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir outputs/models/Qwen3-VL-2B-Instruct
# 32B は TP ハングのため保留（使うなら別途 outputs/models/Qwen3-VL-32B-Instruct を取得）
```

`pai_dataset.patched.py` は既にリポジトリにあります（無ければ `_extract_trajdata_pai.pbs` で
sif から取り出して SLERP 箇所を clamp）。

## 1. 小規模検証（3 clip subset, end-to-end）

trajdata は data_dir 全 clip を前処理するので、まず subset を作ります（17.5 秒以上の長め clip を選ぶ）。

```bash
# 22–28s の Town03 clip を 3 本選んで subset 化
module load frameworks
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/carla_alpamayo_pai/clip_index.parquet")
dur = (df.end_timestamp - df.start_timestamp)/1e6
sel = dur[(dur>=22)&(dur<=28)].sort_index().head(3)
open("logs/coc_verify_cliplist.txt","w").write("\n".join(sel.index)+"\n")
EOF
python scripts/aurora/_make_pai_subset.py \
  --src data/carla_alpamayo_pai --dst data/carla_alpamayo_pai_verify \
  --clip-list logs/coc_verify_cliplist.txt
```

```bash
# スモーク（2B, 1 タイル）
qsub -v COC_MODEL=qwen3_vl_2b scripts/aurora/coc_autolabel.pbs
# 本番モデルの動作確認（8B, 1 タイル）
qsub -v COC_MODEL=qwen3_vl_8b scripts/aurora/coc_autolabel.pbs
tail -f logs/aurora_coc_autolabel.out
```

既定は `PAI_DIR=data/carla_alpamayo_pai_verify`, `STAGE=all`。成果物:

```text
outputs/coc_autolabel_aurora/experiments/<timestamp>_<model>_carla_aurora/<clip_id>/cot_<ts>.yaml
  └─ final_content.ego_behavior_schema.effect_on_ego_behavior: "<CoC ラベル文>"
```

## 2. 全件（4459 clip）— 計算量と分割方針

CoC ラベリングは 1 clip あたり複数 segment（history 2s + future 6s を 0.5s 間隔 ≒ 16 フレーム/segment、
1 clip に十数〜数十 segment）で、17.5 秒以上の **~3253 clip** が対象。1 サーバ直列では
数百時間規模なので、**CPU 段と VLM 段を分け、VLM 段を複数サーバ/ノードに分割**します。

**推奨: 12 タイル並列（`coc_autolabel_parallel.pbs`）**。8B は 1 タイルに載るので、1 ノードの 12 タイルに
サーバを 12 個立て（`ZE_AFFINITY_MASK=i` でタイル分離、port :50000+i）、clip を 12 分割して同時にラベリングします。
**Aurora はノード単位課金**なので、同じ node-hours で **約 12×** 速い（直列 1.5〜2.5 日 → **~3〜4 時間**）。prep 未実施なら
ジョブ内で自動実行。`RESUME_DIR` により **walltime で切れても同じ qsub を再投入すれば続きから**再開します。

```bash
qsub -q capacity -l walltime=8:00:00 scripts/aurora/coc_autolabel_parallel.pbs
# 既定: NSERVERS=12, PAI_DIR=data/carla_alpamayo_pai,
#       WORK=outputs/coc_autolabel_aurora_full,
#       RESUME_DIR=outputs/coc_autolabel_aurora_full/coc_labels_8b
tail -f logs/aurora_coc_parallel.out
```

出力: `outputs/coc_autolabel_aurora_full/coc_labels_8b/<clip_id>/cot_<ts>.yaml`。
さらに速くするなら複数ノード（各ノード 12 サーバ）へ clip を分けて投げる（`PAI_DIR`/`RESUME_DIR` 共有、
`CLIP_LIST` かシャード json で担当を分割）。

> 単一タイル版（`coc_autolabel.pbs`, `STAGE=all`）も resumable で残してあるが、ノードの 1/12 しか使わないため
> 全件には非推奨。32B にこだわる場合は XPU Ray TP ハングの解消が前提。

## トラブルシュート

- **`validation error for ModelConfig` / registry subprocess SIGSEGV**: modelinfo cache 未生成。
  `logs/aurora_coc_autolabel_server.log` を確認し、WA の prepopulation が走ったか見る。
- **`server not ready`**: 32B は TP=2 で数分〜十数分かかる。`server log tail` を確認。メモリ不足なら
  `MAX_MODEL_LEN` を下げる（既定 32768）か `MM_IMAGE_LIMIT` を下げる。
- **meta-action が `Interpolation times must be within [...]` で落ちる**: `pai_dataset.patched.py` の
  bind-mount が効いていない（`scripts/aurora/_wa/pai_dataset.patched.py` の存在を確認）。

## CoC 動作確認状況（2026-07-09）

- [x] Aurora XPU で vLLM(frameworks) 起動。初回 modelinfo SIGSEGV を WA cache 事前生成で回避。
- [x] remote 形式の互換性を probe（`_coc_vllm_probe.*`）: 連結 `video_url`=NG、`image_url` リスト=OK、
  `guided_json`=OK。→ wrapper を `image_url` リスト送信に修正。
- [x] **2B スモークで 3 clip の end-to-end 成功**（ジョブ `8658231`）: meta→keyframe→video→CoC が完走し、
  3 clip 合計 17 個の `cot_*.yaml`（`ego_behavior_schema.effect_on_ego_behavior`）を生成。
- [x] **本番 8B（1 タイル）で 3 clip の end-to-end 成功**（ジョブ `8659877`, 全体 ~16 分）: server ready→
  17 個の `cot_*.yaml` を生成。ラベル例「Nudge left to pass the parked vehicle on the right while
  maintaining a safe distance from the lead vehicle」など 2B より具体的。
- [x] **32B（TP=2）は BLOCKED**: vLLM 0.15 XPU の Ray テンソル並列が worker 初期化でハング。
  2 タイル（ジョブ `8658828`）・12 タイル（`8659603`, ALCF 既定どおり全タイル公開）とも "Connected to
  Ray cluster" 以降 progress 無しで walltime まで停止。`enforce_eager`・`VLLM_ENGINE_READY_TIMEOUT_S`
  延長でも同じ。online サーバに `external_launcher`（mpiexec+xccl）は使えない（SPMD で rank gating 無し）。
- [x] **SLERP パッチを全件で確認**（`STAGE=meta`, fresh `WORK`, ジョブ `8659936`, ~6 分）: trajdata が
  **4459/4459 clip を前処理完了**（従来は clip #17 で SLERP エラー→全体 crash）。meta-action は
  17.5 秒以上の **3253 clip** 分の出力を生成。全件前処理が壊れ clip の巻き添えで落ちないことを確認。
- [x] **12 タイル並列の仕組みを検証**（`coc_autolabel_parallel.pbs`, N=2, 3-clip subset, ジョブ `8660410`）:
  2 サーバが別タイル（`ZE_AFFINITY_MASK`）・別 port（:50000/:50001）で起動、clip 分割した 2 client が
  shard_00=8 / shard_01=9 segment を処理し 17 個の `cot_*.yaml` を生成（起動込み実働 ~4 分）。
  ※最初は `ONEAPI_DEVICE_SELECTOR` ではタイル分離できず 2 サーバが同一タイルに載って KV OOM →
  `ZE_AFFINITY_MASK=i` で解決。
- [x] **全件 12 タイル並列 CoC 完了**（`coc_autolabel_parallel.pbs`, `capacity`, ジョブ `8660604`, Exit 0）。
  1 ノードで **8B サーバ 12 台**（`ZE_AFFINITY_MASK=i`）＋ 12 シャードクライアントを回し、**ノード稼働 2h20m**
  （起動 ~12 分＋ラベリング ~2 時間、`used walltime 02:20:47`）で完了。単一タイル ~2 日 → ~2.3 時間（同 node-hours）。
  結果: **11,134 個の `cot_*.yaml`／3,076 clip**（対象 3,253 clip の 94.6%。残りは agent の video/timestamp チェックで
  segment が落ちたもの）。出力 `outputs/coc_autolabel_aurora_full/coc_labels_8b/<clip_id>/cot_<ts>.yaml`。
  - 起動時の注意: 12 サーバ同時起動は (a) rayon/OMP スレッドの枯渇（`EAGAIN`/panic）と (b) 12×16GB の同時
    Lustre 読みで失敗する。**スレッド上限**（`RAYON/OMP/MKL_NUM_THREADS`）＋**逐次起動**（1 台 ready→次）で回避。
  - タイル分離は `ONEAPI_DEVICE_SELECTOR` では効かず（2 台が同一タイルに載り KV OOM）、**`ZE_AFFINITY_MASK=i`** が必須。

---

# Aurora で Alpamayo 1.5 を追加学習（nav+CoC SFT）

`data/carla_alpamayo_pai`（PAI データセット）＋ `outputs/coc_autolabel_aurora_full/coc_labels_8b`
（上で付与した CoC ラベル）を使い、**Alpamayo 1.5**（`nvidia/Alpamayo-1.5-10B`）を Aurora の
**1 計算ノード（12 XPU タイル）**上で 2 段 SFT します。使うのは **1.5 レシピ**
（`3rd_party/alpamayo-recipes/recipes/alpamayo1_5_sft`, `alpamayo1_5_sft.train_hf`）で、
旧 R1 レシピ（`3rd_party/alpamayo`, `finetune.sft`）とは別物です。

```
Stage 1a (VLM SFT): 画像 + 過去軌道 + nav command -> 離散 trajectory token（nav processor, cot 無し, 全サンプル）
Stage 1b (VLM SFT): 上に CoC を追加学習（nav_cot processor, cot+traj, CoC 付きサブセット, 1a から継続）
Stage 2  (traj expert): VLM 凍結で trajectory diffusion expert を学習（nav processor, 教師は traj_future のみ）
```

> CoC はレシピ上オプション（`sft_stage1_nav` の既定プロセッサは cot 無しの `nav`）。論文の Stage 1 は
> 2 フェーズで、1a で全データの軌道トークン化を学習してから、1b で CoC 付きデータに cot+nav を上乗せする。
> 1b は `model.checkpoint_path` で 1a の出力を読む（ローダが `vlm.` 接頭辞を剥がす）。fresh optimizer。

- **分散は frameworks module + venv + mpiexec/xccl**（container は XPU 不可、torchrun は oneCCL で
  segfault。VLA-JEPA / CoC と同じ。[[aurora-xpu-distributed]]）。1 rank / タイルで
  `alpamayo15_xpu_entry.py` が xccl group を張ってから `train_hf` を起動する。
- **実効 batch size は 12 タイルで割り切れる 36 / 12 に丸める**（当初指定の 32/8 は 12 で割り切れない）。
  Stage 1 = 12 × micro 1 × accum 3 = **36**, 2 epoch。Stage 2 = 12 × 1 × 1 = **12**, 3 epoch。
  **micro-batch 3（accum なし）は XPU OOM**（cross-entropy が大語彙で `UR_RESULT_ERROR_OUT_OF_RESOURCES`）
  なので **micro 1 × accum 3** を使う。
- W&B: `report_to=wandb`（`WANDB_PROJECT=alpamayo15-carla-sft`, key は `.env`）。

## リポジトリ側の変更・追加

| ファイル | 変更/役割 |
| --- | --- |
| `scripts/aurora/build_nav_coc_annotations.py`(新規) | 1.5 nav SFT が読む annotation JSON `{clip_id,t0_relative,nav_text,[cot]}` を生成。`--mode cot`: CoC ラベル起点（Stage 1, cot 必須, 11,076）。`--mode grid`: `clip_index.event_t0s` グリッド起点（Stage 2, 全サンプル ~194,923; cot は ±0.6s 以内にあれば付与）。Alpamayo window(1.6s hist + 6.4s future) が clip に収まる segment のみ採用 |
| `.../alpamayo-recipes/src/alpamayo/data/pai_nav.py` | annotation の `cot` を sample へ転送（`nav_cot` プロセッサが cot を教師化できるように） |
| `.../recipes/alpamayo1_5_sft/configs/vla_processor/nav_cot.yaml`(新規) | Stage 1 用プロセッサ。`components_order=[image,traj_history,route,prompt,cot,traj_future]`, `label_components=[cot,traj_future]` |
| `scripts/aurora/convert_features_csv_pai022.py`(新規) | 旧 export の `features.csv` を physical_ai_av **0.2.2** スキーマ(`directory`/`chunk_path`/template 化 `clip_files_in_zip`)へ変換（データ zip は不変、原本は `features.csv.oldschema` に退避） |
| `scripts/aurora/setup_venv_alpamayo15.sh`(新規) | frameworks に 1.5 レシピの leaf 依存だけ足す venv(`outputs/alpamayo15_venv`)。torch/transformers/deepspeed は module。flash-attn は入れない(XPU は eager) |
| `scripts/aurora/prepare_ckpt_alpamayo15.sh`(新規) | `nvidia/Alpamayo-1.5-10B` を DL → `convert_checkpoint.py to-a1` で A1 形式へ（symlink + config 書換のみ、CPU） |
| `scripts/aurora/alpamayo15_xpu_entry.py`(新規) | mpiexec/xccl bootstrap → `alpamayo1_5_sft.train_hf`（R1 の `alpamayo_xpu_entry.py` と同型） |
| `scripts/aurora/sft_alpamayo15.pbs`(新規) | 1 ノード 12 タイルで Stage1/Stage2 を実行。A1 ckpt の `-xpu` view(attn=eager, vlm_name_or_path=ローカル Qwen) を作り offline ロード。`ALPAMAYO_AUTOCAST_DEVICE=xpu` を設定 |
| `.../alpamayo_r1/action_space/{utils.py,unicycle_accel_curvature.py}` | fp32 強制の autocast ガード `device_type="cuda"` を env `ALPAMAYO_AUTOCAST_DEVICE`(既定 cuda)で切替可能に。XPU では "cuda" 指定が無効で、Stage 2 の trajectory 復元(`cholesky_solve`)が bf16(rhs) vs fp32(L) で落ちるため `xpu` を指定して fp32 を復元 |

## 0. 事前準備（ログインノード, proxy 経由）

```bash
cd /lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research
# 1) venv（frameworks + 1.5 leaf 依存）
bash scripts/aurora/setup_venv_alpamayo15.sh    # -> outputs/alpamayo15_venv
# ※ import 検証はログインノードだと torch.xpu(デバイス0個)で SIGSEGV する。pip 導入自体は成功。
#    実 import 検証は計算ノード（下の probe）で行う。
# 2) checkpoint（gated, HF_TOKEN は .env）
bash scripts/aurora/prepare_ckpt_alpamayo15.sh  # -> outputs/checkpoints/Alpamayo-1.5-10B-A1-format
# 3) annotation JSON（2 種。Stage 1=cot 必須、Stage 2=cot 不要のため別ファイル）
module use /soft/modulefiles; module load frameworks
#  Stage 1b (nav_cot): CoC ラベル 1 個 = 1 サンプル。cot が教師なので cot 必須
python scripts/aurora/build_nav_coc_annotations.py --mode cot \
  --pai-dir data/carla_alpamayo_pai \
  --coc-dir outputs/coc_autolabel_aurora_full/coc_labels_8b \
  --out data/carla_alpamayo_pai/nav_coc_annotations.json   # -> 11,076 sample / 3,076 clip
#  Stage 1a / Stage 2 (nav): clip_index.event_t0s グリッド 1 点 = 1 サンプル。全クリップ・全サンプル。
#  cot はグリッド点の ±0.6s 以内に CoC ラベルがあれば付与（nav processor は無視、provenance 用）
python scripts/aurora/build_nav_coc_annotations.py --mode grid \
  --pai-dir data/carla_alpamayo_pai \
  --coc-dir outputs/coc_autolabel_aurora_full/coc_labels_8b \
  --out data/carla_alpamayo_pai/nav_grid_annotations.json   # -> ~194,923 sample / 4,459 clip
# 4) features.csv を physical_ai_av 0.2.2 スキーマへ（1 回だけ。原本は .oldschema に退避）
python scripts/aurora/convert_features_csv_pai022.py --pai-dir data/carla_alpamayo_pai
```

> **base VLM の注意**: A1 ckpt の config は `vlm_name_or_path=nvidia/Cosmos-Reason2-8B` だが、1.5 レシピは
> processor/tokenizer に `Qwen/Qwen3-VL-8B-Instruct` を使う（Cosmos-Reason2-8B は Qwen3-VL-8B ベースで
> 語彙互換）。`sft_alpamayo15.pbs` は `-xpu` view で `vlm_name_or_path` をローカル
> `outputs/models/Qwen3-VL-8B-Instruct` に差し替え、offline でロードする（10B ロード成功を確認済み）。

## 1. micro-batch の probe（debug キュー, 4 step）

本番前に、1 ノードで実 10B・実データ・本番 config のまま数 step だけ回して OOM とパイプラインを確認する。

```bash
# micro 3（accum なし）… OOM するはず（記録用）
qsub -q debug -l walltime=1:00:00 -o logs/aurora_sft_a15_probe_mb3.out \
  -v STAGE=1,BS1=3,ACCUM1=1,MAX_STEPS=4,SAVE_STRATEGY=no,REPORT_TO=none \
  scripts/aurora/sft_alpamayo15.pbs
# micro 1（本番と同じ per-step メモリ）… 通る
qsub -q debug -l walltime=1:00:00 -o logs/aurora_sft_a15_probe_mb1.out \
  -v STAGE=1,BS1=1,ACCUM1=1,MAX_STEPS=4,SAVE_STRATEGY=no,REPORT_TO=none \
  scripts/aurora/sft_alpamayo15.pbs
```

## 2. 本番（capacity キュー, Stage1a → Stage1b → Stage2）

```bash
qsub scripts/aurora/sft_alpamayo15.pbs            # STAGE=all（1a,1b,2 を順に）
tail -f logs/aurora_sft_alpamayo15.out
# 個別: 1a/1b/2。1b は 1a の checkpoint-*、2 は 1b の checkpoint-* を自動参照
qsub -v STAGE=1  scripts/aurora/sft_alpamayo15.pbs   # 1a→1b
qsub -v STAGE=1a scripts/aurora/sft_alpamayo15.pbs
qsub -v STAGE=1b scripts/aurora/sft_alpamayo15.pbs
qsub -v STAGE=2  scripts/aurora/sft_alpamayo15.pbs
# 連鎖の smoke（各フェーズ 4 step, checkpoint を保存して 1a→1b→2 の受け渡しを検証）:
qsub -q debug -l walltime=1:00:00 \
  -v MAX_STEPS=4,SAVE_STRATEGY=steps,SAVE_STEPS=4,REPORT_TO=none \
  scripts/aurora/sft_alpamayo15.pbs
```

既定値（`qsub -v KEY=VAL,...` で上書き可）:

```bash
STAGE=all  NPROC=12  PPN=12
BS1=1 ACCUM1=3 EPOCHS1A=2 EPOCHS1B=2   # Stage1 eff bs = 12*1*3 = 36
BS2=1 ACCUM2=1 EPOCHS2=3     # Stage2 eff bs = 12*1*1 = 12
WARMUP1A=500 WARMUP1B=50       # 1a は ~10k step, 1b は ~600 step
ATTN_IMPL=eager               # flash_attention_2 は CUDA 専用
PAI_DIR=.../data/carla_alpamayo_pai
NAV_ANNOTATIONS=.../data/carla_alpamayo_pai/nav_grid_annotations.json  # grid, ~194,923 (Stage 1a & 2)
COT_ANNOTATIONS=.../data/carla_alpamayo_pai/nav_coc_annotations.json   # cot,  11,076    (Stage 1b)
# ANNOTATIONS=... を明示すると両方を上書き（後方互換）
# ANNOTATIONS=... を明示すると両ステージ共通で上書き（後方互換）
TRAIN_CHUNKS=0-48             # 全 48 chunk
CKPT=.../outputs/checkpoints/Alpamayo-1.5-10B-A1-format
QWEN_DIR=.../outputs/models/Qwen3-VL-8B-Instruct
OUT_ROOT=.../outputs/alpamayo15_sft_aurora
```

出力:

```text
outputs/alpamayo15_sft_aurora/stage1/checkpoint-*/   (model.safetensors.index.json + shards)
outputs/alpamayo15_sft_aurora/stage2/checkpoint-*/
```

## トラブルシュート

- **`KeyError('directory')`（PAIDatasetWithNav instantiate 時）**: `features.csv` が旧スキーマ。
  §0-4 の `convert_features_csv_pai022.py` を実行する。
- **`UR_RESULT_ERROR_OUT_OF_RESOURCES`（forward の cross_entropy）**: micro-batch が大きすぎ。
  `BS1=1` にして `ACCUM1` で実効 batch を稼ぐ。
- **Stage 2 で `RuntimeError: Expected b and A to have the same dtype ...`（`cholesky_solve`）**:
  `ALPAMAYO_AUTOCAST_DEVICE=xpu` が効いていない（`sft_alpamayo15.pbs` が export 済み）。action_space の
  autocast ガードが XPU autocast を無効化できず bf16 が混ざるのが原因。
- **末尾の `|CCL_ERROR| internal_kvs_server ... Bad file descriptor`**: 学習完了後の xccl KVS
  teardown ノイズ。`=== Alpamayo 1.5 SFT done ===` が出ていれば正常終了（mpiexec は 0 で返る）。
- **ログインノードで import が SIGSEGV**: `torch.xpu`（デバイス 0 個）が原因。検証は計算ノードで。

## 動作確認状況（2026-07-12）

- [x] annotation JSON 生成: **11,076 sample / 3,076 clip**（58 segment は future window が clip 末尾を
  超過してスキップ、label 欠損 0）。nav_text 分布 = Follow the lane 7702 / Turn left 1972 / …。
- [x] `features.csv` を 0.2.2 スキーマへ変換し、CPU で interface + egomotion(Interpolator) +
  camera(SeekVideoReader) + 1 sample の full load（image `(4,4,3,600,800)`, nav_text, cot）を確認。
- [x] Hydra 合成をドライラン（torch 非依存）で検証。Stage1 は `label_components=[cot,traj_future]`,
  Stage2 は `[traj_future]`。`save_strategy` は `+` 追記、Stage2 に `vlm_name_or_path` は無い、を修正。
- [x] **10B checkpoint の XPU ロード成功**（eager attn, Cosmos/Qwen 語彙不整合なし）。12 rank xccl 初期化・
  DeepSpeed ZeRO-2 も起動。
- [x] **micro-batch probe**（debug, 4 step）: **mb3 は OOM**（cross_entropy）、**mb1 は成功**
  （loss 5.5→6.9→6.5→6.7, ~9s/step @ eff bs 12, `=== Alpamayo 1.5 SFT done ===`）。
- [x] **Stage 1 完走**（capacity, ジョブ `8668722`, eff bs 36, 2 epoch, 616 step, ~3h53m,
  **train_loss 6.7→2.68**）。`outputs/alpamayo15_sft_aurora/stage1/checkpoint-616` 保存。W&B
  `alpamayo15-carla-sft/alpamayo15_carla_stage1` へオンライン記録。
- [x] **Stage 2 の XPU dtype バグを修正**: 初回 Stage 2 は `cholesky_solve`(bf16 rhs vs fp32 L) で全 rank
  crash。`ALPAMAYO_AUTOCAST_DEVICE=xpu` で fp32 を復元。debug 4-step probe で loss が出て
  `=== Alpamayo 1.5 SFT done ===`（~2.4s/step @ eff bs 12）を確認。
- [x] **Stage 2 完走**（capacity, ジョブ `8669070`, `-l walltime=6:00:00`, eff bs 12, 3 epoch,
  2769 step, **train_runtime ~1h17m**, train_loss ~5.76 の diffusion loss）。dtype 修正後は cholesky
  エラーなし。`outputs/alpamayo15_sft_aurora/stage2/checkpoint-2769`（30GB, 5 shard + index）保存。W&B
  `alpamayo15_carla_stage2`。**これで 2 段 SFT 完了**。
  - メンテ枠（24h ジョブがブロックされる期間）では `-l walltime=6:00:00` 等で短く投げ直すと走る
    （Stage 2 実測 ~1.3h + ロード/保存）。
