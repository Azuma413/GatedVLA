# オンプレサーバでの残りデータ収集手順

Polarisでポイントが枯渇した残り **833ルート**（全3274本の25.4%）をオンプレのA6000 GPUサーバで収集する手順です。

**方針: 1TB級の既存収集データはオンプレに転送しません。** Polaris側で「未収集ルートの一覧」（約52KBのテキスト1枚）だけを確定・転送し、オンプレはそれを直接収集します。ルートXMLはgit管理下（3364本）なので、`git clone` 済みのオンプレには既に存在します。

## 前提条件

| 項目 | 内容 |
|------|------|
| GPU | NVIDIA RTX A6000（48GB）× 3 |
| CARLA | 0.9.15（バイナリ展開済み） |
| Python環境 | `.venv`（`lead` パッケージインストール済み） |
| リポジトリ | このリポジトリを `git clone`、サブモジュール済み |
| 収集済みデータ | **転送不要**（未収集リストのみ転送） |

---

## 0. 環境確認（オンプレ側）

```bash
cd /path/to/VLA4AD_Research
export REPO_DIR="$(pwd)"

# Pythonとleadが動くか確認
.venv/bin/python3 -c "import lead; print('lead OK')"

# CARLAバイナリのパスを確認（以下のどちらかが存在するはず）
ls 3rd_party/CARLA_0915/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
# または
ls 3rd_party/CARLA_0.9.15/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping

# GPUが3枚見えているか
nvidia-smi --list-gpus

# ルートXMLがclone済みか（3364本あるはず）
git ls-files data/data_routes/lead/ | grep -c '\.xml$'
```

CARLA バイナリが標準パスにない場合は後続のコマンドで `CARLA_BIN=` を明示します。

---

## 1. Polarisで未収集リストを確定・エクスポート（すでに実行済み）

既存の1TBクリップツリーをスキャンできるのはPolarisだけです。`export_missing_manifest.sh` が
`find_missing_routes.py` を1回だけ実行し、結果を小さなテキストに凍結します。

```bash
# Polaris側
cd /lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research
bash scripts/local/export_missing_manifest.sh
```

出力:

```
missing_manifest/
  missing_routes.txt    # 未収集ルートXMLパス一覧（833行, 約52KB）
  missing_summary.txt   # town別・シナリオ別の集計（人間用）
```

---

## 3. 未収集ルートの確認（オンプレ・ドライラン）

`MISSING_LIST=` に転送したリストを指定し、`LAUNCH=0`（デフォルト）で計画だけ表示します。
この経路では `find_missing_routes.py` は呼ばれず、1TBデータは不要です。

```bash
cd /path/to/VLA4AD_Research
MISSING_LIST=missing_manifest/missing_routes.txt \
  bash scripts/local/collect_missing_local.sh
```

出力例:

```
=== On-prem missing-route collection ===
MISSING_LIST  = missing_manifest/missing_routes.txt  (frozen; dataset NOT scanned)
CHUNK         = 2 routes/batch
N_WORKERS     = 3 (GPUs: 0..2)
QUALITY_LEVEL = Epic
WORK          = logs/local_collection/20260701_120000

Missing routes: 833

Time estimate per town (A6000, sequential):
  Town15    :  361 routes × 6.3 min = 37.9 h
  Town12    :  179 routes × 12.1 min = 35.9 h
  ...
Total sequential : 110.7 h (4.6 days)
With 3 workers  : 36.9 h (1.54 days)

Batches: 419  (2 routes/batch)
>> DRY-RUN. Re-run with LAUNCH=1 to start collection.
```

---

## 4. 収集の実行（オンプレ）

### 3-GPU フル並列（推奨）

```bash
MISSING_LIST=missing_manifest/missing_routes.txt \
  LAUNCH=1 bash scripts/local/collect_missing_local.sh
```

### GPU数を絞る場合

```bash
# GPU 1枚だけ使う
N_WORKERS=1 MISSING_LIST=missing_manifest/missing_routes.txt \
  LAUNCH=1 bash scripts/local/collect_missing_local.sh
```

### CARLAバイナリが標準パスにない場合

```bash
CARLA_BIN=/your/custom/path/CarlaUE4-Linux-Shipping \
  MISSING_LIST=missing_manifest/missing_routes.txt \
  LAUNCH=1 bash scripts/local/collect_missing_local.sh
```

---

## 4. ログとモニタリング

収集中は各ログファイルでリアルタイム確認できます。

```bash
WORK=logs/local_collection/<timestamp>   # 実際のWORKパスに置き換え

# 全体サマリ（完了・失敗の集計）
tail -f ${WORK}/summary.jsonl

# ワーカー0の進捗
tail -f ${WORK}/worker_00.log

# ワーカー0のCARLAサーバログ
tail -f ${WORK}/server_00.log

# ワーカー0のクライアントログ（step/frame進捗）
tail -f ${WORK}/client_00.log
```

オンプレで新規収集したclip数の確認（既存データは転送していないので0から増える）:

```bash
find data/carla_alpamayo_4c_py123d -name "sync.arrow" | wc -l
# 0 → 833 に近づけば完了（成功率に依存）
```

成功/失敗の集計:

```bash
.venv/bin/python3 -c "
import json, collections
c = collections.Counter(json.loads(l).get('status') for l in open('${WORK}/summary.jsonl') if l.strip())
print(dict(c))
"
```

---

## 5. 中断と再開

**Ctrl-C で安全に中断**できます。スクリプトが SIGINT をキャッチし、起動中の CARLA プロセスをすべて SIGKILL して終了します。

再開は `WORK=` に前回のディレクトリを指定し、**同じ `MISSING_LIST`** で `LAUNCH=1` 再実行します。
バッチXMLと各バッチの出力先（`data/outputs/<batch>/`）が固定されるため、leaderboardの
`checkpoint_endpoint.json` + `--resume` により**収集済みルートは自動スキップ**されます。

```bash
WORK=logs/local_collection/20260701_120000 \
  MISSING_LIST=missing_manifest/missing_routes.txt \
  LAUNCH=1 bash scripts/local/collect_missing_local.sh
```

> **注意:** オンプレには1TBデータがないため `find_missing_routes.py` は使えません。
> 再開時のスキップ判定は「凍結リストを再分割した同一バッチ + per-batch `--resume`」に依存します。
> したがって再開時は **`WORK` と `MISSING_LIST` を前回と同じにする**ことが重要です。

---

## 6. 完了確認

**未収集ルートの「特定」はステップ1でaurora側に確定済み**なので、オンプレでは
「凍結リストの833本すべてが成功収集できたか」だけを確認すれば完了です。
ルート単位の再特定（`find_missing_routes.py`）は不要です。

判定は `summary.jsonl` のバッチ成否で行います。

```bash
# バッチ成否の集計（ok 以外が残っていれば未完）
.venv/bin/python3 -c "
import json, collections
c = collections.Counter(json.loads(l).get('status') for l in open('${WORK}/summary.jsonl') if l.strip())
print(dict(c))
"

# 失敗バッチ一覧（あれば再収集対象）
cat ${WORK}/failed_batches.txt 2>/dev/null
```

CARLAはルートにより収集が失敗します（実測でTown03等は失敗率が高い）。
失敗バッチが残っていたら、**同じ `WORK` + `MISSING_LIST`** で再実行すれば
`--resume` により成功済みルートはスキップされ、失敗分だけが再収集されます。
これを `summary.jsonl` が全 `ok` になるまで繰り返せば、833本すべてが揃った状態になります。

```bash
# 失敗分の再収集（成功済みはスキップ）
WORK=logs/local_collection/<ts> \
  MISSING_LIST=missing_manifest/missing_routes.txt \
  LAUNCH=1 bash scripts/local/collect_missing_local.sh
```


## スクリプト構成

| ファイル | 実行場所 | 役割 |
|--------|--------|------|
| [scripts/local/export_missing_manifest.sh](scripts/local/export_missing_manifest.sh) | **Polaris** | 1TBクリップをスキャンし未収集リスト(52KB)を凍結出力 |
| [scripts/local/collect_missing_local.sh](scripts/local/collect_missing_local.sh) | オンプレ | 凍結リスト → バッチ生成 → 収集起動のオーケストレーター |
| [scripts/local/collect_local_parallel.sh](scripts/local/collect_local_parallel.sh) | オンプレ | N_WORKERS 並列 CARLA+client エンジン（ファイルベースキュー） |
| [scripts/tools/find_missing_routes.py](scripts/tools/find_missing_routes.py) | Polaris | 完全クリップのないルートを特定（フルデータスキャン） |
| [scripts/tools/build_route_batches.py](scripts/tools/build_route_batches.py) | 両方 | ルートをtown別バッチXMLにグループ化 |

### ログ構成

```
logs/local_collection/<timestamp>/
  batch_manifest.txt        # バッチXMLパス一覧
  routes_filtered.txt       # 未収集ルートパス一覧
  summary.jsonl             # 全バッチの成功/失敗レコード
  failed_batches.txt        # 失敗したバッチのパス
  worker_00.log             # ワーカー0の進捗ログ
  server_00.log             # ワーカー0のCARLAサーバログ
  client_00.log             # ワーカー0のクライアントログ
  carla_saved_00/           # ワーカー0のCARLA Savedディレクトリ
  ...（worker_01, 02 も同様）
```
