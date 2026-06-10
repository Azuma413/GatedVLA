# CARLAデータ収集からAlpamayo SFTまで

この手順では、LEADでCARLAデータをAlpamayo/Physical AI AV互換の4カメラ構成で収集し、PAI風ローカル形式へexportしてからAlpamayo-R1をSFTします。CoCラベルは生成しません。

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

## 1. 123D形式でCARLAデータを収集

### 計算ノードでCARLAを起動する場合

CARLA単体の描画疎通は以下で確認します。

```bash
qsub scripts/polaris/carla_render_diagnostics.pbs
tail -f logs/polaris_carla_render_diagnostics.out
```

成功時はCARLA world名とRGB画像サイズが出ます。

```text
world Carla/Maps/Town10HD_Opt
rgb 320 240 ...
```

### 短いルートを確認する場合

現状のPolaris smokeは、マップ切替を避けるため `data/data_routes/lead/noScenarios/route_town10hd_smoke.xml` を使います。

```bash
qsub scripts/polaris/collect_py123d_smoke.pbs
tail -f logs/polaris_collect_py123d_smoke.out
```

成功時は `data/polaris_smoke/carla_alpamayo_4c_py123d/logs/carla_train/.../` 以下に、4カメラとLiDARなどのArrowファイルが生成されます。

```text
camera.ftcam_l0.arrow
camera.ftcam_f0.arrow
camera.ftcam_r0.arrow
camera.ftcam_tele_f0.arrow
lidar.lidar_top.arrow
ego_state_se3.arrow
sync.arrow
```

このsmokeは `py123d_rgb_only_smoke=true` の軽量構成です。RGB 4カメラ、LiDAR、IMU、GNSS、speedでPy123D保存まで確認します。

### SLURMで並列収集する場合

未確認です。PolarisではまずPBS用の短いsmokeだけ確認済みです。

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
