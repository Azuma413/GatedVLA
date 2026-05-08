# 1. データ収集
alpamayo-carla-bridge (3rd_party/alpamayo-carla-bridge) を確認し、alpamayo (3rd_party/alpamayo) の入力形式でCarlaデータを収集する。
データ収集パイプラインはLEADに準拠し、サーバ上で並列実行する。 
123D (3rd_party/py123d) はalpamayoが採用しているPhysical AI AV (PA AV) 形式と、LEAD形式を共通の123D形式に変換する機能を持っているが、123D形式からPA AV形式に変換する実装はない。
一方で、LEADには123D形式で書き出す機能がある。
つまり、LEAD側で123D形式でデータ収集し、その後、py123dに123D形式からPA AV形式にexportする機能を実装するのが良い。
alpamayoのSFTにCoCラベルは必須ではないが、性能を出すためには必要かもしれない。
LEADにCoCラベルを付ける機能はないので、SimLingoのcommentary生成コード (3rd_party/ref_simlingo) を参考に、これも実装すべき。
# 2. alpamayo SFT・評価
収集したCarlaデータでalpamayoをSFTし、LEADで評価する。
alpamayo-carla-bridgeを参考に、評価コードを実装する。 
# 3. 独自モデル実装・学習・評価
alpamayoの評価に基づき、改善案を考え、独自モデルを実装する。
同一データセットで独自モデルを学習し、alpamayoに対し性能向上が見られるかを確認する。
HY-Embodied0.5 MoT-2Bベースで、Latent CoTを導入する。
Latent CoTはLeWorldModelに基づく1 Token分の潜在状態とし、それを推論と同期させた状態でAR出力していく。
Flow側はStreaming Flow Policyで実装し、VLMの10倍の周波数で動作させる。
VLM特徴量はGr00t-N1.7と同様に、Flow側のDiTの偶数番目のTransformer層にcross-attentionする。
DiTは独自のImage Encoderを持つ。