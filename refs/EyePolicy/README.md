# EyePolicy
頭の視線移動は別。
自然移動にはタスクに応じて意識的に大きく動くものと、視線のズレを補正するために細かく動くもののふたつがある。
細かい動きは眼球運動。

refs/SeeLess_SpecifyMoreが重要
See lessの部分を言語的なコンテキストではなく、VLA内のVLMの出力Hiddenに置き換えることは可能か
また、その時に、soft maskをこの論文のように自己教師、教師なしで学習することは可能か

## Setup
```bash
git clone --recurse-submodules https://github.com/Azuma413/EyePolicy.git && cd EyePolicy
uv sync
```

既存の clone では `git submodule update --init --recursive` を実行してから `uv sync` してください。
`libs/lerobot` は公式 upstream の `v0.6.1` タグに固定しています。

## ルール
policyは src/eye_policy に実装する
src/lerobot_policy_register を用いてlerobotにpolicyを登録する
論文や参照実装は refs/ に格納する．
実験/実装レポートは reports/ に格納する．
外部ライブラリは libs/ に格納する．
