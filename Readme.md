# Gemma / PLaMo 3 NPU推論

モデルごとにコード・モデル・測定結果・テストを分離しています。以下のコマンドはリポジトリのルートで実行します。

```text
gemma/
  gemma_npu.py
  models/
  baselines/
  test_gemma.py
  README.md
plamo3/
  plamo_genai.py
  plamo_translate.py
  export_model.py
  benchmark_genai.py
  benchmark_npu.py
  models/
  baselines/
    investigations/
  test_app.py
  README.md
```

共通環境の準備:

```powershell
uv sync --python 3.12 --locked
```

Gemma 3 4B INT4 / OpenVINO GenAI / NPU:

```powershell
uv run python -X utf8 -m gemma.gemma_npu
```

PLaMo 3 8B INT4 / OpenVINO GenAI / NPU:

```powershell
uv run python -X utf8 -m plamo3.plamo_genai
```

FP16の演算グラフを選択する場合:

```powershell
uv run python -X utf8 -m plamo3.plamo_genai --graph-precision fp16
```

出力層もINT4にして速度を上げる場合（再量子化により訳文が変わる可能性があります）:

```powershell
uv run python -X utf8 -m plamo3.plamo_genai --int4-lm-head
```

このオプションの実測は22.91 tokens/sで、`main@61a1daf` の21.57 tokens/sから約6%改善しました。
既定モデルは維持し、別モデルとして保存します。[比較と検証範囲](plamo3/README.md#出力層の追加int4量子化)

生成上限は両方とも `--max-new-tokens`（既定256）で指定します。
実測はGemma 35.08 tokens/s、FP32グラフの最適化後のPLaMo 21.73 tokens/sです。PLaMoは同条件3回の中央値で、変更前の5.21 tokens/sから約4.17倍に改善しました。
モデルサイズと入力形式が異なり、PLaMoの計測はPython側のトークン化・復号を含みません。

テスト:

```powershell
uv run python -X utf8 -m unittest -v
```

詳細: [Gemma](gemma/README.md) / [PLaMo 3](plamo3/README.md)
