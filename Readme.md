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
  benchmark_npu.py
  models/
  baselines/
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

生成上限は両方とも `--max-new-tokens`（既定256）で指定します。
実測はGemma 35.08 tokens/s、最適化後のPLaMo 21.73 tokens/sです。PLaMoは同条件3回の中央値で、変更前の5.21 tokens/sから約4.17倍に改善しました。
モデルサイズと入力形式が異なり、PLaMoの計測はPython側のトークン化・復号を含みません。

テスト:

```powershell
uv run python -X utf8 -m unittest -v
```

詳細: [Gemma](gemma/README.md) / [PLaMo 3](plamo3/README.md)
