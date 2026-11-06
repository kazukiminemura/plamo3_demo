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

モデルの事前ダウンロード（変換・推論は行いません）:

```powershell
# 両方を取得。片方だけなら末尾に plamo3 または gemma を指定
uv run python -X utf8 download_models.py
uv run python -X utf8 download_models.py plamo3
uv run python -X utf8 download_models.py gemma
```

取得済みのファイルは再利用し、中断した場合も同じコマンドで再実行できます。
Gemmaは利用条件への同意と `uv run hf auth login` が必要です。
Gemmaの保存先は `gemma/models/gemma-3-4b-it-int4-cw-ov` です。
PLaMoは約17GBの元モデルを `plamo3/models/.source/<revision>` に取得します。
PLaMoは取得だけでは推論できません。初回は次のコマンドでINT4へ変換してください（取得済みの元モデルを再利用します）。

```powershell
uv run plamo-translate export --npu-optimized --weight-format int4 --group-size -1 --revision 681bd55cf01f5412aab8881e6a4a276c948fabc1 --model-dir plamo3/models/plamo-npu-rotated-cw-int4
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

生成上限は両方とも `--max-new-tokens`（既定256）で指定します。
実測はGemma 35.08 tokens/s、FP32グラフの最適化後のPLaMo 21.73 tokens/sです。PLaMoは同条件3回の中央値で、変更前の5.21 tokens/sから約4.17倍に改善しました。
モデルサイズと入力形式が異なり、PLaMoの計測はPython側のトークン化・復号を含みません。

テスト:

```powershell
uv run python -X utf8 -m unittest -v
```

詳細: [Gemma](gemma/README.md) / [PLaMo 3](plamo3/README.md)
