# Gemma 3 4B INT4 / OpenVINO GenAI / NPU

以下のコマンドはリポジトリのルートで実行します。

```powershell
uv run python -X utf8 -m gemma.gemma_npu
uv run python -X utf8 -m gemma.gemma_npu --max-new-tokens 256
```

初回は[公式モデル](https://huggingface.co/OpenVINO/gemma-3-4b-it-int4-cw-ov)を取得します。利用条件への同意とHugging Faceのログインが必要です。

```powershell
uv run hf auth login
```

モデルは `gemma/models/gemma-3-4b-it-int4-cw-ov`、測定結果は `gemma/baselines/` に保存します。
既定は240文字の日本語を英訳する指示です。入力指示は位置引数で変更できます。
ウォームアップ1回＋本測定1回を行い、TTFT・Decode速度・訳文を記録します。

実測はTTFT 0.149秒、Decode 35.08 tokens/sでした。
[測定結果](baselines/gemma3-4b-npu-20261105T220532+0900.json)

Gemma側のテスト:

```powershell
uv run python -X utf8 -m unittest gemma.test_gemma -v
```
