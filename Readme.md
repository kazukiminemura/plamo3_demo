# Gemma / PLaMo 3 のNPU推論

Intel NPUとOpenVINO GenAIで、Gemma 3 4Bの生成とPLaMo 3 Translate 8Bの翻訳を実行します。
Gemmaは変換済みモデルを取得します。PLaMoは元モデルを取得してINT4へ変換し、GenAI用のモデルを作成します。
モデル本体はGitに含まれていません。

## 1. 環境を準備する

WindowsのPowerShellで、**このREADMEがあるリポジトリのルート**からコマンドを実行してください。
Intel NPUと対応ドライバー、`uv`、モデル取得用のインターネット接続が必要です。

```powershell
uv sync --python 3.12 --locked
uv run plamo-translate devices
```

デバイス一覧に `NPU` が表示されることを確認します。OpenVINO / OpenVINO GenAIは2026.4.1に固定しています。

Gemmaを使う場合は、[取得元モデル](https://huggingface.co/OpenVINO/gemma-3-4b-it-int4-cw-ov)の利用条件への同意とHugging Faceへのログインが必要です。

```powershell
uv run hf auth login
uv run hf auth whoami
```

## 2. モデルをダウンロードする

使うモデルのコマンドを実行します。ダウンロードだけを行い、変換・推論は開始しません。

```powershell
# PLaMoだけ
uv run python -X utf8 download_models.py plamo3

# Gemmaだけ
uv run python -X utf8 download_models.py gemma

# 両方
uv run python -X utf8 download_models.py
```

完了時に `Downloaded: ...` と保存先が表示されます。中断した場合は同じコマンドで再実行してください。取得済みファイルを再利用します。
PLaMoの元重みは約17GBです。変換後のモデルも別途保存するため、追加のディスク容量が必要です。変換には推論より多くのメモリを使います。

| モデル | Hugging Faceリポジトリ | 固定リビジョン |
| --- | --- | --- |
| PLaMo | `pfnet/plamo-3-translate-8b` | `681bd55cf01f5412aab8881e6a4a276c948fabc1` |
| Gemma | `OpenVINO/gemma-3-4b-it-int4-cw-ov` | `5397b33a5fadf59466bf9f9690348ed8499f2b68` |

## 3. PLaMoを変換して実行する

### 初回: NPU用INT4モデルを作成する

ダウンロード完了後、次を実行します。

```powershell
uv run plamo-translate export --npu-optimized --weight-format int4 --group-size -1 --model-dir plamo3/models/plamo-npu-rotated-cw-int4
```

事前ダウンロードと同じ固定リビジョン・保存先の元モデルを使います。`--revision` の指定は不要です。
`Exported: ...` が表示されたら変換完了です。空でない出力先には上書きしません。
公式モデルのカスタムPython実装を `trust_remote_code=True` で読み込みます。

上の `--npu-optimized`、`--group-size -1`、`--model-dir` はGenAIで使うモデルを作るための指定です。
`export` の引数を省略すると、通常の未回転KVキャッシュモデルを別の保存先に作成します。

### 変換後: GenAIで日本語を英訳する

```powershell
# 既定の日本語サンプルを英訳
uv run python -X utf8 -m plamo3.plamo_genai

# 自分の文章を英訳
uv run python -X utf8 -m plamo3.plamo_genai "会議は来週の月曜日に変更されました。"
```

初回はNPU用INT4モデルからGenAI用のstateful IRを作成し、以降は再利用します。
ウォームアップ1回＋本測定1回を行い、訳文・TTFT・Decode速度を表示します。
結果は `plamo3/baselines/plamo3-genai-npu-*.json` に保存します。

既定はFP32の演算グラフで、主要重みはINT4です。生成上限は `--max-new-tokens`（既定256）で変更できます。
32トークン単位に切り上げた入力長と生成上限の合計を2,048トークン以内に収めてください。

### 別の設定を使う

```powershell
# FP16の演算グラフ。主要重みはINT4を維持
uv run python -X utf8 -m plamo3.plamo_genai --graph-precision fp16

# 出力層もINT4に再量子化。訳文が変わる場合があります
uv run python -X utf8 -m plamo3.plamo_genai --int4-lm-head
```

英日翻訳・ファイル入出力・対話モードは翻訳CLIを使います。このCLIはGenAIとは別の生成処理です。

```powershell
uv run plamo-translate translate "The meeting has been rescheduled for next Monday."
uv run plamo-translate translate "会議は来週の月曜日です。" --target-lang en
uv run plamo-translate translate --input input.txt --output translated.txt
uv run plamo-translate translate --target-lang en
```

## 4. Gemmaを実行する

```powershell
uv run python -X utf8 -m gemma.gemma_npu
uv run python -X utf8 -m gemma.gemma_npu "Translate into English: 会議は明日です。" --max-new-tokens 256
```

変換済みINT4モデルを使うため、エクスポート操作は不要です。
起動時にもダウンロード処理を呼び、取得済みファイルを再利用します。
ウォームアップ1回＋本測定1回を行い、結果を `gemma/baselines/gemma3-4b-npu-*.json` に保存します。

## 保存先と参照関係

以下はリポジトリのルートからの相対パスです。自動ダウンロード・読み込みのパスはコードの配置を基準に解決します。
`export --model-dir` に指定する相対パスは、コマンドを実行したディレクトリを基準に解決します。

| 用途 | 保存先 | 作成・使用する処理 |
| --- | --- | --- |
| PLaMo元モデル | `plamo3/models/.source/<revision>/` | ダウンロード → `export` |
| PLaMo NPU用INT4 | `plamo3/models/plamo-npu-rotated-cw-int4/` | 上記の `export` → GenAI変換・NPU翻訳CLI |
| PLaMo GenAI・FP32 | `plamo3/models/plamo-genai-int4/` | GenAIの既定設定で自動作成・使用 |
| PLaMo GenAI・FP16 | `plamo3/models/plamo-genai-fp16-int4/` | `--graph-precision fp16` |
| PLaMo GenAI・出力層INT4 | `plamo3/models/plamo-genai-head-int4/` | `--int4-lm-head`（FP32） |
| PLaMo GenAI・FP16＋出力層INT4 | `plamo3/models/plamo-genai-fp16-head-int4/` | 上の2オプションを併用 |
| PLaMo通常モデル | `plamo3/models/plamo-3-translate-8b-int4/` | 引数省略時の `export` → CPU/GPU翻訳CLIの既定 |
| Gemma | `gemma/models/gemma-3-4b-it-int4-cw-ov/` | ダウンロード → Gemma推論 |

PLaMoの `--model-dir` は**変換後の出力先**です。元モデルの取得先は変更しません。
GenAIは上表のNPU用INT4モデルを参照するため、出力先を変更したモデルには自動で切り替わりません。
NPU翻訳CLIはNPU用モデルの `plamo_openvino.json` があれば優先し、なければ通常モデルを参照します。

変換後のファイル確認:

```powershell
Test-Path plamo3/models/plamo-npu-rotated-cw-int4/plamo_openvino.json
Test-Path plamo3/models/plamo-npu-rotated-cw-int4/openvino_model.xml
Test-Path plamo3/models/plamo-npu-rotated-cw-int4/openvino_model.bin
```

3つとも `True` なら完了メタデータとモデルファイルが存在します。推論の成功は実行して確認してください。

## テストとベンチマーク

```powershell
uv run python -X utf8 -m unittest -v
uv run python -X utf8 -m plamo3.benchmark_genai --runs 3 --validate --output plamo3/baselines/genai-measured.json
```

ユニットテストは8Bの元重みを取得しません。公式実装との比較テストでは設定とPython実装を取得し、小型モデルを作って検証します。
ベンチマークには変換済みモデルとNPUが必要です。

保存済みの過去の測定では、PLaMo GenAIのDecode中央値は21.73 tokens/s、出力層INT4の別比較では22.91 tokens/s、Gemmaは35.08 tokens/sでした。
測定条件は異なります。これらは新しい環境での動作確認や速度を保証する値ではありません。
測定条件・結果・追加オプションは [PLaMoの詳細](plamo3/README.md) / [Gemmaの詳細](gemma/README.md) を参照してください。
