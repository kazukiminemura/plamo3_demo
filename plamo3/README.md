# PLaMo 3 Translate 8B / OpenVINO CUI

以下のコマンドはリポジトリのルートで実行します。初回は [ルートREADMEの準備・ダウンロード・変換手順](../Readme.md) を先に実行してください。Gemmaは隣の `gemma/` フォルダーに分離しています。

既定は **NPU・INT4・最大256生成トークン** です。
長さを調整するCLI引数は `--max-new-tokens` だけです。原文は自動判定し、既定では日本語へ翻訳します。

## 実行

```powershell
uv sync --python 3.12
uv run plamo-translate devices

# 省略時は --max-new-tokens 256
uv run plamo-translate translate "The meeting has been rescheduled for next Monday."

# 生成上限を変更
uv run plamo-translate translate "The meeting has been rescheduled for next Monday." --max-new-tokens 128

# 日本語から英語
uv run plamo-translate translate "会議は来週の月曜日に変更されました。" --target-lang en

# ファイル入力・出力
uv run plamo-translate translate --input input.txt --output translated.txt

# 対話モード（/quit または Ctrl+C で終了）
uv run plamo-translate translate --target-lang en
```

テキスト・UTF-8ファイル・標準入力に対応します。単発翻訳では訳文だけを標準出力に出し、読込状況・警告は標準エラーへ出します。
各翻訳のTTFT（最初の生成トークンまでの秒数）とDecode速度（2トークン目以降のtokens/s）も標準エラーへ表示します。モデル読込・コンパイル時間は含みません。生成数が不足して計測できない値は `N/A` になります。
生成は停止トークンに達すると終了します。`--max-new-tokens` は出力数を固定する設定ではなく上限です。

## INT4モデル

NPU用モデルの保存先は `plamo3/models/plamo-npu-rotated-cw-int4` です。モデル本体はGitに含まれないため、新しい環境では作成が必要です。
INT4対称・チャネル単位の重み圧縮を使い、Embeddingと最終出力層はNNCFの既定のINT8バックアップです。
過去のエクスポートではBINファイルは4,710,677,278バイトでした。アプリと `generation_config.json` の既定生成上限は256です。
NPUでは回転済みKVキャッシュと1トークンDecodeを使います。

新しい環境で同じモデルを作成する場合（初回のみ）:

```powershell
uv run python -X utf8 download_models.py plamo3
uv run plamo-translate export --npu-optimized --weight-format int4 --group-size -1 --model-dir plamo3/models/plamo-npu-rotated-cw-int4
```

既存のモデルは作り直す必要はありません。空でない変換先には上書きしません。
事前ダウンロードと変換は同じ固定リビジョン・`plamo3/models/.source/<revision>` を使います。
別リビジョンを変換する場合だけ `export --revision <ブランチまたはコミットID>` を指定します。
元重みの取得は約17GBで、変換には推論より多くのメモリを使います。
公式PFNリポジトリのカスタム実装を `trust_remote_code=True` で読み込みます。

## 内部コンテクストの自動設定

`--max-seq-len`、`--npu-context-length`、ベンチマークの `--context-length` は指定しません。
アプリがPFML3プロンプトを数え、入力と生成上限が収まる長さを32トークン単位で自動設定します。
たとえば入力26トークン・生成上限256なら内部長は288トークンです。
対話中は同じグラフを再利用し、必要な場合だけ大きいグラフをコンパイルします。

回転済みキャッシュの最適化モデルは、スライディングウィンドウ内の最大2,048トークンに対応します。
入力と生成上限の合計が制限を超える場合は、入力を分割するか `--max-new-tokens` を減らしてください。
モデルは `--model-dir`、デバイスは `--device CPU` / `--device GPU` で変更できます。
NPUでは最適化済みINT4モデルを優先し、CPU/GPUでは通常の未回転キャッシュIRを既定として使用します。

## 検証とベンチマーク

```powershell
uv run python -X utf8 -m unittest plamo3.test_app -v
uv run python -X utf8 -m plamo3.benchmark_npu --max-new-tokens 256
```

小型モデルで公式PyTorch実装とのlogits一致、キャッシュの位置情報、連続リクエストの初期化を検証します。
ベンチマークはウォームアップ1回＋3回測定し、TTFT・Decode・翻訳時間・訳文・モデルのSHA-256をJSON/CSVへ保存します。
入力には、240文字の段落を17回繰り返し、末尾に16文字の文を加えた4096文字の合成負荷を使います（翻訳品質の評価用ではありません）。
18分割し、同じコンパイル済みモデルを再利用します。生成上限256は各分割に適用されます。
ウォームアップは先頭の段落だけで行い、本測定は4096文字全体を翻訳します。
全文の処理文字数/秒と、各分割のTTFTを除いたDecode速度も記録・表示します。
4096文字の実測（ウォームアップ後1回）は355.722秒、11.515文字/秒、TTFT 0.701秒、Decode 4.428 tokens/sでした。
全18分割が停止トークンまで翻訳されました。160文字以下への分割も試しましたが、480文字の比較測定で11.200文字/秒となり、採用していません。
入力を増やすことによる速度向上は確認できていません。
内部長はアプリと同じ規則で自動設定します。

以前の128トークン固定長での測定（現在の自動設定・256生成トークンとは条件が異なります）:

| 方式 | Decode中央値 | 翻訳全体の生成速度中央値 |
| --- | ---: | ---: |
| 旧・接頭辞再計算 | 1.937トークン/秒 | 1.697トークン/秒 |
| 回転済みKV / CW INT4 | 5.456トークン/秒 | 4.392トークン/秒 |

単一の日英翻訳文を用いた測定で、15トークン/秒は未達です。

現在のNPUコンパイラーでは融合Attentionのキャッシュ出力でバッファ割り当てエラーが出るため、
最適化IRのAttentionをMatMul・Softmaxに分解しています。NPU指定時にCPU/GPUへ自動では切り替えません。
INT4圧縮により原モデルと訳文が変わる場合があります。

## 参照

PLaMo 3 INT4をOpenVINO GenAIのNPUパイプラインで試す場合:

```powershell
uv run python -X utf8 -m plamo3.plamo_genai
```

FP16の演算グラフでNPU推論する場合:

```powershell
uv run python -X utf8 -m plamo3.plamo_genai --graph-precision fp16
```

主要重みはINT4のまま、FP32へのグラフ統一を省略したモデルを `plamo3/models/plamo-genai-fp16-int4` に作成します。
元のモデルでFP32を使う正規化処理と、GenAIが要求するFP32のlogits出力は維持します。
既定のFP32グラフとは別に保存します。再測定は `plamo3.benchmark_genai --graph-precision fp16` でも選択できます。
FP16モードの実機確認（ウォームアップ後1回）はTTFT 0.332秒、Decode 5.50 tokens/sで、停止トークンまで翻訳できました。
[FP16測定結果](baselines/plamo3-genai-npu-20261106T110314+0900.json)

初回は既存の `plamo3/models/plamo-npu-rotated-cw-int4` の重みを使い、GenAI用のstateful IRを `plamo3/models/plamo-genai-int4` に作成します。
入力のトークン化・出力の復号には公式Pythonトークナイザーを使い、生成ループとKVキャッシュ管理はGenAIの `LLMPipeline` に任せます。
既定の240文字を日英翻訳し、ウォームアップ1回＋本測定1回の結果を `plamo3/baselines/plamo3-genai-npu-*.json` に保存します。
生成上限は `--max-new-tokens`（既定256）です。入力と生成上限の合計は、このIRでは2048トークン以内に収めます。
GenAIの計測値はPython側のトークン化・復号時間を含みません。

- [公式モデル・PFML3仕様](https://huggingface.co/pfnet/plamo-3-translate-8b)
- [PLaMo Community License](https://huggingface.co/pfnet/plamo-3-translate-8b/tree/main/LICENSE)
- [OpenVINO GQA形式に関する報告](https://github.com/openvinotoolkit/openvino/issues/39385)

モデルの利用条件は公式のライセンスを参照してください。

## GenAI NPUの高速化

INT4重みを保ったまま演算グラフの型をFP32に統一し、出力端の不要なSliceを除去しています。
既存のadapter version 3モデルは、次回実行時にversion 4へ自動変換されます。再ダウンロード・再量子化は不要です。

同じNPU、同じ240文字・入力110トークン、生成上限256、ウォームアップ1回＋本測定3回の中央値:

| 指標 | 変更前 | 変更後 |
| --- | ---: | ---: |
| Decode | 5.214 tokens/s | **21.728 tokens/s** |
| TTFT | 0.333秒 | **0.193秒** |
| 生成全体 | 17.402秒 | **4.288秒** |

速度は約4.17倍です。3回とも同じトークン列で停止トークンまで生成し、変更前ともトークン列が一致しました。
追加の日英・英日4文でも、数値と日付を含めて最後まで翻訳できました。モデル読込・コンパイルとPython側のトークン化・復号は計測に含めません。

[変更前の3回測定](baselines/genai-before-optimization.json) / [変更後の3回測定と追加翻訳](baselines/genai-after-optimization.json)

切り分けでは、出力Sliceだけの除去で6.51 tokens/s、FP32の演算グラフと組み合わせた探索測定で18.88 tokens/sを確認しました。
MLP投影の分割（5.23 tokens/s）とEmbedding行選択の移動（5.26 tokens/s）は改善せず、採用していません。これらの探索値は本測定1回であり、上表の最終比較はそれぞれ3回の中央値です。

探索測定の記録は [baselines/investigations/](baselines/investigations/) にまとめています。
元の重みは `plamo3/models/.source/<revision>/` に保存し、再エクスポート時に再利用します。削除した場合は再ダウンロードが必要です。変換済みモデルでの推論には元の重みは使いません。

不要なSliceは、[OpenVINO 2026.4.1の出力層分離処理](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp)のパターン認識を妨げていました。
主な改善はFP16からFP32へのIR表現の統一で得られました。この環境でのコンパイラー最適化との相性を実測で切り分けた結果であり、一般にFP32計算がFP16計算より速いという意味ではありません。
カーネル別プロファイルは有効化時のコンパイルが完了せず中断したため、個々のNPU演算の時間内訳は未確定です。

再測定:

```powershell
uv run python -X utf8 -m plamo3.benchmark_genai --runs 3 --validate --output plamo3/baselines/genai-measured.json
```

変更前の初回GenAI測定: [TTFT 0.335秒、Decode 5.54 tokens/s](baselines/plamo3-genai-npu-20261105T222234+0900.json)

## 出力層の追加INT4量子化

`main` のコミット `61a1daf62853065b943ae541c703a5feda3723c1` を基準に、追加の最適化を測定しました。
基準モデルの出力層はINT8です。`--int4-lm-head` はこの出力層だけを対称・チャネル単位のINT4へ再量子化します。
既存の128個のINT4重み（3,825,205,248バイト）はSHA-256で一致し、変更していません。
埋め込みはINT8のままで、モデル全体のすべての重みがINT4になるわけではありません。

```powershell
uv run python -X utf8 -m plamo3.plamo_genai --int4-lm-head
```

FP32グラフの追加モデルは `plamo3/models/plamo-genai-head-int4/` に保存します。
元の重みキャッシュの再ダウンロードは不要で、既存のINT4エクスポートから作成します。
フラグを省略すると従来のINT8出力層を使います。

同じ240文字・入力110トークン・生成上限256、ウォームアップ1回＋本測定5回の中央値:

| 指標 | main基準 | 基準モデルの再測定 | 出力層INT4 |
| --- | ---: | ---: | ---: |
| Decode | 21.571 tokens/s | 21.598 tokens/s | **22.910 tokens/s** |
| TTFT | 0.176秒 | 0.178秒 | 0.175秒 |

初回基準比6.2%、再測定比6.1%の改善です。IRの重みファイルは4,715,129,625バイトから4,494,821,145バイトになりました。
モデル読込・コンパイルとPython側のトークン化・復号は上表に含めません。
環境はOpenVINO/GenAI 2026.4.1、Intel AI Boost（アーキテクチャ5010）、コンパイラー8.4です。

これは数値を変える再量子化で、訳文の完全一致を保つ最適化ではありません。
確認した5文のうち3文で出力トークンが変わりました。例えば `Please save this file and then close the window.` が
`Please save this file before closing the window.` になりました。5文とも停止トークンまで生成し、請求額12,345円と期限2026年11月30日も保持しました。
広い翻訳コーパスでの品質評価は未実施なので、品質を優先する場合は既定モデルを使ってください。

- [mainの基準測定](baselines/main-61a1daf-baseline.json) / [基準モデルの再測定](baselines/main-61a1daf-recheck.json)
- [実装した変換処理による追加モデルの測定](baselines/optimized-head-int4.json)
- [出力比較と速度比較](baselines/optimization-61a1daf-comparison.json) / [既存重みの保持確認](baselines/head-int4-weight-verification.json)
- [FP16の追加基準測定](baselines/main-61a1daf-fp16-baseline.json)

再測定:

```powershell
uv run python -X utf8 -m plamo3.benchmark_genai --runs 5 --validate --output plamo3/baselines/recheck.json
uv run python -X utf8 -m plamo3.benchmark_genai --runs 5 --validate --int4-lm-head --output plamo3/baselines/head-int4-measured.json
```

コンパイラーの動的量子化、実行要求の展開、最適化レベル、latency設定、正規化の精度、Turbo、恒等Convertの除去も比較しましたが、
基準のDecode速度を明確に上回る改善は確認できず、既定設定には採用していません。
行列積だけをFP32にしたFP16グラフも改善しませんでした。
[探索記録と使用設定](baselines/investigations/main-61a1daf.json) に全測定をまとめています。
NPUコンパイラーのソースやシステムの電源設定は変更していません。
