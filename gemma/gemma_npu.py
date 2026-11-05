"""Run the official Gemma 3 4B INT4 model on NPU with OpenVINO GenAI."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

MODEL_ID = "OpenVINO/gemma-3-4b-it-int4-cw-ov"
REVISION = "5397b33a5fadf59466bf9f9690348ed8499f2b68"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "gemma-3-4b-it-int4-cw-ov"
DEFAULT_PROMPT = (
    "Translate the following Japanese text into English. Output only the translation.\n\n"
    "来週の月曜日に予定している開発会議では、翻訳アプリケーションの性能と使いやすさについて確認します。"
    "参加者は、これまでに見つかった問題点と改善案を金曜日までに共有してください。"
    "新しいバージョンでは、文章を翻訳した後に、最初のトークンが生成されるまでの時間と、その後の生成速度が表示されます。"
    "短い文章だけでなく、複数の文からなる案内や業務上の連絡も使い、訳文が最後まで自然につながることを確認してください。"
    "測定結果には使用したモデルと入力文を記録し、同じ条件で比較できるようにします。"
)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1以上を指定してください")
    return number


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Gemma 3 4B INT4をNPUで推論・計測")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=positive, default=256)
    args = parser.parse_args(argv)

    # Use the standard HTTP downloader on this machine.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    import openvino as ov
    import openvino_genai as genai

    try:
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                          ignore_patterns=[".gitattributes"])
    except GatedRepoError:
        print(f"取得権限が必要です: https://huggingface.co/{MODEL_ID}\n"
              "利用条件に同意後、uv run hf auth login を実行してください。", file=sys.stderr)
        return 1

    config = {"MAX_PROMPT_LEN": 256, "MIN_RESPONSE_LEN": args.max_new_tokens,
              "GENERATE_HINT": "BEST_PERF"}
    print("Gemma 3 4B INT4をNPU用にコンパイル中 ...", file=sys.stderr, flush=True)
    started = perf_counter()
    pipe = genai.VLMPipeline(MODEL_DIR, "NPU", **config)
    initialization_seconds = perf_counter() - started
    print("ウォームアップ中 ...", file=sys.stderr, flush=True)
    pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
    result = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(result.texts[0])
    metrics = result.perf_metrics
    report = {
        "model_id": MODEL_ID, "revision": REVISION, "model_dir": str(MODEL_DIR),
        "device": "NPU", "npu": ov.Core().get_property("NPU", "FULL_DEVICE_NAME"),
        "openvino": ov.__version__, "openvino_genai": genai.__version__,
        "pipeline_config": config, "max_new_tokens": args.max_new_tokens,
        "initialization_seconds": initialization_seconds,
        "warmup_runs": 1, "measured_runs": 1, "prompt": args.prompt,
        "output_text": result.texts[0],
        "input_tokens": metrics.get_num_input_tokens(),
        "generated_tokens": metrics.get_num_generated_tokens(),
        "ttft_seconds": metrics.get_ttft().mean / 1000,
        "decode_tokens_per_second": metrics.get_throughput().mean,
        "generation_seconds": metrics.get_generate_duration().mean / 1000,
    }
    print(f"TTFT: {report['ttft_seconds']:.3f} s | "
          f"Decode: {report['decode_tokens_per_second']:.3f} tokens/s", file=sys.stderr)
    now = datetime.now(timezone(timedelta(hours=9)))
    output = Path(__file__).resolve().parent / "baselines" / f"gemma3-4b-npu-{now:%Y%m%dT%H%M%S%z}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}", file=sys.stderr)
    if report["generated_tokens"] >= args.max_new_tokens:
        print("警告: 生成上限に達したため、出力が途中の可能性があります。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
