"""Measure NPU translation with automatic context sizing. Run with uv run python -X utf8."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

from .plamo_translate import DEFAULT_MAX_NEW_TOKENS, Translator, context_for_text, default_model_dir, positive


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_metrics(samples, elapsed):
    """Aggregate independent chunks without counting their TTFT as decode time."""
    emitted = sum(sample["generated_tokens"] for sample in samples)
    decode_tokens = sum(max(0, sample["generated_tokens"] - 1) for sample in samples)
    decode_seconds = sum((sample["generated_tokens"] - 1) / sample["decode_tokens_per_second"]
                         for sample in samples if sample["decode_tokens_per_second"])
    return {
        "prompt_tokens": sum(sample["prompt_tokens"] for sample in samples),
        "generated_tokens": emitted,
        "inference_calls": sum(sample["inference_calls"] for sample in samples),
        "inference_seconds": sum(sample["inference_seconds"] for sample in samples),
        "translation_seconds": elapsed,
        "first_token_seconds": samples[0]["first_token_seconds"],
        "tokens_per_second": emitted / elapsed,
        "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else None,
        "decode_seconds": decode_seconds,
        "stopped_on_eos": all(sample["stopped_on_eos"] for sample in samples),
    }


def main():
    args_parser = argparse.ArgumentParser(description="NPU翻訳ベースラインをJSON/CSVに保存")
    args_parser.add_argument("--model-dir", type=Path)
    args_parser.add_argument("--runs", type=positive, default=3)
    args_parser.add_argument("--max-new-tokens", type=positive, default=DEFAULT_MAX_NEW_TOKENS)
    args_parser.add_argument("--no-kv-cache", action="store_true", help="旧方式との比較用")
    args_parser.add_argument("--kv-cache", action="store_true", help="出力検証が必要な実験版")
    args_parser.add_argument("--bucketed", action="store_true", help="短文用32/64トークングラフも使用")
    args_parser.add_argument("--validate", action="store_true", help="計測後に日英・英日の3文を追加検証")
    args_parser.add_argument("--output-dir", type=Path)
    args = args_parser.parse_args()
    args.model_dir = args.model_dir or Path(default_model_dir("NPU"))
    metadata = json.loads((args.model_dir / "plamo_openvino.json").read_text(encoding="utf-8"))
    use_kv_cache = (args.kv_cache or metadata.get("cache_mode") == "rotated") and not args.no_kv_cache and not args.bucketed

    import openvino as ov
    from transformers import AutoTokenizer

    text = (
        "来週の月曜日に予定している開発会議では、翻訳アプリケーションの性能と使いやすさについて確認します。"
        "参加者は、これまでに見つかった問題点と改善案を金曜日までに共有してください。"
        "新しいバージョンでは、文章を翻訳した後に、最初のトークンが生成されるまでの時間と、その後の生成速度が表示されます。"
        "短い文章だけでなく、複数の文からなる案内や業務上の連絡も使い、訳文が最後まで自然につながることを確認してください。"
        "測定結果には使用したモデルと入力文を記録し、同じ条件で比較できるようにします。"
    )
    # Synthetic throughput workload: seventeen complete copies plus a closing sentence.
    chunks = [text] * 17 + ["以上の内容を再確認してください。"]
    text = "".join(chunks)
    assert len(text) == 4096
    planning_started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    args.context_length = max(context_for_text(tokenizer, chunk, "ja", "en", args.max_new_tokens,
                                              metadata["context_length"]) for chunk in chunks)
    planning_seconds = perf_counter() - planning_started
    now = datetime.now(timezone(timedelta(hours=9)))
    output = args.output_dir or Path(__file__).resolve().parent / "baselines" / f"npu-context{args.context_length}-{now:%Y%m%dT%H%M%S%z}"
    output.mkdir(parents=True, exist_ok=False)
    core = ov.Core()
    driver = None
    if "NPU_DRIVER_VERSION" in core.get_property("NPU", "SUPPORTED_PROPERTIES"):
        driver = str(core.get_property("NPU", "NPU_DRIVER_VERSION"))
    result = {
        "status": "initializing", "started_at": now.isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "openvino": ov.__version__, "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
            "npu": core.get_property("NPU", "FULL_DEVICE_NAME"), "npu_driver_version": driver,
            "npu_plugin": {name: version.build_number for name, version in core.get_versions("NPU").items()},
        },
        "model": metadata,
        "files": {},
        "settings": {"device": "NPU", "context_tokens": args.context_length,
                     "source_lang": "ja", "target_lang": "en", "input_text": text,
                     "input_characters": len(text), "max_new_tokens": args.max_new_tokens,
                     "chunk_characters": [len(chunk) for chunk in chunks],
                     "workload": "Synthetic 4096-character document: repeated 240-character paragraph plus closing sentence",
                     "warmup_runs": 1, "measured_runs": args.runs,
                     "decoding": "greedy", "kv_cache": use_kv_cache, "openvino_compile_cache": False,
                     "attention_layout": "rank4_repeated_kv_heads",
                     "bucketed": args.bucketed,
                     "compile_config": {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "NPU",
                                        "NPUW_ONLINE_PIPELINE": "REP", "NPUW_FOLD": "YES"}},
        "method": {
            "translation_seconds": "Tokenization, inference, greedy decoding and detokenization; excludes initialization.",
            "first_token_seconds": "From translation start through the first emitted token's argmax; no streaming IO.",
            "ttft_seconds": "Alias of first_token_seconds; excludes model initialization and compilation.",
            "tokens_per_second": "Emitted tokens (excluding EOS) / translation_seconds; EOS inference is included in time.",
            "decode_tokens_per_second": "Sum of per-chunk emitted tokens after the first / summed per-chunk decode_seconds; excludes TTFT and EOS.",
            "decode_seconds": "Sum of per-chunk time between first and last emitted tokens; excludes TTFT and EOS.",
            "decode_ms_per_token": "1000 / decode_tokens_per_second; mean interval after the first token, excluding EOS.",
            "compile_seconds": "One compile in a new process with no configured OpenVINO blob cache; driver caches are uncontrolled.",
            "scope": "Synthetic 4096-character Japanese document, sequential chunks, batch=1; throughput only, not a translation quality evaluation.",
            "warmup": "First chunk only; measured runs translate the complete 4096-character document.",
            "input_characters_per_second": "Complete document input characters / translation_seconds; excludes initialization.",
        },
        "warmup": None, "runs": [],
    }

    def save():
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "result.json")

    save()
    try:
        for path in [args.model_dir / "openvino_model.xml", args.model_dir / "openvino_model.bin",
                     Path(__file__).resolve().with_name("plamo_translate.py"), Path(__file__).resolve().parent.parent / "uv.lock"]:
            result["files"][str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        translator = Translator(args.model_dir, "NPU", args.context_length, use_kv_cache, args.bucketed, tokenizer=tokenizer)
        result["initialization"] = translator.load_metrics
        result["initialization"]["context_planning_seconds"] = planning_seconds
        result["settings"]["compile_config"] = translator.compile_config
        result["execution_devices"] = str(translator.compiled.get_property("EXECUTION_DEVICES"))
        if translator.npu_kv_cache:
            result["decode_execution_devices"] = str(translator.decode_compiled.get_property("EXECUTION_DEVICES"))
        result["status"] = "measuring"
        save()
        for index in range(args.runs + 1):
            started = perf_counter()
            chunk_samples = []
            outputs = []
            run_chunks = chunks[:1] if index == 0 else chunks
            for chunk_index, chunk in enumerate(run_chunks, 1):
                outputs.append(translator.translate(chunk, "ja", "en", args.max_new_tokens))
                chunk_samples.append({"input_characters": len(chunk), "output_text": outputs[-1],
                                      **translator.last_metrics})
                print(f"{'warmup' if index == 0 else f'run {index}'} chunk {chunk_index}/{len(run_chunks)}: "
                      f"{translator.last_metrics['translation_seconds']:.3f}s", flush=True)
                if not translator.last_metrics["stopped_on_eos"]:
                    raise RuntimeError("翻訳が出力上限に達しました。--max-new-tokensを増やしてください。")
            elapsed = perf_counter() - started
            sample = {"run": index, "output_text": "\n".join(outputs), "chunks": chunk_samples,
                      "input_characters": sum(map(len, run_chunks)),
                      "input_characters_per_second": sum(map(len, run_chunks)) / elapsed,
                      "wall_seconds": elapsed, **document_metrics(chunk_samples, elapsed)}
            sample["ttft_seconds"] = sample["first_token_seconds"]
            speed = sample["decode_tokens_per_second"]
            sample["decode_ms_per_token"] = 1000 / speed if speed else None
            if index == 0:
                result["warmup"] = sample
            else:
                result["runs"].append(sample)
            save()
            if not sample["stopped_on_eos"]:
                raise RuntimeError("翻訳が出力上限に達したため、この実行を有効なベースラインにしません。")
            print(f"{'warmup' if index == 0 else f'run {index}'}: "
                  f"{sample['translation_seconds']:.3f}s, {sample['tokens_per_second']:.3f} tokens/s, "
                  f"TTFT {sample['ttft_seconds']:.3f}s, Decode {speed:.3f} tokens/s, "
                  f"{sample['input_characters_per_second']:.3f} chars/s", flush=True)
        metrics = ["translation_seconds", "first_token_seconds", "ttft_seconds", "tokens_per_second",
                   "decode_tokens_per_second", "decode_seconds", "decode_ms_per_token", "inference_seconds",
                   "input_characters_per_second"]
        result["summary"] = {
            key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for key in metrics if (values := [run[key] for run in result["runs"] if run[key] is not None])
        }
        result["outputs_identical"] = len({run["output_text"] for run in result["runs"]}) == 1
        result["all_stopped_on_eos"] = all(run["stopped_on_eos"] for run in result["runs"])
        fields = ["run", "prompt_tokens", "generated_tokens", "inference_calls", "translation_seconds",
                  "first_token_seconds", "ttft_seconds", "tokens_per_second", "decode_tokens_per_second",
                  "decode_seconds", "decode_ms_per_token",
                  "inference_seconds", "wall_seconds", "stopped_on_eos", "output_text",
                  "input_characters", "input_characters_per_second"]
        with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["runs"])
        if args.validate:
            result["validation_runs"] = []
            for source, source_lang, target_lang in [
                ("明日の午前十時に駅で待ち合わせましょう。", "ja", "en"),
                ("The server is down. Please restart the application.", "en", "ja"),
                ("このファイルを保存してから、画面を閉じてください。", "ja", "en"),
            ]:
                translated = translator.translate(source, source_lang, target_lang, args.max_new_tokens)
                result["validation_runs"].append({"source": source, "source_lang": source_lang,
                    "target_lang": target_lang, "output": translated, **translator.last_metrics})
                save()
                if not translator.last_metrics["stopped_on_eos"]:
                    raise RuntimeError("追加検証の訳文が出力上限に達しました。")
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["finished_at"] = datetime.now(timezone(timedelta(hours=9))).isoformat()
        save()
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
