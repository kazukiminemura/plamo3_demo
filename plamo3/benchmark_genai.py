"""Compare GenAI NPU configurations with identical tokenized translation inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

from .plamo_genai import DEFAULT_TEXT, SOURCE, prepare_model
from .plamo_translate import build_prompt, positive


def main():
    parser = argparse.ArgumentParser(description="GenAI NPU performance investigation")
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument("--model-dir", type=Path)
    model_options.add_argument("--graph-precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--max-new-tokens", type=positive, default=256)
    parser.add_argument("--validate", action="store_true", help="追加4文の日英・英日翻訳も検証")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import openvino as ov
    import openvino_genai as genai
    from transformers import AutoTokenizer

    args.model_dir = args.model_dir or prepare_model(args.graph_precision)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True, local_files_only=True)
    prompt = build_prompt(DEFAULT_TEXT, "ja", "en")
    encoded = tokenizer(prompt, return_tensors="np")
    inputs = genai.TokenizedInputs(ov.Tensor(encoded["input_ids"]), ov.Tensor(encoded["attention_mask"]))
    config = {"MAX_PROMPT_LEN": 128, "MIN_RESPONSE_LEN": args.max_new_tokens, "GENERATE_HINT": "BEST_PERF"}
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    report = {"status": "compiling", "model_dir": str(args.model_dir), "config": config,
              "device": "NPU", "npu": ov.Core().get_property("NPU", "FULL_DEVICE_NAME"),
              "model_xml_sha256": hashlib.sha256((args.model_dir / "openvino_model.xml").read_bytes()).hexdigest(),
              "openvino": ov.__version__, "genai": genai.__version__, "input_text": DEFAULT_TEXT,
              "input_tokens": encoded["input_ids"].shape[1], "max_new_tokens": args.max_new_tokens,
              "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    print("Compiling", args.model_dir, config, flush=True)
    started = perf_counter()
    pipe = genai.LLMPipeline(args.model_dir, "NPU", **config)
    report["initialization_seconds"] = perf_counter() - started
    report["status"] = "measuring"
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                  "eos_token_id": 2, "stop_token_ids": {1, 2, 16}}

    def collect(result):
        metrics = result.perf_metrics
        return {"output_text": tokenizer.decode(result.tokens[0], skip_special_tokens=True),
                  "token_ids": result.tokens[0], "tokens": metrics.get_num_generated_tokens(),
                  "ttft_seconds": metrics.get_ttft().mean / 1000,
                  "decode_tokens_per_second": metrics.get_throughput().mean,
                  "generation_seconds": metrics.get_generate_duration().mean / 1000,
                  "inference_ms": metrics.get_inference_duration().mean,
                  "finish_reasons": [str(reason) for reason in result.finish_reasons],
                  "stopped_on_eos": all(reason == genai.GenerationFinishReason.STOP for reason in result.finish_reasons)}

    for index in range(args.runs + 1):
        sample = collect(pipe.generate(inputs, **generation))
        if index == 0:
            report["warmup"] = sample
        else:
            report["runs"].append(sample)
        save()
        print(index, sample["decode_tokens_per_second"], sample["ttft_seconds"], sample["tokens"], flush=True)
        if not sample["stopped_on_eos"]:
            raise RuntimeError("Translation reached the output limit")
    report["median_decode_tokens_per_second"] = statistics.median(r["decode_tokens_per_second"] for r in report["runs"])
    report["median_ttft_seconds"] = statistics.median(r["ttft_seconds"] for r in report["runs"])
    report["median_generation_seconds"] = statistics.median(r["generation_seconds"] for r in report["runs"])
    report["outputs_identical"] = all(r["token_ids"] == report["runs"][0]["token_ids"] for r in report["runs"])
    if args.validate:
        report["validation"] = []
        for text, source, target in [
            ("明日の午前十時に駅で待ち合わせましょう。", "ja", "en"),
            ("The server is down. Please restart the application.", "en", "ja"),
            ("このファイルを保存してから、画面を閉じてください。", "ja", "en"),
            ("請求額は税込みで12,345円です。支払い期限は2026年11月30日です。", "ja", "en"),
        ]:
            x = tokenizer(build_prompt(text, source, target), return_tensors="np")
            extra = genai.TokenizedInputs(ov.Tensor(x["input_ids"]), ov.Tensor(x["attention_mask"]))
            checked = {"input_text": text, "source_lang": source, "target_lang": target,
                       **collect(pipe.generate(extra, **generation))}
            report["validation"].append(checked)
            save()
            print("validation", checked["output_text"], flush=True)
            if not checked["stopped_on_eos"]:
                raise RuntimeError("Validation translation reached the output limit")
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
