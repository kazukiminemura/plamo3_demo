"""Prepare a stateful PLaMo INT4 IR and run OpenVINO GenAI on NPU."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

from .plamo_translate import build_prompt, positive

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models/plamo-npu-rotated-cw-int4"
DESTINATION = ROOT / "models/plamo-genai-int4"
DEFAULT_TEXT = (
    "来週の月曜日に予定している開発会議では、翻訳アプリケーションの性能と使いやすさについて確認します。"
    "参加者は、これまでに見つかった問題点と改善案を金曜日までに共有してください。"
    "新しいバージョンでは、文章を翻訳した後に、最初のトークンが生成されるまでの時間と、その後の生成速度が表示されます。"
    "短い文章だけでなく、複数の文からなる案内や業務上の連絡も使い、訳文が最後まで自然につながることを確認してください。"
    "測定結果には使用したモデルと入力文を記録し、同じ条件で比較できるようにします。"
)


def normalize_genai_precision(model):
    """Use GenAI's FP32 graph convention while keeping packed INT4 weights intact."""
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    count = 0
    for node in list(model.get_ordered_ops()):
        if node.get_type_name() == "Parameter" and node.get_output_element_type(0) == ov.Type.f16:
            node.set_element_type(ov.Type.f32)
            count += 1
        elif node.get_type_name() == "Constant" and node.get_output_element_type(0) == ov.Type.f16:
            replacement = ops.constant(node.get_data().astype(np.float32))
            replacement.set_friendly_name(node.get_friendly_name())
            replacement.output(0).get_tensor().set_names(node.output(0).get_names())
            node.output(0).replace(replacement.output(0))
            count += 1
            continue
        elif node.get_type_name() == "Convert" and node.get_output_element_type(0) == ov.Type.f16:
            replacement = ops.convert(node.input_value(0), ov.Type.f32)
            replacement.set_friendly_name(node.get_friendly_name())
            replacement.output(0).get_tensor().set_names(node.output(0).get_names())
            node.output(0).replace(replacement.output(0))
            count += 1
            continue
        node.validate_and_infer_types()
    model.validate_nodes_and_infer_types()
    return count


def remove_redundant_output_slice(model):
    """Expose MatMul -> Convert -> Result to GenAI's shared LM-head matcher."""
    count = 0
    for result in model.get_results():
        node = result.input_value(0).get_node()
        if node.get_type_name() == "Convert":
            node = node.input_value(0).get_node()
        if node.get_type_name() != "Slice" or node.get_input_size() != 5:
            continue
        constants = [node.input_value(i).get_node() for i in range(1, 5)]
        if not all(c.get_type_name() == "Constant" for c in constants):
            continue
        starts, stops, steps, axes = [c.get_data().reshape(-1).tolist() for c in constants]
        shape = node.get_input_partial_shape(0)
        if shape.rank.is_dynamic:
            continue
        rank = shape.rank.get_length()
        if all(step == 1 and start == 0 and shape[axis % rank].is_static and
               stop >= shape[axis % rank].get_length()
               for start, stop, step, axis in zip(starts, stops, steps, axes, strict=True)):
            node.output(0).replace(node.input_value(0))
            count += 1
    model.validate_nodes_and_infer_types()
    return count


def make_stateful(model):
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
    from openvino.preprocess import PrePostProcessor
    from openvino._offline_transformations import apply_make_stateful_transformation

    # GenAI selects the relevant logits itself; retain the sequence dimension.
    for node in list(model.get_ordered_ops()):
        if node.get_type_name() != "Slice" or node.get_input_size() != 5:
            continue
        start, _, step, axes = [node.input_value(i).get_node() for i in range(1, 5)]
        if all(n.get_type_name() == "Constant" for n in (start, step, axes)):
            if start.get_data().tolist() == [-1] and axes.get_data().tolist() == [1] and step.get_data().tolist() == [1]:
                node.output(0).replace(node.input_value(0))
    mapping = {}
    for port in model.inputs:
        name = port.get_any_name()
        if name.startswith("past_"):
            _, index, kind = name.split("_")
            source_name, result_name = f"past_key_values.{index}.{kind}", f"present.{index}.{kind}"
            model.output(name.replace("past_", "present_")).get_tensor().set_names({result_name})
            port.get_tensor().set_names({source_name})
            mapping[source_name] = result_name
    apply_make_stateful_transformation(model, mapping)
    beam_idx = ops.parameter([-1], dtype=np.int32, name="beam_idx")
    beam_idx.output(0).get_tensor().set_names({"beam_idx"})
    model.add_parameters([beam_idx])
    for node in model.get_ordered_ops():
        if node.get_type_name() == "ReadValue" and node.get_input_size() == 0:
            shape = [dim.get_length() if dim.is_static else 0 for dim in node.get_output_partial_shape(0)]
            initial = ops.constant(np.zeros(shape, dtype=node.get_output_element_type(0).to_dtype()))
            node.set_arguments([initial])
            targets = list(node.output(0).get_target_inputs())
            reordered = ops.gather(node, beam_idx, ops.constant(0, dtype=np.int64))
            for target in targets:
                target.replace_source_output(reordered.output(0))
    model.validate_nodes_and_infer_types()
    processing = PrePostProcessor(model)
    processing.output("logits").tensor().set_element_type(ov.Type.f32)
    return processing.build()


def prepare_model():
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from tokenizers import AddedToken, Tokenizer, decoders, models, processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    marker = DESTINATION / "plamo_genai.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("adapter_version") == 4:
        return
    metadata = json.loads((SOURCE / "plamo_openvino.json").read_text(encoding="utf-8"))
    if metadata["cache_mode"] != "rotated" or metadata["weight_format"] != "int4":
        raise ValueError("回転済みKV形式のINT4モデルが必要です。")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for path in SOURCE.iterdir():
        if path.suffix in (".json", ".jsonl", ".py", ".jinja"):
            shutil.copyfile(path, DESTINATION / path.name)
        elif path.name == "LICENSE":
            shutil.copytree(path, DESTINATION / path.name, dirs_exist_ok=True)

    print("PLaMoトークナイザーIRを準備中 ...", file=sys.stderr, flush=True)
    rows = [json.loads(line) for line in (SOURCE / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines()]
    backend = Tokenizer(models.Unigram([(row[0], round(row[1] * 1e4) / 1e4) for row in rows],
                                       unk_id=0, byte_fallback=True))
    backend.add_tokens([AddedToken(row[0], normalized=False) for row in rows if len(row) > 2 and row[2] == "CONTROL"])
    backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    backend.post_processor = processors.TemplateProcessing(single="BOS $A",
        special_tokens=[{"id": "BOS", "ids": [1], "tokens": [rows[1][0]]}])
    # TemplateProcessing's string parser treats ':' in PLaMo's BOS as a type ID.
    serialized = json.loads(backend.to_str())
    processor = serialized["post_processor"]
    processor["single"][0]["SpecialToken"]["id"] = rows[1][0]
    processor["special_tokens"][rows[1][0]] = processor["special_tokens"].pop("BOS")
    processor["special_tokens"][rows[1][0]]["id"] = rows[1][0]
    backend = Tokenizer.from_str(json.dumps(serialized, ensure_ascii=False))
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=rows[0][0], bos_token=rows[1][0],
                                  eos_token=rows[2][0], pad_token=rows[3][0],
                                  additional_special_tokens=["<|plamo:tag|>"], clean_up_tokenization_spaces=False)
    tokenizer_model, detokenizer_model = convert_tokenizer(fast, with_detokenizer=True)
    ov.save_model(tokenizer_model, DESTINATION / "openvino_tokenizer.xml")
    ov.save_model(detokenizer_model, DESTINATION / "openvino_detokenizer.xml")
    # Generation uses the official tokenizer's IDs; no approximate tokenization is used.
    original = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True, local_files_only=True)
    native = ov.Core().compile_model(tokenizer_model, "CPU")
    checks = [build_prompt(DEFAULT_TEXT, "ja", "en"), build_prompt("Hello, world!", "en", "ja"),
              "こんにちは。", "The meeting is next Monday."]
    matches = [native([text])[native.output("input_ids")][0].tolist() == original.encode(text) for text in checks]
    print(f"トークナイザー一致: {sum(matches)}/{len(matches)}（推論入力は公式実装を使用）", file=sys.stderr)
    print("INT4重みを再利用してstateful IRを保存中 ...", file=sys.stderr, flush=True)
    model = ov.Core().read_model(SOURCE / "openvino_model.xml")
    promoted = normalize_genai_precision(model)
    model = make_stateful(model)
    removed = remove_redundant_output_slice(model)
    ov.save_model(model, DESTINATION / "openvino_model.xml", compress_to_fp16=False)
    marker.write_text(json.dumps({**metadata, "adapter_version": 4, "stateful": True, "tokenization": "official_python",
                                 "graph_precision": "float32", "promoted_nodes": promoted,
                                 "removed_output_slices": removed,
                                 "native_tokenizer_checks": matches}, indent=2), encoding="utf-8")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="PLaMo 3 INT4をOpenVINO GenAIとNPUで推論")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--max-new-tokens", type=positive, default=256)
    args = parser.parse_args(argv)
    prepare_model()
    import openvino as ov
    import openvino_genai as genai
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True, local_files_only=True)
    prompt = build_prompt(args.text, "ja", "en")
    encoded = tokenizer(prompt, return_tensors="np")
    tokens = encoded["input_ids"].shape[1]
    prompt_length = ((tokens + 31) // 32) * 32
    if prompt_length + args.max_new_tokens > 2048:
        raise ValueError("このIRでは入力と生成上限の合計は2048トークン以内にしてください。")
    inputs = genai.TokenizedInputs(ov.Tensor(encoded["input_ids"]), ov.Tensor(encoded["attention_mask"]))
    config = {"MAX_PROMPT_LEN": prompt_length, "MIN_RESPONSE_LEN": args.max_new_tokens,
              "GENERATE_HINT": "BEST_PERF"}
    print("PLaMo 3 INT4をGenAIのNPUパイプラインでコンパイル中 ...", file=sys.stderr, flush=True)
    started = perf_counter()
    pipe = genai.LLMPipeline(DESTINATION, "NPU", **config)
    initialization_seconds = perf_counter() - started
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                  "eos_token_id": 2, "stop_token_ids": {1, 2, 16}}
    print("ウォームアップ中 ...", file=sys.stderr, flush=True)
    pipe.generate(inputs, **generation)
    result = pipe.generate(inputs, **generation)
    output_text = tokenizer.decode(result.tokens[0], skip_special_tokens=True)
    print(output_text)
    metrics = result.perf_metrics
    report = {
        "model_dir": str(DESTINATION), "device": "NPU", "pipeline": "OpenVINO GenAI LLMPipeline",
        "weight_format": "int4", "graph_precision": "float32", "adapter_version": 4,
        "openvino": ov.__version__, "openvino_genai": genai.__version__,
        "pipeline_config": config, "max_new_tokens": args.max_new_tokens,
        "initialization_seconds": initialization_seconds, "warmup_runs": 1, "measured_runs": 1,
        "input_text": args.text, "input_tokens": tokens, "output_text": output_text,
        "generated_tokens": metrics.get_num_generated_tokens(),
        "ttft_seconds": metrics.get_ttft().mean / 1000,
        "decode_tokens_per_second": metrics.get_throughput().mean,
        "generation_seconds": metrics.get_generate_duration().mean / 1000,
        "finish_reasons": [str(reason) for reason in result.finish_reasons],
        "tokenizer": "Official Python tokenizer; tokenization/detokenization excluded from GenAI timings",
    }
    now = datetime.now(timezone(timedelta(hours=9)))
    output = ROOT / "baselines" / f"plamo3-genai-npu-{now:%Y%m%dT%H%M%S%z}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TTFT: {report['ttft_seconds']:.3f} s | Decode: {report['decode_tokens_per_second']:.3f} tokens/s", file=sys.stderr)
    print(f"Saved: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
