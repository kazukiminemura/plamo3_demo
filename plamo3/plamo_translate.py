"""PLaMo 3 Translate CLI. Run with --help for export and translation commands."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from .model_source import REVISION

LANGUAGES = "ja en zh-hans zh-hant ko ar az be bg bn ca ceb de es et eu fa fr gu he hi hr hu hy id it jv ka kk km lt lv ml mr ms my ne nl pa pl ru si sk sl sr sw ta te th tl ur uz vi".split()
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "models/plamo-3-translate-8b-int4")
DEFAULT_NPU_MODEL_DIR = str(Path(__file__).resolve().parent / "models/plamo-npu-rotated-cw-int4")
DEFAULT_MAX_NEW_TOKENS = 256


def default_model_dir(device):
    if device.upper().split(".")[0] == "NPU" and (Path(DEFAULT_NPU_MODEL_DIR) / "plamo_openvino.json").is_file():
        return DEFAULT_NPU_MODEL_DIR
    return DEFAULT_MODEL_DIR


def normalize_attention(model):
    """Use rank-4 GQA so OpenVINO's attention optimizations can recognize it."""
    from openvino import opset13 as ops

    for attention in list(model.get_ordered_ops()):
        if attention.get_type_name() != "ScaledDotProductAttention":
            continue
        query = attention.input_value(0)
        if query.get_partial_shape().rank.get_length() != 5:
            continue
        shape = ops.shape_of(query)

        def dimensions(value, indices):
            return ops.gather(ops.shape_of(value), ops.constant(indices, dtype="int64"), 0)

        def flatten(value):
            target = ops.concat([dimensions(value, [0]), ops.constant([-1], dtype="int64"),
                                 dimensions(value, [3, 4])], 0)
            return ops.reshape(value, target, False)

        repeated = []
        for index in (1, 2):
            value = attention.input_value(index)
            target = ops.concat([dimensions(value, [0, 1]), dimensions(query, [2]),
                                 dimensions(value, [3, 4])], 0)
            repeated.append(flatten(ops.broadcast(value, target, "NUMPY")))
        mask = ops.squeeze(attention.input_value(3), ops.constant([2], dtype="int64"))
        scale = attention.input_value(4) if attention.get_input_size() > 4 else None
        replacement = ops.scaled_dot_product_attention(flatten(query), *repeated, mask, scale,
                                                       causal=attention.get_attributes()["causal"])
        replacement = ops.reshape(replacement, shape, False)
        replacement.set_friendly_name(attention.get_friendly_name() + "/rank4")
        attention.output(0).replace(replacement.output(0))
    model.validate_nodes_and_infer_types()
    return model


def decompose_attention(model):
    """Avoid the NPU driver's fused-attention cache aliasing path."""
    import numpy as np
    from openvino import Type, opset13 as ops

    for attention in list(model.get_ordered_ops()):
        if attention.get_type_name() != "ScaledDotProductAttention":
            continue
        if attention.get_attributes()["causal"]:
            raise ValueError("明示マスクのAttentionのみ対応しています。")
        query, key, value = [attention.input_value(i) for i in range(3)]
        dtype = query.get_element_type().to_dtype()
        scale = (attention.input_value(4) if attention.get_input_size() > 4 else
                 ops.constant(np.array(1 / query.get_partial_shape()[-1].get_length() ** 0.5, dtype=dtype)))
        scores = ops.multiply(ops.matmul(query, key, False, True), scale)
        mask = attention.input_value(3)
        if mask.get_element_type() == Type.boolean:
            mask = ops.select(mask, ops.constant(np.array(0, dtype=dtype)),
                              ops.constant(np.array(-np.inf, dtype=dtype)))
        scores = ops.add(scores, mask)
        probability = ops.convert(ops.softmax(ops.convert(scores, "f32"), -1), query.get_element_type())
        result = ops.matmul(probability, value, False, False)
        result.set_friendly_name(attention.get_friendly_name() + "/decomposed")
        attention.output(0).replace(result.output(0))
    model.validate_nodes_and_infer_types()
    return model


def static_model(model, context_length, keep_cache=False):
    """Remove KV I/O and fix the sequence dimension for the NPU compiler."""
    import openvino as ov
    import numpy as np

    model = normalize_attention(model)
    if any(p.get_any_name() == "position_ids" for p in model.inputs):
        model = decompose_attention(model)
    model = ov.Model(list(model.outputs) if keep_cache else [model.output("logits")], model.get_parameters())
    for parameter in list(model.get_parameters()):
        if parameter.output(0).get_any_name().startswith("past_"):
            # Replace Concat(past, current) with current directly. A general
            # ConstantFolding pass would also decompress gigabytes of weights.
            for target in list(parameter.output(0).get_target_inputs()):
                concat = target.get_node()
                if concat.get_type_name() == "ShapeOf":
                    shape = parameter.get_partial_shape()
                    empty_shape = np.array([1, shape[1].get_length(), 0, shape[3].get_length()],
                                           dtype=concat.output(0).get_element_type().to_dtype())
                    concat.output(0).replace(ov.opset13.constant(empty_shape).output(0))
                    continue
                if concat.get_type_name() != "Concat" or concat.get_input_size() != 2:
                    raise ValueError("NPU固定長化に対応していないKVキャッシュ構造です。")
                concat.output(0).replace(concat.input_value(1 - target.get_index()))
            model.remove_parameter(parameter)
    shapes = {"input_ids": [1, context_length], "attention_mask": [1, context_length]}
    if any(p.get_any_name() == "position_ids" for p in model.inputs):
        shapes["position_ids"] = [1, context_length]
    model.reshape(shapes)
    return model


def static_decode_model(model, context_length):
    """One input token, fixed left-padded unrotated KV cache."""
    model = normalize_attention(model)
    if any(p.get_any_name() == "position_ids" for p in model.inputs):
        model = decompose_attention(model)
    if context_length < 2:
        raise ValueError("KVキャッシュには2トークン以上のコンテクストが必要です。")
    shapes = {"input_ids": [1, 1], "attention_mask": [1, context_length]}
    if any(p.get_any_name() == "position_ids" for p in model.inputs):
        shapes["position_ids"] = [1, 1]
    for port in model.inputs:
        name = port.get_any_name()
        if name.startswith("past_"):
            shape = port.get_partial_shape()
            shapes[name] = [1, shape[1].get_length(), context_length - 1, shape[3].get_length()]
    model.reshape(shapes)
    return model


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1以上を指定してください")
    return number


def build_prompt(text, source_lang=None, target_lang="ja"):
    if not text.strip():
        raise ValueError("翻訳するテキストが空です。")
    if "<|plamo:" in text:
        raise ValueError("入力にはPFML3の制御トークンを含められません。")
    source = f"<|plamo:key|>lang<|plamo:val|>{source_lang}" if source_lang else ""
    return (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        f"<|plamo:tag|>input{source}<|plamo:msg|>{text}"
        f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target_lang}<|plamo:msg|>"
    )


def context_for_text(tokenizer, text, source_lang, target_lang, max_new_tokens, model_limit):
    prompt = build_prompt(text, source_lang, target_lang)
    prompt_tokens = tokenizer(prompt, return_tensors="np", add_special_tokens=True)["input_ids"].shape[1]
    required = prompt_tokens + max_new_tokens
    if required > model_limit:
        raise ValueError(f"入力 {prompt_tokens} + 出力上限 {max_new_tokens} がモデル上限 {model_limit} を超えます。")
    return min(model_limit, ((required + 31) // 32) * 32)


class Translator:
    def __init__(self, model_dir, device, npu_context_length=128, npu_kv_cache=None, npu_bucketed=False, tokenizer=None):
        started = perf_counter()
        import openvino as ov
        from transformers import AutoTokenizer

        directory = Path(model_dir)
        metadata_path = directory / "plamo_openvino.json"
        if not metadata_path.is_file():
            raise ValueError(f"変換済みモデルがありません: {directory}。先に export を実行してください。")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.rotated_cache = self.metadata.get("cache_mode") == "rotated"
        tokenizer_started = perf_counter()
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        tokenizer_finished = perf_counter()
        core = ov.Core()
        print(f"モデルを読み込み中 ({device}) ...", file=sys.stderr, flush=True)
        self.npu_context_length = npu_context_length if device.upper().split(".")[0] == "NPU" else None
        self.npu_kv_cache = bool(self.npu_context_length and (self.rotated_cache if npu_kv_cache is None else npu_kv_cache))
        self.npu_bucketed = bool(self.npu_context_length and npu_bucketed)
        if self.npu_bucketed and self.npu_kv_cache:
            raise ValueError("バケット方式と実験用KVキャッシュは同時に使用できません。")
        if self.npu_context_length and self.npu_context_length > self.metadata["context_length"]:
            raise ValueError(f"このモデルのコンテクスト上限は{self.metadata['context_length']}トークンです。")
        model_path = str(directory / "openvino_model.xml")
        model = model_path
        if self.npu_context_length:
            print(f"NPU用固定長グラフを構築中 ({npu_context_length} tokens) ...", file=sys.stderr, flush=True)
            model = static_model(core.read_model(model), npu_context_length, keep_cache=self.npu_kv_cache)
        config = {}
        if self.npu_context_length:
            config = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "NPU",
                      "NPUW_ONLINE_PIPELINE": "REP", "NPUW_FOLD": "YES"}
            if self.npu_kv_cache:
                config["NPU_COMPILATION_MODE_PARAMS"] = "compute-layers-with-higher-precision=Sqrt,ReduceMean,Power,RMS"
                config.update({"NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
                               "NPUW_DQ": "YES", "NPUW_DQ_FULL": "NO",
                               "NPUW_WEIGHTS_BANK": "shared", "NPUW_FUNCALL_FOR_ALL": "YES",
                               "NPUW_FUNCALL_ASYNC": "YES",
                               "NPU_TURBO": "YES", "PERFORMANCE_HINT": "LATENCY",
                               "NPU_TILES": core.get_property("NPU", "NPU_MAX_TILES")})
            if self.npu_bucketed:
                config.update({"NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES", "NPUW_DQ": "YES",
                               "NPUW_DQ_FULL": "NO", "NPUW_WEIGHTS_BANK": "shared",
                               "NPUW_FUNCALL_FOR_ALL": "YES", "NPUW_FUNCALL_ASYNC": "YES",
                               "NPU_TURBO": "YES", "PERFORMANCE_HINT": "LATENCY"})
            print("NPU分割コンパイル中 ...", file=sys.stderr, flush=True)
        self.compile_config = config
        compile_started = perf_counter()
        self.compiled = core.compile_model(model, device, config)
        prefill_finished = perf_counter()
        if self.npu_kv_cache:
            print("NPU 1トークンDecodeをコンパイル中 ...", file=sys.stderr, flush=True)
            decoder = static_decode_model(core.read_model(model_path), npu_context_length)
            self.decode_compiled = core.compile_model(decoder, device, config)
            self.decode_request = self.decode_compiled.create_infer_request()
            print(f"Decode実行デバイス: {self.decode_compiled.get_property('EXECUTION_DEVICES')}",
                  file=sys.stderr, flush=True)
        self.bucket_compiled = {npu_context_length: self.compiled} if self.npu_bucketed else {}
        if self.npu_bucketed:
            for size in (32, 64):
                if size < npu_context_length:
                    print(f"NPU短文用グラフをコンパイル中 ({size} tokens) ...", file=sys.stderr, flush=True)
                    bucket = static_model(core.read_model(model_path), size)
                    self.bucket_compiled[size] = core.compile_model(bucket, device, config)
        compile_finished = perf_counter()
        print(f"実行デバイス: {self.compiled.get_property('EXECUTION_DEVICES')}", file=sys.stderr, flush=True)
        self.request = self.compiled.create_infer_request()
        self.bucket_requests = {size: compiled.create_infer_request() for size, compiled in self.bucket_compiled.items()}
        self.stop_ids = {self.tokenizer.convert_tokens_to_ids(t) for t in ("<|plamo:tag|>", "<|plamo:bos|>", "<|plamo:eos|>")}
        self.load_metrics = {
            "tokenizer_load_seconds": tokenizer_finished - tokenizer_started,
            "graph_prepare_seconds": compile_started - tokenizer_finished,
            "compile_seconds": compile_finished - compile_started,
            "prefill_compile_seconds": prefill_finished - compile_started,
            "decode_prepare_and_compile_seconds": compile_finished - prefill_finished if self.npu_kv_cache else 0,
            "bucket_compile_seconds": compile_finished - prefill_finished if self.npu_bucketed else 0,
            "initialize_seconds": perf_counter() - started,
        }

    def translate(self, text, source_lang=None, target_lang="ja", max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        import numpy as np
        import openvino as ov

        started = perf_counter()
        prompt = build_prompt(text, source_lang, target_lang)
        ids = self.tokenizer(prompt, return_tensors="np", add_special_tokens=True)["input_ids"].astype(np.int64)
        length = ids.shape[1]
        if self.npu_context_length:
            return self.translate_static(ids, max_new_tokens, started)
        if length + max_new_tokens > self.metadata["context_length"]:
            raise ValueError(f"入力 {length} + 出力上限 {max_new_tokens} がモデル上限 {self.metadata['context_length']} を超えます。入力を分割してください。")
        values = {"input_ids": ids, "attention_mask": np.ones((1, length), dtype=np.int64)}
        if self.rotated_cache:
            values["position_ids"] = np.arange(length, dtype=np.int64)[None]
        cache_names = [p.get_any_name() for p in self.compiled.inputs if p.get_any_name().startswith("past_")]
        for name in cache_names:
            port = self.compiled.input(name)
            shape = [1, self.metadata["num_key_value_heads"], 0, self.metadata["head_dim"]]
            values[name] = ov.Tensor(port.get_element_type(), shape)
        generated = []
        emitted_at = []
        for _ in range(max_new_tokens):
            self.request.infer(values)
            token = int(np.argmax(self.request.get_tensor("logits").data[0, -1]))
            if token in self.stop_ids:
                break
            generated.append(token)
            emitted_at.append(perf_counter())
            length += 1
            # ponytail: copying the full cache costs bandwidth on long prompts;
            # use device-resident state if profiling shows this is the bottleneck.
            # Copy before infer() overwrites the request's output buffers.
            values = {
                name: self.request.get_tensor(name.replace("past_", "present_")).data.copy()
                for name in cache_names
            }
            values["input_ids"] = np.array([[token]], dtype=np.int64)
            values["attention_mask"] = np.ones((1, length), dtype=np.int64)
            if self.rotated_cache:
                values["position_ids"] = np.array([[length - 1]], dtype=np.int64)
        else:
            print("警告: 出力トークン上限に達しました。訳文が途中の可能性があります。", file=sys.stderr)
        self.last_metrics = {
            "first_token_seconds": emitted_at[0] - started if emitted_at else None,
            "decode_tokens_per_second": ((len(emitted_at) - 1) / (emitted_at[-1] - emitted_at[0])
                                         if len(emitted_at) > 1 else None),
        }
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def translate_static(self, ids, max_new_tokens, started=None):
        import numpy as np

        started = perf_counter() if started is None else started
        capacity = self.npu_context_length
        tokens = ids[0].tolist()
        if len(tokens) + max_new_tokens > capacity:
            raise ValueError(f"入力 {len(tokens)} + 出力上限 {max_new_tokens} がNPU固定長 {capacity} を超えます。"
                             "入力を短くするか--max-new-tokensを減らしてください。")
        generated = []
        inference_times = []
        emitted_at = []
        use_cache = self.npu_kv_cache
        rotated = self.rotated_cache
        cache_arrays = {}
        previous_request = None
        for step in range(max_new_tokens):
            bucketed = self.npu_bucketed
            padded_length = min(size for size in self.bucket_requests if len(tokens) <= size) if bucketed else capacity
            padded = np.full((1, padded_length), self.tokenizer.pad_token_id, dtype=np.int64)
            mask = np.zeros((1, padded_length), dtype=np.int64)
            padded[0, -len(tokens):] = tokens
            mask[0, -len(tokens):] = 1
            request = self.bucket_requests[padded_length] if bucketed else self.request
            values = {"input_ids": padded, "attention_mask": mask}
            if rotated:
                values["position_ids"] = np.maximum(mask.cumsum(axis=1) - 1, 0)
            if use_cache and step:
                request = self.decode_request
                values["input_ids"] = np.array([[tokens[-1]]], dtype=np.int64)
                if rotated:
                    cached_length = len(tokens) - 1
                    mask.fill(0)
                    mask[0, :cached_length] = 1
                    mask[0, -1] = 1
                    values["position_ids"] = np.array([[cached_length]], dtype=np.int64)
                for port in self.decode_compiled.inputs:
                    name = port.get_any_name()
                    if name.startswith("past_"):
                        cache = previous_request.get_tensor(name.replace("past_", "present_")).data
                        # SWA layers may return fewer keys than the global context.
                        if rotated:
                            if step == 1:
                                array = np.zeros(port.get_shape(), dtype=port.get_element_type().to_dtype())
                                array[:, :, :cached_length, :] = cache[:, :, -cached_length:, :]
                                cache_arrays[name] = array
                            else:
                                array = cache_arrays[name]
                                array[:, :, cached_length - 1:cached_length, :] = cache[:, :, -1:, :]
                        else:
                            array = np.zeros(port.get_shape(), dtype=port.get_element_type().to_dtype())
                            retained = min(cache.shape[2], capacity - 1)
                            array[:, :, -retained:, :] = cache[:, :, -retained:, :]
                        values[name] = array
            infer_started = perf_counter()
            request.infer(values)
            inference_times.append(perf_counter() - infer_started)
            previous_request = request
            token = int(np.argmax(request.get_tensor("logits").data[0, -1]))
            if token in self.stop_ids:
                break
            generated.append(token)
            emitted_at.append(perf_counter())
            tokens.append(token)
        else:
            print("警告: 出力トークン上限に達しました。訳文が途中の可能性があります。", file=sys.stderr)
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        elapsed = perf_counter() - started
        self.last_metrics = {
            "prompt_tokens": ids.shape[1],
            "generated_tokens": len(generated),
            "generated_token_ids": generated,
            "inference_calls": len(inference_times),
            "inference_seconds": sum(inference_times),
            "inference_call_seconds": inference_times,
            "translation_seconds": elapsed,
            "first_token_seconds": emitted_at[0] - started if emitted_at else None,
            "tokens_per_second": len(generated) / elapsed,
            "decode_tokens_per_second": ((len(emitted_at) - 1) / (emitted_at[-1] - emitted_at[0])
                                         if len(emitted_at) > 1 else None),
            "stopped_on_eos": token in self.stop_ids,
        }
        return result


def parser():
    root = argparse.ArgumentParser(description="PLaMo 3 Translate 8B / OpenVINO CUI")
    commands = root.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="公式モデルをダウンロードしOpenVINOへ変換")
    export.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    export.add_argument("--revision", default=REVISION, help="Hugging FaceのブランチまたはコミットID（既定: 事前ダウンロードと同じ固定コミット）")
    export.add_argument("--weight-format", choices=["int4", "int8", "fp16"], default="int4")
    export.add_argument("--npu-optimized", action="store_true", help="回転済みKV形式（SWAウィンドウ内の文脈専用）")
    export.add_argument("--group-size", type=int, choices=[-1, 128], default=128,
                        help="INT4の重み圧縮単位（-1はチャネル単位）")
    commands.add_parser("devices", help="OpenVINOデバイス一覧")
    translate = commands.add_parser("translate", help="単発翻訳 / 標準入力 / 対話モード")
    translate.add_argument("text", nargs="?")
    translate.add_argument("--input", type=Path, help="UTF-8の入力ファイル")
    translate.add_argument("--output", type=Path, help="訳文のUTF-8出力ファイル（単発翻訳用）")
    translate.add_argument("--source-lang", choices=LANGUAGES, help="省略時は自動判定")
    translate.add_argument("--target-lang", choices=LANGUAGES, default="ja")
    translate.add_argument("--model-dir", help="省略時はNPU用の最適化モデルを優先（CPU/GPUは通常モデル）")
    translate.add_argument("--device", default="NPU", help="NPU（既定） / CPU / GPU / AUTO など")
    translate.add_argument("--npu-no-kv-cache", action="store_true", help="旧方式: 毎回接頭辞を再計算")
    translate.add_argument("--npu-kv-cache", action="store_true",
                           help="KVキャッシュを使用（回転済みIR推奨、通常はモデルから自動判定）")
    translate.add_argument("--npu-bucketed", action="store_true", help="短文用32/64トークングラフを併用")
    translate.add_argument("--max-new-tokens", type=positive, default=DEFAULT_MAX_NEW_TOKENS,
                           help="生成トークン数の上限（既定256）。内部のコンテクスト長は自動設定")
    return root


def main(argv=None):
    # NNCF's progress renderer includes characters outside Windows cp932.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.command == "export":
            from .export_model import export_model

            export_model(args)
            return 0
        if args.command == "devices":
            import openvino as ov

            core = ov.Core()
            for device in core.available_devices:
                print(f"{device}: {core.get_property(device, 'FULL_DEVICE_NAME')}")
            return 0
        if args.text is not None and args.input is not None:
            arguments.error("textと--inputは同時に指定できません")
        args.model_dir = args.model_dir or default_model_dir(args.device)
        text = args.text
        if args.input is not None:
            text = args.input.read_text(encoding="utf-8-sig")
        elif text is None and not sys.stdin.isatty():
            text = sys.stdin.read()
        if text is not None:
            build_prompt(text, args.source_lang, args.target_lang)
        elif args.output:
            arguments.error("--outputにはtext、--input、または標準入力が必要です")
        translator = None
        tokenizer = None
        is_npu = args.device.upper().split(".")[0] == "NPU"
        if is_npu:
            from transformers import AutoTokenizer

            metadata = json.loads((Path(args.model_dir) / "plamo_openvino.json").read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)

        def run(value):
            nonlocal translator
            capacity = (context_for_text(tokenizer, value, args.source_lang, args.target_lang,
                                         args.max_new_tokens, metadata["context_length"]) if is_npu else None)
            if translator is None or (is_npu and translator.npu_context_length < capacity):
                translator = Translator(args.model_dir, args.device, capacity,
                    False if args.npu_no_kv_cache or args.npu_bucketed else (True if args.npu_kv_cache else None),
                    args.npu_bucketed, tokenizer=tokenizer)
            result = translator.translate(value, args.source_lang, args.target_lang, args.max_new_tokens)
            metrics = translator.last_metrics
            ttft = metrics["first_token_seconds"]
            decode = metrics["decode_tokens_per_second"]
            ttft_text = f"{ttft:.3f} s" if ttft is not None else "N/A"
            decode_text = f"{decode:.3f} tokens/s" if decode is not None else "N/A"
            print(f"TTFT: {ttft_text} | Decode: {decode_text}", file=sys.stderr)
            return result

        if text is not None:
            result = run(text)
            if args.output:
                args.output.write_text(result + "\n", encoding="utf-8")
            else:
                print(result)
        else:
            print("1行ずつ翻訳します。終了: /quit または Ctrl+C", file=sys.stderr)
            while True:
                try:
                    line = input("> ")
                except EOFError:
                    break
                if line.strip() == "/quit":
                    break
                if line.strip():
                    try:
                        print(run(line))
                    except ValueError as error:
                        print(f"エラー: {error}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
