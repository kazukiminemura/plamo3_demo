"""Run: uv run python -X utf8 -m unittest -v (no original weights required)."""

import tempfile
import unittest
import json
import io
from pathlib import Path
from unittest.mock import Mock, patch

from .plamo_translate import build_prompt, parser


class AppTest(unittest.TestCase):
    def test_genai_precision_selects_separate_models(self):
        from . import plamo_genai

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp32 = root / "models/plamo-genai-int4"
            fp16 = root / "models/plamo-genai-fp16-int4"
            for path, precision in [(fp32, "float32"), (fp16, "float16")]:
                path.mkdir(parents=True)
                (path / "plamo_genai.json").write_text(json.dumps({
                    "adapter_version": 4, "graph_precision": precision}), encoding="utf-8")
            with patch.object(plamo_genai, "ROOT", root), patch.object(plamo_genai, "DESTINATION", fp32):
                self.assertEqual(plamo_genai.prepare_model("fp16"), fp16)
                self.assertEqual(plamo_genai.prepare_model("fp32"), fp32)
                with self.assertRaises(ValueError):
                    plamo_genai.prepare_model("fp64")

    def test_genai_precision_preserves_int4_weights(self):
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
        from .plamo_genai import normalize_genai_precision

        packed = ov.Tensor(ov.Type.i4, [8, 4])
        packed.data[:] = np.arange(16, dtype=np.int8)
        weight = ops.multiply(ops.convert(ov.op.Constant(packed), "f16"),
                              ops.constant(np.full((8, 1), 0.125, np.float16)))
        data = ops.parameter([1, 4], np.float16, name="data")
        original = ov.Model([ops.matmul(data, weight, False, True)], [data])
        changed = original.clone()
        self.assertGreater(normalize_genai_precision(changed), 0)
        self.assertEqual(changed.input(0).get_element_type(), ov.Type.f32)
        self.assertEqual(changed.output(0).get_element_type(), ov.Type.f32)
        low_bit = [n for n in changed.get_ordered_ops()
                   if n.get_type_name() == "Constant" and n.get_output_element_type(0) == ov.Type.i4]
        self.assertEqual(len(low_bit), 1)
        np.testing.assert_array_equal(low_bit[0].get_data(), packed.data)
        sample = np.array([[1, -2, 3, -4]], np.float16)
        before = list(ov.Core().compile_model(original, "CPU")(sample).values())[0]
        after = list(ov.Core().compile_model(changed, "CPU")(sample.astype(np.float32)).values())[0]
        np.testing.assert_allclose(before, after, rtol=1e-3, atol=1e-3)


    def test_remove_only_full_output_slice(self):
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
        from .plamo_genai import remove_redundant_output_slice

        data = ops.parameter([1, -1, 8], np.float16, name="data")
        for stop, expected in [(8, 1), (7, 0)]:
            sliced = ops.slice(data, ops.constant([0], np.int64), ops.constant([stop], np.int64),
                               ops.constant([1], np.int64), ops.constant([2], np.int64))
            original = ov.Model([ops.convert(sliced, "f32")], [data])
            changed = original.clone()
            self.assertEqual(remove_redundant_output_slice(changed), expected)
            sample = np.arange(16, dtype=np.float16).reshape(1, 2, 8)
            before = list(ov.Core().compile_model(original, "CPU")(sample).values())[0]
            after = list(ov.Core().compile_model(changed, "CPU")(sample).values())[0]
            np.testing.assert_array_equal(before, after)


    def test_genai_stateful_cache(self):
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
        from .plamo_genai import make_stateful

        ids = ops.parameter([1, -1], np.int64, name="input_ids")
        key = ops.parameter([1, 1, -1, 1], np.float16, name="past_0_key")
        value = ops.parameter([1, 1, -1, 1], np.float16, name="past_0_value")
        for parameter in (ids, key, value):
            parameter.output(0).get_tensor().set_names({parameter.get_friendly_name()})
        current = ops.unsqueeze(ops.convert(ids, "f16"), ops.constant([1, 3], np.int64))
        present_key = ops.concat([key, current], 2)
        present_value = ops.concat([value, ops.multiply(current, ops.constant(2.0, np.float16))], 2)
        total = ops.add(ops.reduce_sum(present_key, ops.constant([1, 2, 3], np.int64), True),
                        ops.reduce_sum(present_value, ops.constant([1, 2, 3], np.int64), True))
        logits = ops.reshape(total, ops.constant([1, 1, 1], np.int64), False)
        for node, name in [(logits, "logits"), (present_key, "present_0_key"), (present_value, "present_0_value")]:
            node.output(0).get_tensor().set_names({name})
        model = make_stateful(ov.Model([logits, present_key, present_value], [ids, key, value]))
        self.assertEqual(model.output("logits").get_element_type(), ov.Type.f32)
        self.assertTrue(all(node.get_output_element_type(0) == ov.Type.f16
                            for node in model.get_ordered_ops() if node.get_type_name() == "ReadValue"))
        request = ov.Core().compile_model(model, "CPU").create_infer_request()
        self.assertEqual(len(request.query_state()), 2)
        for tokens, expected in [([2, 3], 15.0), ([4], 27.0)]:
            request.infer({"input_ids": np.array([tokens], np.int64), "beam_idx": np.array([0], np.int32)})
            self.assertEqual(request.get_tensor("logits").data.item(), expected)
        request.reset_state()
        request.infer({"input_ids": np.array([[2, 3]], np.int64), "beam_idx": np.array([0], np.int32)})
        self.assertEqual(request.get_tensor("logits").data.item(), 15.0)


    def test_document_metrics(self):
        from .benchmark_npu import document_metrics

        sample = {"prompt_tokens": 10, "generated_tokens": 3, "inference_calls": 4,
                  "inference_seconds": 2.0, "first_token_seconds": 0.5,
                  "decode_tokens_per_second": 2.0, "stopped_on_eos": True}
        metrics = document_metrics([sample, sample], 5.0)
        self.assertEqual(metrics["generated_tokens"], 6)
        self.assertEqual(metrics["first_token_seconds"], 0.5)
        self.assertEqual(metrics["decode_seconds"], 2.0)
        self.assertEqual(metrics["decode_tokens_per_second"], 2.0)
        self.assertEqual(metrics["tokens_per_second"], 1.2)
        empty = {**sample, "generated_tokens": 0, "decode_tokens_per_second": None,
                 "first_token_seconds": None, "stopped_on_eos": False}
        self.assertIsNone(document_metrics([empty], 1.0)["decode_tokens_per_second"])
        self.assertFalse(document_metrics([sample, empty], 5.0)["stopped_on_eos"])

    def test_cli_metrics(self):
        from .plamo_translate import main

        with patch("plamo3.plamo_translate.Translator") as translator, patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            translator.return_value.translate.return_value = "こんにちは"
            for ttft, decode, expected in [(0.25, 5.5, "TTFT: 0.250 s | Decode: 5.500 tokens/s"),
                                            (None, None, "TTFT: N/A | Decode: N/A")]:
                translator.return_value.last_metrics = {"first_token_seconds": ttft, "decode_tokens_per_second": decode}
                stdout.seek(0)
                stdout.truncate()
                stderr.seek(0)
                stderr.truncate()
                self.assertEqual(main(["translate", "Hello", "--device", "CPU"]), 0)
                self.assertEqual(stdout.getvalue(), "こんにちは\n")
                self.assertIn(expected, stderr.getvalue())

    def test_prompt_and_arguments(self):
        self.assertEqual(build_prompt("Hello", "en", "ja"),
            "<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello"
            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>")
        for text in ("  ", "<|plamo:tag|>output"):
            with self.assertRaises(ValueError):
                build_prompt(text)
        parsed = parser().parse_args(["translate", "Hello"])
        self.assertEqual(parsed.device, "NPU")
        self.assertEqual(parsed.max_new_tokens, 256)
        self.assertNotIn("npu_context_length", vars(parsed))
        self.assertEqual(parser().parse_args(["translate", "Hello", "--max-new-tokens", "128"]).max_new_tokens, 128)

    def test_automatic_context_budget(self):
        import numpy as np
        from .plamo_translate import context_for_text

        tokenizer = Mock(return_value={"input_ids": np.zeros((1, 26), np.int64)})
        self.assertEqual(context_for_text(tokenizer, "Hello", "en", "ja", 256, 2048), 288)
        self.assertEqual(context_for_text(tokenizer, "Hello", "en", "ja", 230, 2048), 256)
        with self.assertRaises(ValueError):
            context_for_text(tokenizer, "Hello", "en", "ja", 256, 256)

    def test_openvino_matches_official_model(self):
        import numpy as np
        import openvino as ov
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForCausalLM
        from .export_model import convert

        torch.manual_seed(7)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            snapshot_download("pfnet/plamo-3-translate-8b", local_dir=directory,
                              revision="681bd55cf01f5412aab8881e6a4a276c948fabc1",
                              allow_patterns=["config.json", "modeling_plamo.py"])
            config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
            config.hidden_size = 128
            config.intermediate_size = 256
            config.num_attention_heads = 2
            config.num_key_value_heads = 1
            config.head_dim = 128
            config.num_hidden_layers = 2
            config.sliding_window_pattern = 2
            config.window_size = 4
            config.vocab_size = 32
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float().eval()
            converted = convert(model)
            path = Path(directory) / "openvino_model.xml"
            ov.save_model(converted, path, compress_to_fp16=False)
            compiled = ov.Core().compile_model(str(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            request = compiled.create_infer_request()
            values = {f"past_{i}_{kind}": np.empty((1, 1, 0, 128), dtype=np.float32)
                      for i in range(2) for kind in ("key", "value")}
            cache = None
            length = 0
            # Prefill, decode, crossing the sliding window, then a fresh request.
            for tokens in ([4, 5, 6], [7], [8], [9, 10], [11]):
                ids = torch.tensor([tokens])
                length += len(tokens)
                mask = torch.ones(1, length, dtype=torch.long)
                with torch.no_grad():
                    expected = model(input_ids=ids, attention_mask=mask,
                                     past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = expected.past_key_values
                values["input_ids"] = ids.numpy()
                values["attention_mask"] = mask.numpy()
                request.infer(values)
                np.testing.assert_allclose(request.get_tensor("logits").data,
                                           expected.logits.numpy(), rtol=2e-4, atol=2e-5)
                values = {name: request.get_tensor(name.replace("past_", "present_")).data.copy()
                          for name in values if name.startswith("past_")}
            # Reuse the compiled graph with an empty cache and a different prompt length.
            ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
            values = {name: np.empty((1, 1, 0, 128), np.float32) for name in values}
            values.update(input_ids=ids.numpy(), attention_mask=np.ones((1, 6), np.int64))
            request.infer(values)
            with torch.no_grad():
                expected = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            np.testing.assert_allclose(request.get_tensor("logits").data,
                                       expected.logits.numpy(), rtol=2e-4, atol=2e-5)
            from .plamo_translate import static_model

            fixed = static_model(ov.Core().read_model(str(path)), 12)
            fixed_compiled = ov.Core().compile_model(fixed, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            padded = np.full((1, 12), 3, np.int64)
            padded[0, -6:] = ids.numpy()[0]
            mask = np.zeros((1, 12), np.int64)
            mask[0, -6:] = 1
            fixed_logits = fixed_compiled({"input_ids": padded, "attention_mask": mask})[fixed_compiled.output("logits")]
            np.testing.assert_allclose(fixed_logits, expected.logits.numpy(), rtol=2e-4, atol=2e-5)
            # Exercise the actual CLI runtime, including output buffers and reset.
            from .plamo_translate import Translator

            (Path(directory) / "plamo_openvino.json").write_text(json.dumps({
                "context_length": 32768, "num_key_value_heads": 1, "head_dim": 128,
            }), encoding="utf-8")
            tokenizer = Mock()
            tokenizer.return_value = {"input_ids": np.array([[4, 5, 6]], np.int64)}
            tokenizer.convert_tokens_to_ids.side_effect = [16, 1, 2]
            tokenizer.decode.side_effect = lambda tokens, **kwargs: str(tokens)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                translator = Translator(directory, "CPU")
            first = translator.translate("Hello", max_new_tokens=3)
            self.assertEqual(first, translator.translate("Again", max_new_tokens=3))
            with torch.no_grad():
                expected_ids = model.generate(torch.tensor([[4, 5, 6]]),
                    do_sample=False, max_new_tokens=3, eos_token_id=[16, 1, 2])
            tokens = expected_ids[0, 3:].tolist()
            tokens = [t for t in tokens if t not in (16, 1, 2)]
            self.assertEqual(first, str(tokens))
            translator.npu_context_length = 12
            translator.compiled = fixed_compiled
            translator.request = fixed_compiled.create_infer_request()
            tokenizer.pad_token_id = 3
            static_result = translator.translate("Hello", max_new_tokens=3)
            self.assertEqual(static_result, first)
            metrics = translator.last_metrics
            self.assertEqual(metrics["generated_tokens"], len(tokens))
            self.assertEqual(metrics["generated_token_ids"], tokens)
            self.assertEqual(metrics["prompt_tokens"], 3)
            self.assertEqual(metrics["inference_calls"], len(metrics["inference_call_seconds"]))
            self.assertLessEqual(metrics["inference_seconds"], metrics["translation_seconds"])
            if tokens:
                self.assertGreater(metrics["first_token_seconds"], 0)
                self.assertLessEqual(metrics["first_token_seconds"], metrics["translation_seconds"])
            with self.assertRaises(ValueError):
                translator.translate("Hello", max_new_tokens=10)
            # Fixed-cache single-token decoding must match the recompute path.
            from .plamo_translate import static_decode_model

            prefill = static_model(ov.Core().read_model(str(path)), 12, keep_cache=True)
            decode = static_decode_model(ov.Core().read_model(str(path)), 12)
            translator.compiled = ov.Core().compile_model(prefill, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            translator.request = translator.compiled.create_infer_request()
            translator.decode_compiled = ov.Core().compile_model(decode, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            translator.decode_request = translator.decode_compiled.create_infer_request()
            translator.npu_kv_cache = True
            self.assertEqual(translator.translate("Hello", max_new_tokens=3), first)
            ids_generated = translator.last_metrics["generated_token_ids"]
            cached_prefix = ids_generated if translator.last_metrics["stopped_on_eos"] else ids_generated[:-1]
            if cached_prefix:
                with torch.no_grad():
                    reference = model(input_ids=torch.tensor([[4, 5, 6, *cached_prefix]]),
                                      use_cache=False, logits_to_keep=1)
                np.testing.assert_allclose(translator.decode_request.get_tensor("logits").data,
                                           reference.logits.numpy(), rtol=2e-4, atol=2e-5)
            self.assertEqual(translator.translate("Again", max_new_tokens=3), first)
            # Rotated-cache export is equivalent inside the sliding window.
            model.config.window_size = 16
            rotated_dir = Path(directory) / "rotated"
            rotated_dir.mkdir()
            ov.save_model(convert(model, rotated_cache=True), rotated_dir / "openvino_model.xml",
                          compress_to_fp16=False)
            (rotated_dir / "plamo_openvino.json").write_text(json.dumps({
                "context_length": 16, "num_key_value_heads": 1, "head_dim": 128, "cache_mode": "rotated",
            }), encoding="utf-8")
            tokenizer.convert_tokens_to_ids.side_effect = [16, 1, 2]
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                rotated_translator = Translator(rotated_dir, "CPU")
            dynamic_rotated = rotated_translator.translate("Hello", max_new_tokens=3)
            with torch.no_grad():
                reference_ids = model.generate(torch.tensor([[4, 5, 6]]),
                    do_sample=False, max_new_tokens=3, eos_token_id=[16, 1, 2])
            self.assertEqual(dynamic_rotated, str([t for t in reference_ids[0, 3:].tolist() if t not in (16, 1, 2)]))
            rotated_translator.npu_context_length = 16
            rotated_translator.npu_kv_cache = True
            rotated_path = str(rotated_dir / "openvino_model.xml")
            rotated_translator.compiled = ov.Core().compile_model(
                static_model(ov.Core().read_model(rotated_path), 16, keep_cache=True),
                "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            rotated_translator.request = rotated_translator.compiled.create_infer_request()
            rotated_translator.decode_compiled = ov.Core().compile_model(
                static_decode_model(ov.Core().read_model(rotated_path), 16),
                "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            rotated_translator.decode_request = rotated_translator.decode_compiled.create_infer_request()
            self.assertEqual(rotated_translator.translate("Hello", max_new_tokens=3), dynamic_rotated)
            self.assertEqual(rotated_translator.translate("Again", max_new_tokens=3), dynamic_rotated)
            generated = rotated_translator.last_metrics["generated_token_ids"]
            prefix = generated if rotated_translator.last_metrics["stopped_on_eos"] else generated[:-1]
            if prefix:
                with torch.no_grad():
                    expected = model(input_ids=torch.tensor([[4, 5, 6, *prefix]]),
                                     use_cache=False, logits_to_keep=1)
                np.testing.assert_allclose(rotated_translator.decode_request.get_tensor("logits").data,
                                           expected.logits.numpy(), rtol=2e-4, atol=2e-5)
            # Check the same FP16 -> INT4 path used by the default exporter.
            import nncf

            compressed = nncf.compress_weights(convert(model.half()),
                mode=nncf.CompressWeightsMode.INT4_SYM, ratio=1.0, group_size=128)
            path = Path(directory) / "compressed.xml"
            ov.save_model(compressed, path)
            quantized = ov.Core().compile_model(str(path), "CPU")
            inputs = {name: np.empty((1, 1, 0, 128), np.float16)
                      for name in values if name.startswith("past_")}
            inputs.update(input_ids=np.array([[4, 5, 6]], np.int64),
                          attention_mask=np.ones((1, 3), np.int64))
            logits = quantized(inputs)[quantized.output("logits")]
            self.assertTrue(np.isfinite(logits).all())
            self.assertEqual(logits.shape, (1, 1, 32))


if __name__ == "__main__":
    unittest.main()
