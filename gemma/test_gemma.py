"""Gemma NPU runner checks without downloading model weights."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class GemmaTest(unittest.TestCase):
    def test_gemma_npu_execution(self):
        from . import gemma_npu
        from huggingface_hub.errors import GatedRepoError

        with tempfile.TemporaryDirectory() as directory, patch("gemma.gemma_npu.__file__", str(Path(directory) / "gemma_npu.py")), patch("huggingface_hub.snapshot_download") as download, patch("openvino_genai.VLMPipeline") as pipeline, patch("openvino.Core") as core, patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            core.return_value.available_devices = ["CPU", "GPU.0", "GPU.1", "NPU"]
            metrics = pipeline.return_value.generate.return_value.perf_metrics
            pipeline.return_value.generate.return_value.texts = ["Translation"]
            metrics.get_num_input_tokens.return_value = 20
            metrics.get_num_generated_tokens.return_value = 12
            metrics.get_ttft.return_value.mean = 1200.0
            metrics.get_throughput.return_value.mean = 30.0
            metrics.get_generate_duration.return_value.mean = 2000.0
            core.return_value.get_property.return_value = "Test NPU"
            self.assertEqual(gemma_npu.main(["Translate Hello"]), 0)
            self.assertEqual(pipeline.call_args.args[1], "NPU")
            self.assertEqual(pipeline.call_args.kwargs["GENERATE_HINT"], "BEST_PERF")
            self.assertEqual(pipeline.return_value.generate.call_count, 2)
            report = json.loads(next((Path(directory) / "baselines").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["ttft_seconds"], 1.2)
            self.assertEqual(report["decode_tokens_per_second"], 30.0)
            for device in ("CPU", "GPU", "GPU.1"):
                with self.subTest(device=device):
                    self.assertEqual(gemma_npu.main(["Translate Hello", "--device", device.lower()]), 0)
                    self.assertEqual(pipeline.call_args.args[1], device)
                    self.assertEqual(pipeline.call_args.kwargs, {})
                    report = json.loads(next((Path(directory) / "baselines").glob(f"gemma3-4b-{device.lower()}-*.json")).read_text(encoding="utf-8"))
                    self.assertEqual(report["device"], device)
                    self.assertNotIn("npu", report)
                    core.return_value.get_property.assert_called_with(device, "FULL_DEVICE_NAME")
            pipeline.reset_mock()
            download.side_effect = GatedRepoError("Access required")
            self.assertEqual(gemma_npu.main(["Translate Hello"]), 1)
            pipeline.assert_not_called()

    def test_missing_device_does_not_download(self):
        from . import gemma_npu

        with patch("openvino.Core") as core, patch("huggingface_hub.snapshot_download") as download, patch("sys.stderr", new_callable=io.StringIO):
            core.return_value.available_devices = ["CPU"]
            with self.assertRaises(SystemExit) as error:
                gemma_npu.main(["--device", "GPU"])
            self.assertEqual(error.exception.code, 2)
            download.assert_not_called()



if __name__ == "__main__":
    unittest.main()
