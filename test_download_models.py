"""Check download routing and failures without fetching model weights."""

import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import download_models as downloader


class DownloadModelsTest(unittest.TestCase):
    def test_export_reuses_download_source_with_custom_output(self):
        from plamo3.export_model import export_model
        from plamo3.plamo_translate import parser

        for requested_revision in (downloader.PLAMO_REVISION, "main"):
            with self.subTest(revision=requested_revision), tempfile.TemporaryDirectory() as directory:
                args = parser().parse_args(["export", "--model-dir", str(Path(directory) / "custom-ir")])
                self.assertEqual(args.revision, downloader.PLAMO_REVISION)
                args.revision = requested_revision
                with patch("huggingface_hub.snapshot_download", return_value=str(downloader.PLAMO_DIR)) as download, patch(
                    "huggingface_hub.model_info", return_value=SimpleNamespace(sha=downloader.PLAMO_REVISION)
                ) as info, patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("stop before conversion")) as load, patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(downloader.main(["plamo3"]), 0)
                    downloaded = download.call_args
                    with self.assertRaisesRegex(RuntimeError, "stop before conversion"):
                        export_model(args)
                    self.assertEqual(download.call_count, 2)
                    self.assertEqual(download.call_args, downloaded)
                    load.assert_called_once_with(str(downloader.PLAMO_DIR), trust_remote_code=True)
                    self.assertEqual(info.call_count, int(requested_revision == "main"))

    def test_all_uses_inference_and_export_locations(self):
        with patch("huggingface_hub.snapshot_download") as download, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(downloader.main([]), 0)
        self.assertEqual(download.call_count, 2)
        plamo, gemma = download.call_args_list
        self.assertEqual(plamo.args, (downloader.PLAMO_ID,))
        self.assertEqual(plamo.kwargs["revision"], downloader.PLAMO_REVISION)
        self.assertEqual(plamo.kwargs["local_dir"], downloader.PLAMO_DIR)
        self.assertIn("*.safetensors", plamo.kwargs["allow_patterns"])
        self.assertEqual(gemma.args, (downloader.GEMMA_ID,))
        self.assertEqual(gemma.kwargs["revision"], downloader.GEMMA_REVISION)
        self.assertEqual(gemma.kwargs["local_dir"], downloader.GEMMA_DIR)

    def test_single_model_and_retry(self):
        with patch("huggingface_hub.snapshot_download") as download, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(downloader.main(["plamo3"]), 0)
            self.assertEqual(downloader.main(["plamo3"]), 0)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0], download.call_args_list[1])

    def test_download_errors_return_failure(self):
        from huggingface_hub.errors import GatedRepoError

        for error in (GatedRepoError("Access required"), OSError("Disk full")):
            with self.subTest(error=type(error).__name__), patch("huggingface_hub.snapshot_download", side_effect=error) as download, patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(downloader.main(["gemma"]), 1)
                download.assert_called_once()
                expected = "uv run hf auth login" if isinstance(error, GatedRepoError) else "Disk full"
                self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
