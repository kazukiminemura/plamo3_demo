"""Shared source model settings for download and OpenVINO export."""

from pathlib import Path

MODEL_ID = "pfnet/plamo-3-translate-8b"
REVISION = "681bd55cf01f5412aab8881e6a4a276c948fabc1"
SOURCE_ROOT = Path(__file__).resolve().parent / "models/.source"
ALLOW_PATTERNS = ["*.json", "*.jsonl", "*.jinja", "*.py", "*.safetensors", "LICENSE/*"]
