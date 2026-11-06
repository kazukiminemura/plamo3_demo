"""Download model files without converting models or starting inference."""

import argparse
import os
from pathlib import Path
import sys

from gemma.gemma_npu import MODEL_DIR as GEMMA_DIR, MODEL_ID as GEMMA_ID, REVISION as GEMMA_REVISION

ROOT = Path(__file__).resolve().parent
PLAMO_ID = "pfnet/plamo-3-translate-8b"
PLAMO_REVISION = "681bd55cf01f5412aab8881e6a4a276c948fabc1"
PLAMO_DIR = ROOT / "plamo3/models/.source" / PLAMO_REVISION


def main(argv=None):
    parser = argparse.ArgumentParser(description="PLaMo 3・Gemmaのモデルをダウンロード（変換・推論なし）")
    parser.add_argument("model", nargs="?", choices=("all", "plamo3", "gemma"), default="all",
                        help="取得するモデル（既定: all）")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    models = {
        "plamo3": (PLAMO_ID, PLAMO_REVISION, PLAMO_DIR, {
            "allow_patterns": ["*.json", "*.jsonl", "*.jinja", "*.py", "*.safetensors", "LICENSE/*"],
        }),
        "gemma": (GEMMA_ID, GEMMA_REVISION, GEMMA_DIR, {"ignore_patterns": [".gitattributes"]}),
    }
    for name in models if args.model == "all" else (args.model,):
        model_id, revision, destination, patterns = models[name]
        print(f"Downloading {model_id} ({revision}) ...", flush=True)
        try:
            snapshot_download(model_id, revision=revision, local_dir=destination, **patterns)
        except GatedRepoError:
            print(f"取得権限が必要です: https://huggingface.co/{model_id}\n"
                  "利用条件に同意後、uv run hf auth login を実行してください。", file=sys.stderr)
            return 1
        except (HfHubHTTPError, OSError) as exc:
            print(f"ダウンロード失敗: {model_id}: {exc}", file=sys.stderr)
            return 1
        print(f"Downloaded: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
