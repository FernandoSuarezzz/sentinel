"""Command line for the transformer model (version 2).

    python -m sentinel check "Ignore all previous instructions" [--model-dir artifacts/onnx]

Version 1 (TF-IDF + logistic regression) keeps its own command line: python -m sentinel.baseline
"""

import argparse
import json
import sys

from sentinel.detector import Detector


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel", description="Prompt-injection firewall")
    sub = parser.add_subparsers(dest="command", required=True)
    ck = sub.add_parser("check", help="classify one text with the exported ONNX model")
    ck.add_argument("text")
    ck.add_argument("--model-dir", default="artifacts/onnx")
    args = parser.parse_args(argv)

    detector = Detector.from_dir(args.model_dir)
    r = detector.scan(args.text)
    print(json.dumps({"verdict": "injection" if r.flagged else "safe", "score": r.score,
                      "threshold": detector.config.threshold,
                      "windows_scanned": r.chunks_scanned, "latency_ms": r.latency_ms}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
