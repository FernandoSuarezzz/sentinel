"""Production inference: ONNX Runtime + the `tokenizers` library, no PyTorch.

Long inputs are a known evasion trick: an attacker pads a review with harmless text so the
injection sits past the model's 256-token window and gets truncated away. We never truncate.
Instead the text is split into overlapping windows, every window is scored in one batch,
and the text's score is the *maximum* window score (one malicious window is enough).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

CONFIG_FILE = "sentinel_config.json"


@dataclass
class DetectorConfig:
    cls_id: int
    sep_id: int
    pad_id: int
    max_length: int = 256  # tokens per window, including the two special tokens
    stride: int = 64  # overlap between consecutive windows, in tokens
    threshold: float = 0.5  # set by `sentinel.evaluate` on the validation set
    max_chunks: int = 64  # hard cap per text: bounds the cost of a single request
    onnx_file: str = "model.int8.onnx"
    version: str = "dev"

    @classmethod
    def load(cls, model_dir: str | Path) -> DetectorConfig:
        return cls(**json.loads((Path(model_dir) / CONFIG_FILE).read_text()))

    def save(self, model_dir: str | Path) -> None:
        (Path(model_dir) / CONFIG_FILE).write_text(json.dumps(asdict(self), indent=2))


class Tokenizer(Protocol):
    def encode_ids(self, text: str) -> list[int]:
        """Token ids for `text`, without special tokens."""


class HFTokenizer:
    """Thin wrapper over a Hugging Face `tokenizer.json` (fast, Rust-based)."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _Tok

        self._tok = _Tok.from_file(str(path))
        self._tok.no_truncation()  # we do our own windowing
        self._tok.no_padding()

    def encode_ids(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


@dataclass
class ScanResult:
    score: float  # probability of injection for the most suspicious window
    flagged: bool  # score >= threshold
    chunks_scanned: int
    flagged_chunk: int | None  # index of the most suspicious window, if flagged
    latency_ms: float


class InputTooLongError(ValueError):
    pass


def chunk_ids(ids: Sequence[int], window: int, stride: int) -> list[list[int]]:
    """Split ids into windows of `window` tokens; consecutive windows share `stride` tokens."""
    if window <= stride:
        raise ValueError("window must be larger than stride")
    if len(ids) <= window:
        return [list(ids)]
    step = window - stride
    chunks, start = [], 0
    while True:
        chunks.append(list(ids[start:start + window]))
        if start + window >= len(ids):
            return chunks
        start += step


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    """P(injection) from [N, 2] logits; numerically stable form of softmax(...)[:, 1]."""
    return 1.0 / (1.0 + np.exp(logits[:, 0] - logits[:, 1]))


class Detector:
    def __init__(self, session, tokenizer: Tokenizer, config: DetectorConfig):
        self.session = session
        self.tokenizer = tokenizer
        self.config = config
        self._input_names = {i.name for i in session.get_inputs()}

    @classmethod
    def from_dir(cls, model_dir: str | Path, threads: int | None = None) -> Detector:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        config = DetectorConfig.load(model_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = threads or int(os.environ.get("SENTINEL_THREADS", "0"))
        if threads:
            opts.intra_op_num_threads = threads
        session = ort.InferenceSession(str(model_dir / config.onnx_file), opts,
                                       providers=["CPUExecutionProvider"])
        return cls(session, HFTokenizer(model_dir / "tokenizer.json"), config)

    def _windows(self, text: str) -> list[list[int]]:
        c = self.config
        chunks = chunk_ids(self.tokenizer.encode_ids(text), c.max_length - 2, c.stride)
        if len(chunks) > c.max_chunks:
            raise InputTooLongError(f"text needs {len(chunks)} windows; limit is {c.max_chunks}")
        return [[c.cls_id, *chunk, c.sep_id] for chunk in chunks]

    def _score_windows(self, windows: list[list[int]]) -> np.ndarray:
        longest = max(len(w) for w in windows)
        ids = np.full((len(windows), longest), self.config.pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, w in enumerate(windows):
            ids[row, :len(w)] = w
            mask[row, :len(w)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        (logits,) = self.session.run(["logits"], feeds)
        return _softmax_positive(logits.astype(np.float64))

    def scan_batch(self, texts: Sequence[str]) -> list[ScanResult]:
        """Score many texts with a single model call over all their windows."""
        start = time.perf_counter()
        per_text = [self._windows(t) for t in texts]
        scores = self._score_windows([w for windows in per_text for w in windows])
        elapsed_ms = (time.perf_counter() - start) * 1000

        results, offset = [], 0
        for windows in per_text:
            s = scores[offset:offset + len(windows)]
            offset += len(windows)
            best = int(np.argmax(s))
            flagged = bool(s[best] >= self.config.threshold)
            results.append(ScanResult(score=round(float(s[best]), 6), flagged=flagged,
                                      chunks_scanned=len(windows),
                                      flagged_chunk=best if flagged else None,
                                      latency_ms=round(elapsed_ms, 2)))
        return results

    def scan(self, text: str) -> ScanResult:
        return self.scan_batch([text])[0]
