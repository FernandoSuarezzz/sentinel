from pathlib import Path

import onnxruntime as ort
import pytest

from sentinel.detector import Detector, DetectorConfig

TINY_MODEL = Path(__file__).parent / "fixtures" / "tiny_model"
TRIGGER = 7  # the tiny model flags any window containing this token id


class WordTokenizer:
    """Deterministic fake tokenizer: one id per word; the word ATTACK maps to TRIGGER."""

    def encode_ids(self, text: str) -> list[int]:
        return [TRIGGER if w == "ATTACK" else 10 + sum(map(ord, w)) % 90 for w in text.split()]


@pytest.fixture
def detector() -> Detector:
    config = DetectorConfig.load(TINY_MODEL)
    session = ort.InferenceSession(str(TINY_MODEL / config.onnx_file),
                                   providers=["CPUExecutionProvider"])
    return Detector(session, WordTokenizer(), config)
