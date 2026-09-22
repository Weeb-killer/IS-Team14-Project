"""Tests for the batched prediction loop.

Most tests drive a deterministic fake tokenizer and model rather than a real transformer. The
logic under test is batching, ordering and collection, not BERT, and a real model built from a
hand-written vocabulary maps every request to the same ``[UNK]`` sequence, which would make
assertions about per-row output pass without meaning anything. One test at the end exercises a
genuine Hugging Face model so the pipeline is still proven against the real interface.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from http_attack_agent.data import DatasetConfig, SplitConfig

LABELS = {"alpha": "label_alpha", "beta": "label_beta"}
TEXT = {"method": "method", "path": "path"}
ROWS = 10


def _frame(rows: int = ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rid": [f"r{index}" for index in range(rows)],
            "method": ["GET", "POST"] * (rows // 2),
            # Deliberately varied lengths so a real tokenizer produces differing outputs.
            "path": ["/" + "/".join("abcdefghij"[: index % 5 + 1]) for index in range(rows)],
            "label_alpha": [index % 2 for index in range(rows)],
            "label_beta": [int(index % 3 == 0) for index in range(rows)],
        }
    )


def _config(label_columns=None) -> DatasetConfig:
    return DatasetConfig(
        path=Path("unused.csv"),
        format="csv",
        id_column="rid",
        text_columns=dict(TEXT),
        label_columns=dict(LABELS) if label_columns is None else label_columns,
        waf_concept_columns={},
        split=SplitConfig(),
    )


def _fingerprint(text: str) -> int:
    """A small, stable token id derived from the request text."""

    return sum(ord(character) for character in text) % 97 + 1


class FakeTokenizer:
    """Encode each request as a run of one repeated, text-dependent token id."""

    def __call__(self, texts, padding=True, truncation=True, max_length=16, return_tensors="pt"):
        import torch

        ids = [_fingerprint(text) for text in texts]
        lengths = [min(value % 5 + 2, max_length) for value in ids]
        width = max(lengths)
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), width, dtype=torch.long)
        for row, (value, length) in enumerate(zip(ids, lengths)):
            input_ids[row, :length] = value
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeModel:
    """Return logits that are an exact, checkable function of the batch."""

    hidden_size = 4

    def __init__(self) -> None:
        self.eval_calls = 0
        self.devices: list = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, input_ids, attention_mask):
        import torch

        from http_attack_agent.models.hf_classifier import ClassifierOutput

        masked = (input_ids * attention_mask).sum(dim=1).to(torch.float32)
        logits = torch.stack([masked / 100.0, -masked / 100.0], dim=1)
        embeddings = masked.unsqueeze(1).repeat(1, self.hidden_size) / 100.0
        return ClassifierOutput(logits=logits, embeddings=embeddings)


def _expected_logit(text: str) -> float:
    value = _fingerprint(text)
    return value * min(value % 5 + 2, 16) / 100.0


def _serialized(frame: pd.DataFrame, index: int) -> str:
    row = frame.iloc[index]
    return f"[METHOD]\n{row['method']}\n[PATH]\n{row['path']}"


def _predict(**kwargs):
    from http_attack_agent.evaluation.predict import predict_dataset

    pytest.importorskip("torch")
    import torch

    options = {
        "max_length": 16,
        "batch_size": 4,
        "device": torch.device("cpu"),
        "progress": False,
    }
    options.update(kwargs)
    config = options.pop("config", None) or _config()
    frame = options.pop("frame", None)
    frame = _frame() if frame is None else frame
    return predict_dataset(FakeModel(), FakeTokenizer(), frame, config, **options)


def test_probabilities_are_the_sigmoid_of_the_model_logits():
    frame = _frame()

    result = _predict(frame=frame)

    expected = np.array([_expected_logit(_serialized(frame, i)) for i in range(ROWS)])
    assert np.allclose(result.logits[:, 0], expected, atol=1e-5)
    assert np.allclose(result.probabilities, 1.0 / (1.0 + np.exp(-result.logits)), atol=1e-6)


def test_rows_receive_distinct_predictions():
    """Guards against a fixture where every request collapses to the same input."""

    result = _predict()

    assert len(np.unique(result.probabilities.round(6), axis=0)) > 1


def test_row_order_follows_the_requested_indices():
    frame = _frame()

    result = _predict(frame=frame, indices=np.array([3, 1, 7]))

    assert result.row_ids.tolist() == ["r3", "r1", "r7"]
    expected = [_expected_logit(_serialized(frame, i)) for i in (3, 1, 7)]
    assert np.allclose(result.logits[:, 0], expected, atol=1e-5)


def test_batch_size_does_not_change_the_result():
    """Padding width varies with batch composition; masking must absorb it."""

    one = _predict(batch_size=1)
    four = _predict(batch_size=4)
    whole = _predict(batch_size=ROWS)

    assert np.allclose(one.probabilities, four.probabilities, atol=1e-6)
    assert np.allclose(one.probabilities, whole.probabilities, atol=1e-6)


def test_targets_are_attached_when_the_configuration_declares_labels():
    result = _predict(indices=np.array([0, 1, 2]))

    assert result.has_targets
    assert result.targets.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]


def test_unlabeled_configuration_yields_predictions_but_no_targets():
    """The inference case: requests arrive without ground truth.

    The model still emits one probability per label it was trained on; only the comparison
    against ground truth disappears.
    """

    labeled = _predict()
    unlabeled = _predict(config=_config(label_columns={}))

    assert unlabeled.targets is None
    assert not unlabeled.has_targets
    assert unlabeled.probabilities.shape == (ROWS, 2), "the model still scores both labels"
    assert np.allclose(unlabeled.probabilities, labeled.probabilities)


def test_embeddings_are_returned_only_when_requested():
    without = _predict()
    with_embeddings = _predict(return_embeddings=True)

    assert without.embeddings is None
    assert with_embeddings.embeddings.shape == (ROWS, FakeModel.hidden_size)
    assert np.allclose(without.probabilities, with_embeddings.probabilities)


def test_model_is_switched_to_eval_mode_and_moved_to_the_device():
    from http_attack_agent.evaluation.predict import predict_batches, build_loader

    pytest.importorskip("torch")
    import torch

    model = FakeModel()
    loader = build_loader(
        _frame(),
        FakeTokenizer(),
        text_columns=TEXT,
        targets=np.zeros((ROWS, 2), dtype=np.float32),
        row_ids=[f"r{index}" for index in range(ROWS)],
        indices=np.arange(ROWS),
        max_length=16,
        batch_size=4,
    )

    predict_batches(model, loader, device=torch.device("cpu"), progress=False)

    assert model.eval_calls == 1
    assert model.devices == [torch.device("cpu")]


def test_empty_selection_raises():
    with pytest.raises(ValueError, match="no batches"):
        _predict(indices=np.empty(0, dtype=int))


def test_resolve_device_honours_the_cpu_preference():
    pytest.importorskip("torch")

    from http_attack_agent.evaluation.predict import resolve_device

    assert resolve_device(prefer_cpu=True).type == "cpu"


def test_pipeline_runs_against_a_real_transformer(tmp_path):
    """Smoke test against the genuine Hugging Face interface, not a fake."""

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from http_attack_agent.evaluation.predict import predict_dataset
    from http_attack_agent.models.hf_classifier import build_classifier

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocab))
    model = build_classifier(
        "offline/tiny-bert",
        num_labels=2,
        dropout=0.0,
        backbone_config=transformers.BertConfig(
            vocab_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        ),
    )

    result = predict_dataset(
        model,
        tokenizer,
        _frame(),
        _config(),
        max_length=16,
        batch_size=4,
        device=torch.device("cpu"),
        progress=False,
        return_embeddings=True,
    )

    assert result.probabilities.shape == (ROWS, 2)
    assert result.embeddings.shape == (ROWS, 16)
    assert result.row_ids.tolist() == [f"r{index}" for index in range(ROWS)]
    assert ((result.probabilities >= 0.0) & (result.probabilities <= 1.0)).all()
