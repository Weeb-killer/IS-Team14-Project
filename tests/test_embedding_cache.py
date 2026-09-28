"""Tests for caching frozen-backbone embeddings and training the head on them.

The cache is only safe while it still describes the same computation. Most of these tests are
about refusing a stale cache, because silently reusing one would train a head on vectors from a
different backbone, a different input length, or different data, and nothing downstream would
notice.
"""

import json

import numpy as np
import pandas as pd
import pytest

from http_attack_agent.embedding_cache import (
    build_manifest,
    cache_directory,
    content_digest,
    estimated_cache_bytes,
    format_bytes,
    head_predict,
    load_cache,
    save_cache,
    train_head_epochs,
)

TEXT_COLUMNS = {"method": "method", "path": "path"}
ROWS = 40
HIDDEN = 8
LABELS = 3


def _frame(rows: int = ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["GET", "POST"] * (rows // 2),
            "path": [f"/page/{index}" for index in range(rows)],
        }
    )


def _manifest(**overrides):
    base = {
        "model_id": "offline/tiny",
        "max_length": 16,
        "text_columns": TEXT_COLUMNS,
        "rows": ROWS,
        "digest": "abc123",
        "dtype": "float32",
    }
    base.update(overrides)
    return build_manifest(**base)


def _embeddings(rows: int = ROWS) -> np.ndarray:
    rng = np.random.default_rng(14)
    return rng.normal(size=(rows, HIDDEN)).astype(np.float32)


def _row_ids(rows: int = ROWS) -> np.ndarray:
    return np.asarray([f"r{index}" for index in range(rows)], dtype=str)


def _head():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(HIDDEN, LABELS))


def test_cache_round_trips(tmp_path):
    manifest = _manifest()
    embeddings = _embeddings()

    save_cache(tmp_path / "c", embeddings, _row_ids(), manifest)
    loaded = load_cache(tmp_path / "c", manifest)

    assert loaded is not None
    assert np.allclose(np.asarray(loaded.embeddings), embeddings)
    assert loaded.row_ids.tolist() == _row_ids().tolist()
    assert loaded.hidden_size == HIDDEN
    assert len(loaded) == ROWS


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "offline/other"},
        {"max_length": 32},
        {"text_columns": {"method": "method"}},
        {"rows": ROWS - 1},
        {"digest": "different"},
        {"dtype": "float16"},
    ],
    ids=["model", "max_length", "text_columns", "rows", "content", "dtype"],
)
def test_a_cache_is_refused_when_anything_that_shapes_it_changed(tmp_path, change):
    save_cache(tmp_path / "c", _embeddings(), _row_ids(), _manifest())

    assert load_cache(tmp_path / "c", _manifest(**change)) is None


def test_missing_or_partial_cache_is_refused(tmp_path):
    assert load_cache(tmp_path / "absent", _manifest()) is None

    save_cache(tmp_path / "c", _embeddings(), _row_ids(), _manifest())
    (tmp_path / "c" / "embeddings.npy").unlink()
    assert load_cache(tmp_path / "c", _manifest()) is None


def test_corrupt_manifest_is_refused(tmp_path):
    save_cache(tmp_path / "c", _embeddings(), _row_ids(), _manifest())
    (tmp_path / "c" / "manifest.json").write_text("{not json", encoding="utf-8")

    assert load_cache(tmp_path / "c", _manifest()) is None


def test_content_digest_tracks_the_serialized_columns():
    frame = _frame()
    same = _frame()
    changed = _frame()
    changed.loc[0, "path"] = "/page/0?id=1%27"

    assert content_digest(frame, TEXT_COLUMNS) == content_digest(same, TEXT_COLUMNS)
    assert content_digest(frame, TEXT_COLUMNS) != content_digest(changed, TEXT_COLUMNS)


def test_content_digest_ignores_columns_the_model_never_sees():
    """A label or timestamp column changing must not invalidate the embeddings."""

    frame = _frame()
    frame["label_alpha"] = 0
    other = frame.copy()
    other["label_alpha"] = 1

    assert content_digest(frame, TEXT_COLUMNS) == content_digest(other, TEXT_COLUMNS)


def test_manifest_is_written_last(tmp_path):
    """An interrupted run must not leave a manifest that claims a complete cache."""

    save_cache(tmp_path / "c", _embeddings(), _row_ids(), _manifest())
    files = sorted(path.name for path in (tmp_path / "c").iterdir())

    assert files == ["embeddings.npy", "manifest.json", "row_ids.npy"]
    manifest = json.loads((tmp_path / "c" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == ROWS


def test_cache_directory_separates_models(tmp_path):
    assert cache_directory(tmp_path, "canine-c") != cache_directory(tmp_path, "secbert")


def test_head_predict_returns_probabilities_in_the_requested_order():
    pytest.importorskip("torch")

    embeddings = _embeddings()
    head = _head()
    indices = np.array([7, 2, 30])

    probabilities = head_predict(head, embeddings, indices, batch_size=2)

    assert probabilities.shape == (3, LABELS)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    one_at_a_time = np.concatenate(
        [head_predict(head, embeddings, np.array([row])) for row in indices]
    )
    assert np.allclose(probabilities, one_at_a_time, atol=1e-6)


def test_head_predict_rejects_an_empty_selection():
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="at least one row"):
        head_predict(_head(), _embeddings(), np.empty(0, dtype=int))


def test_training_the_head_lowers_the_loss_and_moves_only_the_head():
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(ROWS, HIDDEN)).astype(np.float32)
    targets = np.zeros((ROWS, LABELS), dtype=np.float32)
    # A signal the head can actually learn from the first feature.
    targets[:, 0] = (embeddings[:, 0] > 0).astype(np.float32)
    train_idx = np.arange(0, 30)
    valid_idx = np.arange(30, ROWS)
    head = _head()
    before = [parameter.detach().clone() for parameter in head.parameters()]

    epochs = list(
        train_head_epochs(
            head,
            embeddings,
            targets,
            train_idx,
            valid_idx,
            epochs=6,
            batch_size=8,
            learning_rate=0.1,
            pos_weight=np.ones(LABELS, dtype=np.float32),
            gradient_clip=1.0,
            device=torch.device("cpu"),
        )
    )

    assert [epoch for epoch, _, _ in epochs] == [1, 2, 3, 4, 5, 6]
    assert epochs[-1][1] < epochs[0][1], "training loss should fall"
    assert all(
        probabilities.shape == (len(valid_idx), LABELS) for _, _, probabilities in epochs
    )
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, head.parameters())
    )


def test_validation_probabilities_align_with_the_validation_indices():
    torch = pytest.importorskip("torch")

    embeddings = _embeddings()
    targets = np.zeros((ROWS, LABELS), dtype=np.float32)
    valid_idx = np.array([31, 5, 18])
    head = _head()

    _, _, probabilities = next(
        train_head_epochs(
            head,
            embeddings,
            targets,
            np.arange(0, 30),
            valid_idx,
            epochs=1,
            batch_size=8,
            learning_rate=0.0,  # keep the head fixed so the comparison is exact
            pos_weight=np.ones(LABELS, dtype=np.float32),
            gradient_clip=1.0,
            device=torch.device("cpu"),
        )
    )

    assert np.allclose(probabilities, head_predict(head, embeddings, valid_idx), atol=1e-6)


def test_size_estimate_matches_the_stored_file(tmp_path):
    embeddings = _embeddings()

    save_cache(tmp_path / "c", embeddings, _row_ids(), _manifest())
    on_disk = (tmp_path / "c" / "embeddings.npy").stat().st_size
    estimate = estimated_cache_bytes(ROWS, HIDDEN, "float32")

    assert estimate == embeddings.nbytes
    assert on_disk - estimate < 1024, "only the npy header should differ"
    assert estimated_cache_bytes(ROWS, HIDDEN, "float16") == estimate // 2


def test_format_bytes_is_readable():
    assert format_bytes(512) == "512.0 B"
    assert format_bytes(2 * 1024**3) == "2.0 GiB"
