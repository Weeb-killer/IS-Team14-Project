import numpy as np

from http_attack_agent.explain.analysis import (
    EmbeddingBundle,
    fit_concept_probe,
    prototype_report,
)


def test_embedding_bundle_round_trip_does_not_require_pickle(tmp_path):
    path = tmp_path / "embeddings.npz"
    original = EmbeddingBundle(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        logits=np.asarray([[0.5]], dtype=np.float32),
        labels=np.asarray([[1]], dtype=np.int8),
        concepts=np.asarray([[1]], dtype=np.int8),
        # Reproduce the object dtype commonly returned by pandas string Series.
        row_ids=np.asarray(["request-1"], dtype=object),
        label_names=["attack"],
        concept_names=["waf_attack"],
    )

    original.save(path)
    loaded = EmbeddingBundle.load(str(path))

    assert loaded.row_ids.tolist() == ["request-1"]
    assert loaded.row_ids.dtype.kind == "U"


def test_probe_finds_linearly_encoded_concept():
    rng = np.random.default_rng(4)
    target = np.repeat([0, 1], 60)
    embeddings = rng.normal(size=(120, 8))
    embeddings[:, 0] += target * 4.0
    report, cav = fit_concept_probe(
        embeddings, target, folds=3, permutations=3, seed=3
    )
    assert report["roc_auc"] > 0.9
    assert cav is not None
    assert abs(np.linalg.norm(cav) - 1.0) < 1e-5


def test_prototype_returns_real_row_ids():
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [-1, 0], [-0.9, 0.1]])
    target = np.asarray([1, 1, 0, 0])
    report = prototype_report(embeddings, target, np.asarray(["a", "b", "c", "d"]))
    assert report["status"] == "ok"
    assert set(report["prototype_row_ids"]).issubset({"a", "b"})
