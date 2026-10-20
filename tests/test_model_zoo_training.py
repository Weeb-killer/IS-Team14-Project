"""Exercise every enabled architecture without downloading pretrained weights."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from http_attack_agent.data import (
    DatasetConfig,
    SplitConfig,
    build_targets,
    deterministic_split_indices,
)
from http_attack_agent.models.zoo import MODEL_ZOO, resolve_model_names


def _tiny_components(name, transformers):
    if name == "canine-c":
        return (
            transformers.CanineConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                num_hash_functions=4,
                num_hash_buckets=64,
                max_position_embeddings=128,
                local_transformer_stride=16,
            ),
            transformers.CanineTokenizer(model_max_length=32),
        )
    if name == "byt5-small":
        return (
            transformers.T5Config(
                vocab_size=259,
                d_model=32,
                d_ff=64,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=4,
                d_kv=8,
            ),
            transformers.ByT5Tokenizer(extra_ids=0, model_max_length=32),
        )

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "GET", "normal", "sql", "xss"]
        )
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    common = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 128,
    }
    if name == "securebert2":
        config = transformers.ModernBertConfig(
            **common,
            pad_token_id=0,
            cls_token_id=2,
            sep_token_id=3,
            bos_token_id=2,
            eos_token_id=3,
            local_attention=16,
        )
    else:
        config = transformers.BertConfig(**common)
    return config, tokenizer


def test_all_selects_only_http_compatible_models():
    assert resolve_model_names(["all"]) == ["canine-c", "byt5-small", "securebert2", "secbert"]
    for names in (["et-bert"], ["netfound-small"], ["all", "secbert"]):
        with pytest.raises(ValueError):
            resolve_model_names(names)


def test_all_command_retains_completed_results_on_later_failure(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from http_attack_agent import training

    frame = pd.DataFrame({"attack": [0] * 6 + [1] * 6})
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column=None,
        text_columns={},
        label_columns={"attack": "attack"},
        waf_concept_columns={},
        split=SplitConfig(test_size=0.25, validation_size=0.25),
    )
    monkeypatch.setattr(training.DatasetConfig, "from_yaml", lambda _: config)
    monkeypatch.setattr(training, "load_table", lambda _: frame)
    monkeypatch.setattr(
        "sys.argv",
        [
            "http-attack-train",
            "--dataset-config",
            "unused.yaml",
            "--model",
            "all",
            "--output",
            str(tmp_path),
        ],
    )
    called = []
    shared_splits = []

    def fake_train(name, config, frame, targets, concepts, splits, output, args):
        called.append(name)
        shared_splits.append(splits)
        if len(called) == 3:
            raise RuntimeError("Model download interrupted")
        return {"model": name, "micro_f1": 0.5}

    monkeypatch.setattr(training, "train_one", fake_train)
    with pytest.raises(RuntimeError, match="interrupted"):
        training.main()
    saved = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert [entry["model"] for entry in saved] == ["canine-c", "byt5-small"]
    assert called == ["canine-c", "byt5-small", "securebert2"]
    assert all(splits is shared_splits[0] for splits in shared_splits)


@pytest.mark.parametrize("model_name", resolve_model_names(["all"]))
@pytest.mark.parametrize("regime", ["full", "frozen", "cached"])
def test_each_model_trains_evaluates_and_reloads(tmp_path, monkeypatch, model_name, regime):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from http_attack_agent import training
    from http_attack_agent.evaluation.predict import predict_dataset
    from http_attack_agent.models.checkpoint import load_local_checkpoint
    from http_attack_agent.models.hf_classifier import build_classifier

    backbone_config, tokenizer = _tiny_components(model_name, transformers)
    monkeypatch.setattr(training, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        training,
        "build_classifier",
        lambda model_id, num_labels, dropout: build_classifier(
            model_id, num_labels, dropout, backbone_config=backbone_config
        ),
    )
    frame = pd.DataFrame(
        {
            "method": ["GET"] * 24,
            "path": ["/normal"] * 12 + ["/sql"] * 6 + ["/xss"] * 6,
            "sql": [0] * 12 + [1] * 6 + [0] * 6,
            "xss": [0] * 18 + [1] * 6,
        }
    )
    config = DatasetConfig(
        path=tmp_path / "unused.csv",
        format="csv",
        id_column=None,
        text_columns={"method": "method", "path": "path"},
        label_columns={"sql": "sql", "xss": "xss"},
        waf_concept_columns={},
        split=SplitConfig(test_size=0.25, validation_size=0.25, strategy="multilabel_stratified"),
    )
    splits = deterministic_split_indices(frame, config)
    args = SimpleNamespace(
        model=resolve_model_names(["all"]),
        epochs=1,
        batch_size=3,
        max_length=32,
        learning_rate=1e-3,
        dropout=0.0,
        gradient_clip=1.0,
        max_pos_weight=50.0,
        cpu=True,
        freeze_backbone=regime != "full",
        cache_embeddings=regime == "cached",
        embedding_cache=None,
        encode_batch_size=4,
        head_batch_size=4,
        cache_dtype="float32",
        refresh_cache=False,
        max_rows=None,
    )
    result = training.train_one(
        model_name,
        config,
        frame,
        build_targets(frame, config.label_columns),
        None,
        splits,
        tmp_path,
        args,
    )
    artifact = tmp_path / model_name
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    assert np.isfinite(metadata["history"][0]["train_loss"])
    assert "normal" in result and "roc_auc" in result["per_label"]["sql"]
    assert metadata["cached_embeddings"] == (regime == "cached")
    assert metadata["freeze_backbone"] == (regime != "full")

    def no_download(*args, **kwargs):
        raise AssertionError("Reloading a local artifact must not download model weights")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", no_download)
    monkeypatch.setattr(transformers.T5EncoderModel, "from_pretrained", no_download)
    model, local_tokenizer, _ = load_local_checkpoint(artifact)
    if model_name == "byt5-small":
        assert type(model.backbone).__name__ == "T5EncoderModel"
        assert not hasattr(model.backbone, "decoder")
    prediction = predict_dataset(
        model,
        local_tokenizer,
        frame,
        config,
        indices=splits[2],
        max_length=32,
        batch_size=3,
        device=torch.device("cpu"),
        progress=False,
        return_embeddings=True,
    )
    with np.load(artifact / "test_embeddings.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(prediction.logits, saved["logits"], atol=1e-5)
        np.testing.assert_allclose(prediction.embeddings, saved["embeddings"], atol=1e-5)
    assert prediction.probabilities.shape == (len(splits[2]), len(config.label_columns))


def test_legacy_byt5_decoder_checkpoint_still_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from http_attack_agent.models.checkpoint import load_local_checkpoint, save_local_checkpoint
    from http_attack_agent.models.hf_classifier import build_classifier

    config, tokenizer = _tiny_components("byt5-small", transformers)
    model = build_classifier(
        MODEL_ZOO["byt5-small"].model_id, 2, backbone_config=config, encoder_only=False
    )
    model.eval()
    inputs = tokenizer("GET /normal", return_tensors="pt")
    with torch.no_grad():
        expected = model(inputs["input_ids"], inputs["attention_mask"]).logits
    save_local_checkpoint(
        model,
        tokenizer,
        tmp_path,
        {
            "model": {"model_id": MODEL_ZOO["byt5-small"].model_id},
            "label_names": ["sql", "xss"],
        },
    )
    # Remove the new architecture field to reproduce a pre-change artifact.
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("backbone_class")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    restored, _, _ = load_local_checkpoint(tmp_path)
    with torch.no_grad():
        actual = restored(inputs["input_ids"], inputs["attention_mask"]).logits
    torch.testing.assert_close(actual, expected)
