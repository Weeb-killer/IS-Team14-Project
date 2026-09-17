import pytest

from http_attack_agent.models.checkpoint import (
    load_local_checkpoint,
    save_local_checkpoint,
)
from http_attack_agent.models.hf_classifier import build_classifier


def test_local_checkpoint_reloads_without_pretrained_download(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    config = transformers.BertConfig(
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = build_classifier(
        "offline/tiny-bert",
        num_labels=2,
        dropout=0.0,
        backbone_config=config,
    )
    model.eval()

    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nattack\n",
        encoding="utf-8",
    )
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocab_path))
    input_ids = torch.tensor([[2, 5, 6, 3]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = model(input_ids, attention_mask).logits.clone()

    checkpoint_dir = tmp_path / "saved-model"
    metadata = {
        "model": {"model_id": "offline/tiny-bert"},
        "label_names": ["sql_injection", "path_traversal"],
        "dropout": 0.0,
        "max_length": 4,
    }
    save_local_checkpoint(model, tokenizer, checkpoint_dir, metadata)

    def no_remote_model(*args, **kwargs):
        raise AssertionError("A saved model must not fetch pretrained weights")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", no_remote_model)
    restored, restored_tokenizer, restored_metadata = load_local_checkpoint(checkpoint_dir)
    with torch.no_grad():
        actual = restored(input_ids, attention_mask).logits

    torch.testing.assert_close(actual, expected)
    assert restored_metadata["label_names"] == metadata["label_names"]
    assert restored_tokenizer.convert_tokens_to_ids("hello") == 5
    assert (checkpoint_dir / "model.pt").is_file()
    assert (checkpoint_dir / "backbone_config" / "config.json").is_file()


def test_best_epoch_snapshot_is_not_mutated_by_later_training():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    from http_attack_agent.training import _cpu_state_snapshot

    model = torch.nn.Linear(3, 2)
    snapshot = _cpu_state_snapshot(model)
    original_weight = snapshot["weight"].clone()
    with torch.no_grad():
        model.weight.add_(10.0)

    torch.testing.assert_close(snapshot["weight"], original_weight)
