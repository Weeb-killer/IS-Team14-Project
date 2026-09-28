"""Tests for training only the classifier head.

Freezing has to hold two properties at once: the optimizer must not change the encoder, and the
encoder must stay deterministic. The second is easy to lose, because ``model.train()`` enables
dropout inside the backbone, and a backbone that resamples dropout every epoch is not frozen in
any useful sense even when its weights never move.
"""

import pytest

from http_attack_agent.models.hf_classifier import (
    build_classifier,
    freeze_backbone,
    parameter_counts,
    set_train_mode,
)


@pytest.fixture
def model():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    return build_classifier(
        "offline/tiny-bert",
        num_labels=3,
        dropout=0.1,
        backbone_config=transformers.BertConfig(
            vocab_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        ),
    )


def _head_and_backbone(model):
    return list(model.classifier.parameters()), list(model.backbone.parameters())


def test_everything_is_trainable_before_freezing(model):
    counts = parameter_counts(model)

    assert counts["frozen"] == 0
    assert counts["trainable"] == counts["total"]


def test_freezing_clears_gradients_on_the_backbone_only(model):
    freeze_backbone(model)
    head, backbone = _head_and_backbone(model)

    assert all(not parameter.requires_grad for parameter in backbone)
    assert all(parameter.requires_grad for parameter in head)


def test_counts_report_the_head_as_the_trainable_part(model):
    head, _ = _head_and_backbone(model)
    expected_head = sum(parameter.numel() for parameter in head)

    counts = freeze_backbone(model)

    assert counts["trainable"] == expected_head
    assert counts["frozen"] == counts["total"] - expected_head
    assert 0 < counts["trainable"] < counts["total"]


def test_a_training_step_moves_the_head_and_leaves_the_backbone_alone(model):
    """The property that matters: an optimizer step must not touch the encoder."""

    torch = pytest.importorskip("torch")

    freeze_backbone(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.1)

    before_backbone = [parameter.detach().clone() for parameter in model.backbone.parameters()]
    before_head = [parameter.detach().clone() for parameter in model.classifier.parameters()]

    set_train_mode(model, backbone_frozen=True)
    input_ids = torch.tensor([[2, 5, 6, 3], [2, 6, 5, 3]])
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids, attention_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.logits, torch.ones_like(output.logits)
    )
    loss.backward()
    optimizer.step()

    for before, after in zip(before_backbone, model.backbone.parameters()):
        torch.testing.assert_close(before, after.detach())
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(before_head, model.classifier.parameters())
    ), "the classifier head must still be learning"


def test_backbone_receives_no_gradients_at_all(model):
    torch = pytest.importorskip("torch")

    freeze_backbone(model)
    set_train_mode(model, backbone_frozen=True)
    input_ids = torch.tensor([[2, 5, 6, 3]])
    output = model(input_ids, torch.ones_like(input_ids))
    output.logits.sum().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_set_train_mode_keeps_a_frozen_backbone_in_eval_mode(model):
    freeze_backbone(model)

    set_train_mode(model, backbone_frozen=True)

    assert model.training, "the head must be in training mode"
    assert not model.backbone.training, "dropout inside a frozen encoder must stay off"


def test_set_train_mode_leaves_an_unfrozen_backbone_training(model):
    set_train_mode(model, backbone_frozen=False)

    assert model.training
    assert model.backbone.training


def test_frozen_backbone_returns_the_same_embedding_every_time(model):
    """Determinism is what makes caching the embeddings valid in a later change."""

    torch = pytest.importorskip("torch")

    freeze_backbone(model)
    set_train_mode(model, backbone_frozen=True)
    input_ids = torch.tensor([[2, 5, 6, 3]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        first = model.encode(input_ids, attention_mask)
        second = model.encode(input_ids, attention_mask)

    torch.testing.assert_close(first, second)


def test_an_unfrozen_backbone_in_training_mode_is_not_deterministic(model):
    """Shows what freezing prevents: dropout makes the same request encode differently."""

    torch = pytest.importorskip("torch")

    set_train_mode(model, backbone_frozen=False)
    input_ids = torch.tensor([[2, 5, 6, 3, 4, 5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        samples = [model.encode(input_ids, attention_mask) for _ in range(8)]

    assert any(
        not torch.equal(samples[0], other) for other in samples[1:]
    ), "backbone dropout should vary the embedding while training mode is on"
