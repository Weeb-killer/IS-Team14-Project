from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _torch_imports() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc
    return torch, nn


@dataclass
class ClassifierOutput:
    logits: Any
    embeddings: Any


def build_embedding_head(
    hidden_size: int, num_labels: int, dropout: float = 0.1
) -> Any:
    """Classifier head kept separate so TCAV can load it without the backbone."""

    _, nn = _torch_imports()
    bottleneck = max(64, int(hidden_size) // 2)
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, bottleneck),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(bottleneck, num_labels),
    )


def build_classifier(
    model_id: str,
    num_labels: int,
    dropout: float = 0.1,
    trust_remote_code: bool = False,
    backbone_config: Any | None = None,
) -> Any:
    """Build lazily so model-zoo inspection does not require torch/transformers."""

    _, nn = _torch_imports()
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc

    class HFRequestClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_id = model_id
            if backbone_config is None:
                self.backbone = AutoModel.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            else:
                # A saved local configuration can reconstruct the architecture
                # without downloading the original pretrained checkpoint again.
                self.backbone = AutoModel.from_config(
                    backbone_config, trust_remote_code=trust_remote_code
                )
            hidden_size = getattr(self.backbone.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.backbone.config, "d_model", None)
            if hidden_size is None:
                raise ValueError(f"Cannot infer hidden size for {model_id}")
            self.hidden_size = int(hidden_size)
            # A nonlinear head makes per-example TCAV sensitivity meaningful.
            self.classifier = build_embedding_head(self.hidden_size, num_labels, dropout)

        @staticmethod
        def _masked_mean(hidden: Any, attention_mask: Any) -> Any:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            total = (hidden * mask).sum(dim=1)
            denominator = mask.sum(dim=1).clamp(min=1.0)
            return total / denominator

        def encode(self, input_ids: Any, attention_mask: Any) -> Any:
            if getattr(self.backbone.config, "is_encoder_decoder", False):
                outputs = self.backbone.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            return self._masked_mean(outputs.last_hidden_state, attention_mask)

        def logits_from_embeddings(self, embeddings: Any) -> Any:
            return self.classifier(embeddings)

        def forward(self, input_ids: Any, attention_mask: Any) -> ClassifierOutput:
            embeddings = self.encode(input_ids, attention_mask)
            logits = self.logits_from_embeddings(embeddings)
            return ClassifierOutput(logits=logits, embeddings=embeddings)

    return HFRequestClassifier()


def load_tokenizer(model_id: str, local_files_only: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError('Install neural dependencies with: pip install -e ".[neural]"') from exc
    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
