from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    input_kind: str
    max_length: int
    pooling: str = "masked_mean"
    enabled: bool = True
    notes: str = ""


MODEL_ZOO: dict[str, ModelSpec] = {
    "canine-c": ModelSpec(
        name="canine-c",
        model_id="google/canine-c",
        input_kind="http_text",
        max_length=2048,
        notes="Character-level primary model; preserves encodings and punctuation.",
    ),
    "byt5-small": ModelSpec(
        name="byt5-small",
        model_id="google/byt5-small",
        input_kind="http_text",
        max_length=2048,
        notes="UTF-8 byte-level encoder-decoder used through its encoder.",
    ),
    "securebert2": ModelSpec(
        name="securebert2",
        model_id="cisco-ai/SecureBERT2.0-base",
        input_kind="http_text",
        max_length=2048,
        notes="Cybersecurity-domain text encoder; not specifically pretrained on HTTP payloads.",
    ),
    "secbert": ModelSpec(
        name="secbert",
        model_id="jackaduma/SecBERT",
        input_kind="http_text",
        max_length=512,
        notes="Legacy cybersecurity-domain comparison model.",
    ),
    "et-bert": ModelSpec(
        name="et-bert",
        model_id="linwhitehat/ET-BERT",
        input_kind="pcap_datagram",
        max_length=0,
        enabled=False,
        notes="Requires datagram/PCAP preprocessing; incompatible with SR-BH tabular HTTP input.",
    ),
    "netfound-small": ModelSpec(
        name="netfound-small",
        model_id="snlucsb/netFound-small",
        input_kind="pcap_flow_burst",
        max_length=0,
        enabled=False,
        notes="Current small checkpoint; requires PCAP/burst preprocessing and a custom adapter.",
    ),
}


def get_model_spec(name: str, require_enabled: bool = True) -> ModelSpec:
    try:
        spec = MODEL_ZOO[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_ZOO)}") from exc
    if require_enabled and not spec.enabled:
        raise ValueError(f"Model {name!r} is registered but unavailable: {spec.notes}")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the HTTP attack model zoo")
    parser.add_argument("command", choices=["list"], nargs="?", default="list")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.json:
        print(json.dumps([asdict(spec) for spec in MODEL_ZOO.values()], indent=2))
        return
    header = f"{'name':<16} {'enabled':<8} {'input':<18} model"
    print(header)
    print("-" * len(header))
    for spec in MODEL_ZOO.values():
        print(f"{spec.name:<16} {str(spec.enabled):<8} {spec.input_kind:<18} {spec.model_id}")
        print(f"  {spec.notes}")


if __name__ == "__main__":
    main()
