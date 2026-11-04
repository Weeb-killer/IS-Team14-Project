from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..models.hf_classifier import build_embedding_head
from .common import checked_manifest, digest, read_json, write_json


def load_head(checkpoint, device="cpu"):
    import torch

    metadata = read_json(Path(checkpoint) / "metadata.json")
    head = build_embedding_head(
        metadata["hidden_size"], len(metadata["label_names"]), metadata.get("dropout", 0.1)
    )
    head.load_state_dict(
        torch.load(Path(checkpoint) / "embedding_head.pt", map_location="cpu", weights_only=True)
    )
    head.to(device).eval().requires_grad_(False)
    return head, metadata


def head_outputs(head, x, batch_size=512):
    import torch

    device = next(head.parameters()).device
    blocks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            block = torch.as_tensor(
                np.asarray(x[start : start + batch_size]).copy(), dtype=torch.float32, device=device
            )
            blocks.append(head(block).float().cpu().numpy())
    return np.concatenate(blocks)


def analyze_head(head, x, labels, label_names, bank, seed=14, random_directions=10, batch_size=512):
    """Evaluate only original head outputs; no substitute concept classifier is trained."""
    import torch

    if random_directions < 1:
        raise ValueError("At least one random direction is required.")
    device = next(head.parameters()).device
    head.eval().requires_grad_(False)
    names = bank["concept_names"].astype(str).tolist()
    valid = np.flatnonzero(bank["valid"])
    original = head_outputs(head, x, batch_size)
    rng = np.random.default_rng(seed)
    report = {name: {} for name in names}
    random_vectors = rng.normal(size=(random_directions, x.shape[1])).astype(np.float32)
    random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
    all_directions = bank["directions"][valid]
    if len(valid) == 0:
        return report
    vector_tensor = torch.as_tensor(
        all_directions.reshape(-1, x.shape[1]).T.copy(), dtype=torch.float32, device=device
    )
    for k, label in enumerate(tqdm(label_names, desc="Original-head TCAV")):
        selected = np.flatnonzero(labels[:, k] > 0)
        if not len(selected):
            for name in names:
                report[name][label] = {"status": "no_class_examples", "samples": 0}
            continue
        derivatives = []
        for start in range(0, len(selected), batch_size):
            rows = selected[start : start + batch_size]
            block = torch.tensor(
                np.asarray(x[rows]), dtype=torch.float32, device=device, requires_grad=True
            )
            gradient = torch.autograd.grad(head(block)[:, k].sum(), block)[0]
            derivatives.append((gradient @ vector_tensor).detach().cpu().numpy())
        derivatives = np.concatenate(derivatives).reshape(len(selected), len(valid), -1)
        for j, i in enumerate(valid):
            scores = (derivatives[:, j] > 0).mean(axis=0)
            report[names[i]][label] = {
                "status": "ok" if len(selected) >= 20 else "low_support",
                "samples": len(selected),
                "tcav_score": float(scores[0]),
                "repeat_tcav_scores": scores.tolist(),
                "mean_directional_derivative": float(derivatives[:, j, 0].mean()),
            }
    # Batched interventions bound RAM/GPU usage. Random perturbations match EACH row's norm.
    for i in tqdm(valid, desc="Concept-direction interventions"):
        direction = bank["directions"][i, 0]
        reference = bank["negative_reference"][i]
        delta = np.empty_like(original)
        null_delta = np.zeros_like(original)
        for start in range(0, len(x), batch_size):
            block = np.asarray(x[start : start + batch_size])
            alpha = (block - reference) @ direction
            erased = block - alpha[:, None] * direction
            base = original[start : start + len(block)]
            delta[start : start + len(block)] = base - head_outputs(head, erased, batch_size)
            for random_vector in random_vectors:
                perturbed = block - alpha[:, None] * random_vector
                null_delta[start : start + len(block)] += (
                    base - head_outputs(head, perturbed, batch_size)
                ) / random_directions
        for k, label in enumerate(label_names):
            chosen = labels[:, k] > 0
            if not chosen.any():
                continue
            values = delta[chosen, k]
            report[names[i]][label].update(
                {
                    "mean_logit_drop": float(values.mean()),
                    "median_logit_drop": float(np.median(values)),
                    "fraction_logit_reduced": float((values > 0).mean()),
                    "random_norm_matched_mean_logit_drop": float(null_delta[chosen, k].mean()),
                }
            )
    return report


def analyze(
    prepared,
    encoded,
    bank_dir,
    checkpoint,
    max_samples=5000,
    random_directions=10,
    seed=14,
    device=None,
):
    import torch

    start = time.perf_counter()
    manifest = checked_manifest(prepared)
    enc = read_json(Path(encoded) / "manifest.json")
    probe = read_json(Path(bank_dir) / "probe_report.json")
    if max_samples and max_samples < len(manifest["label_names"]):
        raise ValueError("--max-samples must be zero or at least the number of attack labels.")
    if not enc.get("complete") or enc["prepared_sha256"] != manifest["requests_sha256"]:
        raise ValueError("Encoded artifact is incomplete or belongs to different prepared data.")
    if probe["prepared_sha256"] != manifest["requests_sha256"]:
        raise ValueError("Probe bank belongs to a different prepared dataset.")
    if (
        digest(Path(checkpoint) / "embedding_head.pt") != probe["head_sha256"]
        or enc["checkpoint_sha256"] != probe["checkpoint_sha256"]
    ):
        raise ValueError("Classification head/checkpoint differs from the analyzed model.")
    x = np.load(Path(encoded) / "embeddings.npy", mmap_mode="r")
    labels = np.load(Path(prepared) / "labels.npy", mmap_mode="r")
    splits = np.load(Path(prepared) / "splits.npy", mmap_mode="r")
    hashes = np.load(Path(prepared) / "input_hashes.npy", mmap_mode="r")
    test = np.flatnonzero(splits == 2)
    test = test[~np.isin(hashes[test], hashes[splits != 2])]
    _, unique = np.unique(hashes[test], return_index=True)
    test = test[unique]
    rng = np.random.default_rng(seed)
    if max_samples and len(test) > max_samples:
        retained = set()
        per_label = min(100, max(1, max_samples // len(manifest["label_names"])))
        for k in range(labels.shape[1]):
            positives = test[labels[test, k] > 0]
            retained.update(
                rng.choice(positives, min(per_label, len(positives)), replace=False).tolist()
            )
        remaining = np.setdiff1d(test, list(retained))
        retained.update(rng.choice(remaining, max_samples - len(retained), replace=False).tolist())
        test = np.asarray(sorted(retained), dtype=int)
    if not len(test):
        raise ValueError("No content-disjoint held-out test requests remain.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head, metadata = load_head(checkpoint, device)
    subset = np.asarray(x[test])
    reference_logits = np.load(Path(encoded) / "logits.npy", mmap_mode="r")[test]
    if not np.allclose(head_outputs(head, subset), reference_logits, atol=2e-4, rtol=2e-4):
        raise ValueError("Saved head does not reproduce exported model logits.")
    with np.load(Path(bank_dir) / "concept_bank.npz", allow_pickle=False) as bank:
        results = analyze_head(
            head, subset, labels[test], metadata["label_names"], bank, seed, random_directions
        )
    report = {
        "samples": len(test),
        "seconds": time.perf_counter() - start,
        "sample_positions": test.tolist(),
        "seed": seed,
        "random_directions": random_directions,
        "concepts": results,
        "selection": "unique test inputs absent from original train/validation; "
        "capped sample retains examples from rare attack labels",
        "interpretation": "Original-head sensitivity, not causal attribution or "
        "a percentage contribution. Direction intervention may "
        "leave the real embedding distribution.",
    }
    write_json(Path(bank_dir) / "sensitivity_report.json", report)
    return report


def explain_request(
    prepared,
    annotations,
    encoded,
    bank_dir,
    checkpoint,
    row_id,
    output,
    ablate_field=None,
    device=None,
):
    import torch

    from ..data import serialize_http_request
    from ..models.checkpoint import load_local_checkpoint
    from .common import records

    manifest = checked_manifest(prepared)
    probe = read_json(Path(bank_dir) / "probe_report.json")
    enc = read_json(Path(encoded) / "manifest.json")
    waf = read_json(Path(annotations) / "manifest.json")
    if any(data["prepared_sha256"] != manifest["requests_sha256"] for data in (probe, enc, waf)):
        raise ValueError("Bank and prepared requests differ.")
    if not enc.get("complete") or not waf.get("complete"):
        raise ValueError("Encoding and annotation stages must be complete.")
    if waf["concept_names"] != list(probe["concepts"]):
        raise ValueError("Concept annotation names/order differ from the probe bank.")
    if digest(Path(annotations) / "concepts.npy") != probe["concepts_sha256"]:
        raise ValueError("Concept annotations differ from the fitted bank.")
    if enc["checkpoint_sha256"] != probe["checkpoint_sha256"]:
        raise ValueError("Encoded checkpoint differs from the fitted bank.")
    if digest(Path(checkpoint) / "embedding_head.pt") != probe["head_sha256"]:
        raise ValueError("Checkpoint head differs from probe bank provenance.")
    ids = np.load(Path(prepared) / "row_ids.npy", mmap_mode="r")
    found = np.flatnonzero(ids == row_id)
    if len(found) != 1:
        raise ValueError(f"Expected one prepared request with ID {row_id}.")
    i = int(found[0])
    row = next(r for r in records(Path(prepared) / "requests.jsonl") if r["position"] == i)
    x = np.load(Path(encoded) / "embeddings.npy", mmap_mode="r")[i : i + 1].copy()
    coverage = np.load(Path(encoded) / "field_coverage.npy", mmap_mode="r")[i]
    events = [e for e in records(Path(annotations) / "events.jsonl") if e["position"] == i]
    status = next(s for s in records(Path(annotations) / "status.jsonl") if s["position"] == i)
    observed = np.load(Path(annotations) / "concepts.npy", mmap_mode="r")[i]
    for event in events:
        field = event["position_info"]["field"]
        code = int(coverage[enc["field_names"].index(field)]) if field in enc["field_names"] else -1
        event["model_field_coverage"] = enc["coverage_codes"][str(code)]
        event["exact_matched_fragment_visibility"] = "unverified"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head, metadata = load_head(checkpoint, device)
    original = head_outputs(head, x)[0]
    probabilities = torch.sigmoid(torch.tensor(original)).numpy()
    concepts = []
    with np.load(Path(bank_dir) / "concept_bank.npz", allow_pickle=False) as bank:
        for j, name in enumerate(bank["concept_names"].astype(str)):
            if not bank["valid"][j]:
                continue
            direction = bank["directions"][j, 0]
            score = float(x[0] @ bank["weights"][j] + bank["intercept"][j])
            shift = float((x[0] - bank["negative_reference"][j]) @ direction)
            modified = x - shift * direction
            new_logits = head_outputs(head, modified)[0]
            tensor = torch.tensor(x, device=device, requires_grad=True)
            output_logits = head(tensor)
            sensitivities = []
            for k in range(len(metadata["label_names"])):
                gradient = torch.autograd.grad(output_logits[0, k], tensor, retain_graph=True)[0]
                sensitivities.append(float(gradient[0].detach().cpu().numpy() @ direction))
            concepts.append(
                {
                    "name": name,
                    "observed_crs_label": int(observed[j]),
                    "probe_decision_score": score,
                    "score_is_probability": False,
                    "probe_threshold": float(bank["thresholds"][j]),
                    "directional_derivatives": sensitivities,
                    "intervention_logit_drops": (original - new_logits).tolist(),
                }
            )
    thresholds = metadata.get("test_metrics", {}).get("thresholds", [0.5] * len(probabilities))
    if isinstance(thresholds, dict):
        thresholds = [thresholds[name] for name in metadata["label_names"]]
    result = {
        "row_id": row_id,
        "label_names": metadata["label_names"],
        "probabilities": probabilities.tolist(),
        "thresholds": thresholds,
        "predicted_labels": [
            name for name, p, t in zip(metadata["label_names"], probabilities, thresholds) if p >= t
        ],
        "waf_status": status,
        "waf_events": events,
        "concepts": concepts,
        "field_coverage": dict(zip(enc["field_names"], map(int, coverage))),
        "interpretation": "CRS evidence, probe scores, and model sensitivity are "
        "separate measurements; exact input-span attribution is unverified.",
    }
    if ablate_field:
        if ablate_field not in row["fields"]:
            raise ValueError(f"Unknown logical field: {ablate_field}")
        model, tokenizer, _ = load_local_checkpoint(checkpoint, device)
        model.requires_grad_(False)
        fields = {**row["fields"], ablate_field: ""}
        text = serialize_http_request(fields, {k: k for k in manifest["text_columns"]})
        tokens = tokenizer(
            text, truncation=True, max_length=metadata["max_length"], return_tensors="pt"
        )
        with torch.inference_mode():
            changed = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        result["field_ablation"] = {
            "field": ablate_field,
            "logit_drops": (original - changed.logits[0].cpu().numpy()).tolist(),
            "limitation": "Whole-field removal can change multiple concepts "
            "and bring later tokens into the visible window.",
        }
    write_json(output, result)
    return result
