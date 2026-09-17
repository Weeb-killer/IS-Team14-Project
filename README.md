# HTTP Attack Agent Research Framework

Research framework for multi-label HTTP attack classification, model comparison, and embedding interpretability using the SR-BH 2020 dataset.

The project deliberately separates three concerns:

1. **Attack classification**: the model predicts one or more attack categories. A request is considered normal when every attack probability remains below its calibrated threshold.
2. **Embedding interpretation**: concept probes, prototypes, TCAV, and concept erasure test whether security concepts are encoded in and used from the learned embeddings.
3. **WAF evidence**: real OWASP CRS matches are stored as external concept annotations. A WAF match is not treated as a model explanation or as a replacement for the dataset labels.

## Built-in model zoo

| Name | Hugging Face ID | Required input | Status |
|---|---|---|---|
| `canine-c` | `google/canine-c` | Character-level HTTP request | Recommended primary model |
| `byt5-small` | `google/byt5-small` | UTF-8 bytes | Recommended byte-level comparison |
| `securebert2` | `cisco-ai/SecureBERT2.0-base` | Tokenized HTTP text | Cybersecurity-domain comparison |
| `secbert` | `jackaduma/SecBERT` | Tokenized HTTP text | Legacy cybersecurity comparison |
| `et-bert` | ET-BERT checkpoint | PCAP/datagram | Registered only; incompatible with tabular HTTP input |
| `netfound-small` | `snlucsb/netFound-small` | PCAP/flow bursts | Registered only; incompatible with tabular HTTP input |

`et-bert` and `netfound-small` are intentionally disabled for the SR-BH table. They require dedicated PCAP preprocessing and model adapters before they can be enabled.

## Installation

Model weights are downloaded by the selected model adapter when needed. SR-BH 2020 can be read locally or downloaded from the URL configured in YAML. Create a Python 3.10 or newer environment, then install every runtime, explainability, Parquet, and development dependency with:

```bash
pip install -r requirements.txt
```

To install or refresh the current checkout explicitly in editable mode (including its CLI entry points), run this from the repository root on Windows:

```powershell
py -m pip install -e .
```

`requirements.txt` already includes `-e .`, so you do not need to run this command again immediately after installing from that file. Use it if you installed dependencies separately or need to refresh the local installation.

Alternatively, install directly from the package extras:

```bash
pip install -e ".[neural,explain,dev]" pyarrow sentencepiece
```

Inspect the model zoo:

```bash
http-attack-models list
```

## Fast smoke demo

After installation, run the offline end-to-end demo before using the full dataset:

```bash
http-attack-demo
```

The demo creates deterministic synthetic HTTP requests under `runs/smoke-demo/` and verifies:

- local CSV discovery and YAML parsing;
- remote ZIP acquisition, extraction, SHA-256 verification, and cache reuse;
- HTTP serialization, multi-label targets, and deterministic train/validation/test splits;
- lightweight TF-IDF/SVD embeddings and multi-label classification;
- embedding probes, prototypes, TCAV, and concept erasure;
- OWASP CRS alert parsing and WAF concept aggregation;
- model-zoo registration and imports for the training/explanation command modules.

A successful run writes `runs/smoke-demo/smoke_report.json` and exits with status code 0. Any failed stage prints its error and exits with status code 1, which also makes this command suitable for CI health checks.

The default run checks the data, modeling, neural, and visualization dependencies used by the runtime paths. For a faster core-only check that skips PyTorch explanations and neural/visualization dependencies, use:

```bash
http-attack-demo --core-only
```

This smoke demo intentionally does not download large pretrained checkpoints, train on the complete SR-BH 2020 table, or start ModSecurity/Coraza. Those operations additionally depend on network access, compute capacity, and an external WAF installation.

## Data preparation

Use `configs/dataset.srbh2020.yaml` with the official SR-BH 2020 `data_capec_multilabel.csv` file. The same configuration contains two source blocks. Select one by changing only `source.mode`:

- `local`: first tries `source.local.path`, then falls back to `source.local.glob`. The default glob accepts a uniquely named CSV anywhere below `data/`.
- `remote`: downloads `source.remote.url`, extracts `source.remote.member` when the response is a ZIP, and caches the final CSV at `source.remote.path`. A configured SHA-256 checksum is verified both after download and when a cached file is reused.

For local use, place the CSV at `data/data_capec_multilabel.csv`, or keep its existing filename anywhere under `data/`. Dataset files in `data/` are ignored by Git; only `data/README.md` is tracked. If the fallback glob finds several CSV files, set `source.local.path` to the exact one to avoid an ambiguous selection.

To resolve the selected source before training, run:

```bash
http-attack-data --dataset-config configs/dataset.srbh2020.yaml
```

Training calls the same resolver automatically, so this preparation command is optional. Later remote runs reuse the cached file and do not download it again. For other table schemas, copy and edit `configs/dataset.example.yaml`. CSV is supported directly, but converting the dataset to Parquet is recommended for repeated experiments.

Important constraints:

- Put only the reviewed CAPEC/attack labels in `label_columns`.
- Populate `waf_concept_columns` only with annotations produced by a separate OWASP CRS replay.
- Never use CRS rule IDs, tags, matches, or anomaly scores as model input. Doing so creates target leakage and circular explanations.
- Use a temporal or collection-batch split for formal experiments. The random split fallback is intended only for development and smoke tests.

A normalized table should contain fields similar to:

```text
request_id, method, path, query, headers, body,
label_sqli, label_xss, label_cmdi, ...,
waf_attack_sqli, waf_attack_xss, ...
```

## Training and model comparison

From PowerShell in the repository root, install the dependencies and verify that the selected dataset is available:

```powershell
py -m pip install -r requirements.txt
py -m http_attack_agent.data --dataset-config configs/dataset.srbh2020.yaml
py -m http_attack_agent.models.zoo list
```

The default dataset source is `local`. Put the SR-BH CSV under `data/`, or change `source.mode` to `remote` in the YAML to download it. The first pretrained-model run also needs access to its Hugging Face checkpoint.

Start with a small **training smoke run** (this still reads the full CSV before sampling rows):

```powershell
py -m http_attack_agent.training --dataset-config configs/dataset.srbh2020.yaml --model canine-c --output runs/canine-c-smoke --epochs 1 --batch-size 2 --max-length 512 --max-rows 2000
```

For a formal run, omit `--max-rows` and choose epochs and batch size to fit your GPU. For example:

```bash
http-attack-train \
  --dataset-config configs/dataset.srbh2020.yaml \
  --model canine-c \
  --output runs/canine-c \
  --epochs 3 --batch-size 4 --max-length 512
```

Training displays a batch-level progress bar for every epoch with elapsed time, ETA, and running loss. Validation, final test evaluation, and test-embedding export also show progress. JSON metrics are printed after each epoch.

To compare the four enabled models on the same deterministic data split, run:

```bash
http-attack-train \
  --dataset-config configs/dataset.srbh2020.yaml \
  --model canine-c byt5-small securebert2 secbert \
  --output runs/model-zoo \
  --epochs 3 --batch-size 4 --max-length 256
```

Each model writes its best-validation-epoch weights locally. A single-model run writes directly to the `--output` directory; a multi-model run writes one subdirectory per model (for example, `runs/model-zoo/canine-c/`). Each model directory contains:

```text
model.pt                 Full fine-tuned model weights (backbone + classifier head)
backbone_config/          Architecture configuration for offline reconstruction
tokenizer/                Local tokenizer files
embedding_head.pt         Classifier-head weights used by TCAV
metadata.json             Labels, inference length, thresholds, history, and metrics
test_embeddings.npz       Embeddings and annotations for the test requests
```

`runs/` is ignored by Git. The saved checkpoint is self-contained for inference and does not need to download the original pretrained weights again. Load it with:

```python
from http_attack_agent.models.checkpoint import load_local_checkpoint

model, tokenizer, metadata = load_local_checkpoint("runs/canine-c")
print(metadata["label_names"])
```

This restores the best saved weights in evaluation mode on CPU by default. Pass `device="cuda"` to load onto an available GPU. `model.pt` is an inference checkpoint, not a resumable training checkpoint: optimizer state is not saved.

## Embedding interpretability

```bash
http-attack-explain \
  --embeddings runs/model-zoo/canine-c/test_embeddings.npz \
  --output runs/model-zoo/canine-c/explanations
```

The report includes:

- **Linear concept probes**: test whether attack or WAF concepts can be decoded from the embeddings.
- **Permutation baselines**: compare each probe with randomly shuffled concept labels so chance structure is not mistaken for semantics.
- **Prototype separation**: measure whether requests sharing a concept occupy a coherent region in cosine space.
- **TCAV**: test whether moving along a learned concept direction systematically increases an attack logit.
- **Concept erasure**: remove a concept direction from an embedding and measure the resulting logit change.
- **Nearest prototypes**: return representative request IDs for case-level analysis without copying request payloads into the report.

Add `--conditional-probes` to test whether the model has learned more than the coarse attack label. Conditional probes test a WAF sub-concept within a single attack class, for example by distinguishing libinjection matches among requests already labeled as SQL injection.

A strong linear probe demonstrates that information is present in the embedding. TCAV and concept erasure further test whether the prediction head uses that direction. None of these measurements alone proves causality, so reports also include random baselines, sample counts, and significance indicators.

## OWASP CRS interface

OWASP CRS is a ruleset and must be executed by an engine such as Coraza or ModSecurity. Replay requests in `DetectionOnly` mode and retain at least:

```text
request_id, transaction_id, rule_id, message, tags,
matched_variable, matched_data, severity, crs_version
```

Convert common ModSecurity or Coraza single-line alerts to JSONL:

```bash
http-attack-parse-waf --input modsec_error.log --output waf_events.jsonl
```

The offline replay process should attach the dataset `request_id` to each WAF event. Aggregate the parsed events into a request-level concept table with:

```bash
http-attack-waf-concepts --events waf_events.jsonl --output waf_concepts.csv
```

Join this table back to the dataset on `request_id`, then add the resulting concept columns to `waf_concept_columns`. If a log does not contain a request ID, the aggregator falls back to the WAF transaction ID. That fallback is safe only when a separate transaction-to-dataset-row mapping has been retained.

Retain the actual attack rules and tags, such as `attack-sqli` and `attack-xss`. Do not keep only summary rules such as `anomaly score exceeded`, because they do not identify the underlying attack evidence.

## SR-BH 2020 data-quality warning

SR-BH multi-label annotations originated from ModSecurity/CRS and were subsequently reviewed manually and semi-automatically. They are therefore not a rule-independent, fully human ground truth.

Keep the following sources separate throughout every formal experiment:

1. Original SR-BH CAPEC labels.
2. Matches produced by replaying the request through the selected current CRS version.
3. Model predictions and embedding-derived explanations.

Never overwrite CAPEC labels with current CRS matches, and never expose CRS concept columns to the classifier during training.

## Project structure

```text
configs/                         Dataset and concept configuration
src/http_attack_agent/data.py   SR-BH table loading and HTTP serialization
src/http_attack_agent/models/   Unified model interface and registry
src/http_attack_agent/explain/  Probes, prototypes, TCAV, and concept erasure
src/http_attack_agent/waf/      CRS event parsing and concept aggregation
tests/                           Unit tests that do not download model weights
```
