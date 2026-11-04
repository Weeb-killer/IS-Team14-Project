# CRS-versus-SecBERT independent baseline

This mandatory baseline measures whether the trained classifier adds detection value over
direct OWASP CRS decisions. Probe decoding scores do not answer that question. No superiority
is assumed: the report can show gains, losses, tradeoffs, or inconclusive differences.

This extension stays inside `concept_analysis/`. Original training and root configuration files
remain unchanged. A pre-extension snapshot is recorded under
`runs/concept-comparison-change-backup/` on the development workspace, so this extension can also
be reverted independently of the earlier concept-analysis addition.

## Quick demo

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m http_attack_agent.concept_analysis.comparison demo \
  --output runs/crs-secbert-comparison-demo
```

Open `runs/crs-secbert-comparison-demo/comparison/README.md`. The fixture demonstrates unknown
inspections, duplicate inputs, unsupported labels, independent detector disagreements, calibration,
confidence intervals, tables, and plots. It contains **no real CRS or SecBERT benchmark results**.

## Real comparison using saved SecBERT results

Use the prepared requests and encoded checkpoint outputs from the parent workflow. If those
do not exist yet, run:

```bash
python -m http_attack_agent.concept_analysis prepare \
  --dataset-config configs/dataset.srbh2020.yaml \
  --output runs/secbert-concepts/prepared

python -m http_attack_agent.concept_analysis encode \
  --prepared runs/secbert-concepts/prepared \
  --checkpoint runs/secbert-frozen --device cuda --batch-size 32 \
  --output runs/secbert-concepts/encoded
```

The checkpoint must be the original trained classifier, with saved validation-selected thresholds.
The comparison reuses its logits. It does not train a new classifier or use concept scores as
attack predictions. Checkpoint metadata, logits, rule events, score arrays, and prepared inputs
are verified by their recorded hashes.

Replay with the updated connector, which now records the actual final anomaly score and threshold:

```bash
python -m http_attack_agent.concept_analysis check-waf --crs runs/crs-4.10.0

python -m http_attack_agent.concept_analysis replay \
  --prepared runs/secbert-concepts/prepared \
  --crs runs/crs-4.10.0 --workers 4 \
  --output runs/secbert-concepts/waf-comparison

python -m http_attack_agent.concept_analysis.comparison run \
  --prepared runs/secbert-concepts/prepared \
  --annotations runs/secbert-concepts/waf-comparison \
  --encoded runs/secbert-concepts/encoded \
  --checkpoint runs/secbert-frozen \
  --output runs/secbert-concepts/comparison \
  --target-fpr 0.01 --bootstrap 500
```

**Older replay outputs lack score telemetry and must be regenerated.** They cannot be converted
by adding up severity values: CRS scoring depends on actual rule actions, chains, transformations,
and configured paranoia levels. The comparison fails explicitly when required score provenance
is missing. Existing completed embedding exports from the parent workflow can be reused.

For Docker, rebuild the image after updating source so the new telemetry is included:

```bash
docker build -f src/http_attack_agent/concept_analysis/Dockerfile -t http-concept-waf .
docker run --rm http-concept-waf check-waf --crs /opt/crs
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/runs/secbert-concepts:/work" http-concept-waf replay \
  --prepared /work/prepared --output /work/waf-comparison --crs /opt/crs --workers 4
```

Run comparison scoring on the host afterward; it requires no native WAF library, model forward
pass, or GPU when the saved outputs already exist. It generally adds minutes of CPU work and
event-log reading rather than another encoder pass. Bootstrap and large logs can increase time;
`--bootstrap 100` is useful for a pilot, while `--bootstrap 0` disables intervals.

## Primary comparison: any attack versus normal

Both detectors are evaluated against `any(original_attack_label == 1)` on the same test IDs:

| Detector | Default decision | Continuous ranking score |
| --- | --- | --- |
| CRS | Actual final `blocking_inbound_anomaly_score >= inbound_anomaly_score_threshold` | Actual aggregate anomaly score |
| SecBERT | At least one probability exceeds its saved class threshold | Maximum logit margin over the saved class thresholds |

CRS runs in DetectionOnly mode so later rules remain observable. Its recorded threshold decision
means **would block under this request policy**. Early blocking is disabled. The connector adds
a log-only phase-2 telemetry rule after CRS request processing; it does not change anomaly scores.
The usual release default inbound threshold is 5, but the evaluator reads the actual threshold
per request rather than assuming 5. An arbitrary rule hit is not automatically an attack verdict.

The report supplies Precision, Recall, F1, FPR, FNR, Accuracy, ROC AUC, AP, and TP/FP/FN/TN.
It also evaluates two alternative policies for each detector using **only validation data**:

- `validation_f1`: select the threshold maximizing validation binary F1.
- `validation_fpr`: select a threshold meeting the configured validation normal FPR cap.

Calibration excludes exact training inputs and repeated validation inputs. Threshold ties are
handled conservatively. The target validation FPR is not guaranteed on the test distribution.
The saved SecBERT class thresholds are not retuned against test labels.

## Attack-category projection

The secondary task maps recorded CRS rule metadata to the dataset taxonomy. Default mapping
uses the **terminal CAPEC ID only**, never every ancestor in a CAPEC path. A SQLi rule tagged
`capec/1000/152/248/66` maps to SQL Injection (66), not additionally to Command Injection (248).
Definitions are fixed before test evaluation. Native CRS does not expose the same 13-way task.

Rules are inventoried from the loaded release's request files at the selected paranoia level.
This is declared mapping coverage; rule conditions and flow control still govern each request.
Unmapped labels remain visible as unsupported. The report includes:

- Full-task category metrics, with zero CRS category predictions for unmapped labels.
- A separate projection onto the labels with declared CRS mappings.
- Per-label precision, recall, F1, confusion counts, support, and mapping counts.

A category is predicted for CRS only when a mapped event occurred **and** the recorded blocking
threshold was reached. The report describes this as a taxonomy projection, not native CRS
classification. Binary detection remains independent of this mapping. A higher full-task Macro
F1 caused only by unsupported CRS labels is insufficient evidence of better common-category
detection. Check primary detection, mapped-label results, support, and actual disagreements.

Override mapping only when justified by a documented experiment definition:

```yaml
labels:
  sql_injection:
    capec_leaf: 66
  path_traversal:
    capec_leaf: 126
  # Other labels omitted here have no mapping under this custom definition.
```

Pass the file through `--mapping path/to/mapping.yaml`. `rule_ids` and `tags` are optional explicit
alternative matchers. The effective mapping is retained in `comparison.json`; do not choose a
mapping after inspecting which one improves test scores.

## Fairness, failures, and uncertainty

`common_test` retains every held-out request with valid CRS score telemetry. `clean_common_test`
also removes exact model inputs present in prepared training/validation and duplicate test inputs.
Both detectors always use the identical subset in a paired table. Sampled preparation cannot
audit duplicates in unselected original rows, and exact deduplication does not remove near duplicates
or retroactively undo historical model exposure.

Unknown CRS inspections are not normal predictions. Their counts, original attack/normal supports,
and failure-policy scenarios are reported separately. SecBERT's all-test metrics are retained too.
Check coverage before interpreting any advantage; failures concentrated on difficult requests
can distort complete-case results.

Paired 95% percentile intervals resample exact-input groups for F1, recall, precision, FPR, and
accuracy differences. Positive F1/recall and negative FPR differences favor SecBERT. These intervals
condition on the supplied checkpoint and sample. They do not establish broad deployment superiority,
account for every source of model-training uncertainty, or constitute corrected multiple testing.

The CRS and model start from the same available CSV fields, but CRS evaluates the reconstructed
request while SecBERT sees its saved token-truncated serialization. This is an operational comparison
of the two configured systems, not equal text coverage or a raw-wire WAF benchmark. Original duplicate
headers, framing, missing fields, and reconstructed Content-Length limit protocol-attack conclusions.

Recorded stage throughput is included with CPU-worker and GPU-batch settings. Replay logging and
embedding export add different overheads; those values are not matched per-request online latency
and must not be used alone to advertise inherent speed or compute-efficiency superiority.

## Outputs

| File | Contents |
| --- | --- |
| `README.md` | English comparison tables, charts, operating points, and limitations |
| `comparison.json` | Complete metrics, mappings, calibration, integrity metadata, and intervals |
| `binary_metrics.csv` | Every binary subset/policy/detector result, including confusion counts |
| `per_label_metrics.csv` | Full-task and mapped-label category results |
| `predictions.csv` | Every original test ID, ground truth, both decisions, scores, and unknown status |
| `disagreement_examples.json` | Seeded examples from each win/loss disagreement type with matched rule IDs |
| `figures/` | Primary detection and per-category comparison plots |

There is no automatic success verdict that assumes SecBERT must win. The deployment decision
should be based on measured error tradeoffs, coverage, and acceptable operational cost.

## Tests

```bash
python -m pytest src/http_attack_agent/concept_analysis/tests -q
```

The synthetic comparison runs without native ModSecurity or pretrained-model downloads. The real
WAF integration test remains conditional on `CONCEPT_ANALYSIS_CRS`; check its result on Linux.

## Primary references

- [CRS 4.10.0 aggregate scoring and blocking evaluation](https://github.com/coreruleset/coreruleset/blob/v4.10.0/rules/REQUEST-949-BLOCKING-EVALUATION.conf)
- [CRS 4.10.0 initialized scoring variables and defaults](https://github.com/coreruleset/coreruleset/blob/v4.10.0/rules/REQUEST-901-INITIALIZATION.conf)
