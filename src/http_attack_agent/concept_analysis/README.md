# Independent SecBERT concept analysis

This directory implements **PCBM-style concept probing**, repeated CAVs, original-head TCAV,
concept-direction interventions, and per-request evidence. The existing classifier continues to
make predictions. A complete PCBM concept-to-attack classifier is outside this workflow.

All implementation, configuration, tests, and documentation added for this feature live here.
Existing training, dataset configuration, model checkpoints, and root entry points are unchanged.
Commands below run from the repository root in a Linux terminal.

## Required direct CRS baseline

Before claiming value over a rules-only detector, run the independent
[CRS-versus-SecBERT comparison](comparison/README.md). It evaluates the original attack labels,
actual CRS anomaly-threshold decisions, validation-only operating points, unknown inspections,
per-category mapping coverage, and paired error differences. It requires no concept-probe training.
Probe decoding quality alone cannot establish a classification advantage over CRS.

## Installation and a small demo

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m http_attack_agent.concept_analysis --help
python -m http_attack_agent.concept_analysis demo --output runs/concept-analysis-demo
python -m pytest src/http_attack_agent/concept_analysis/tests -q
```

The demo uses synthetic embeddings, synthetic concept annotations, and a small classifier.
It checks probe training, held-out evaluation, TCAV, interventions, a single-request explanation,
and English Markdown/CSV/PNG report generation. It downloads no models and does **not** validate
real SecBERT or real CRS execution. Its report is `runs/concept-analysis-demo/bank/README.md`.

The Python dependencies already exist in the root requirements file. Real WAF replay additionally
requires native **libModSecurity 3** and a local **OWASP CRS release**. Those are not pip packages.
Install the matching CUDA-enabled PyTorch build in your A100 environment, then verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Real WAF setup

Option A: on Debian/Ubuntu, install `libmodsecurity3` using your normal system package manager,
then download a fixed CRS release into a local directory. The example below uses an explicit
research version; choose and record the release required by your experiment.

```bash
git clone --depth 1 --branch v4.10.0 https://github.com/coreruleset/coreruleset.git runs/crs-4.10.0
python -m http_attack_agent.concept_analysis check-waf --crs runs/crs-4.10.0
```

`check-waf` must pass before a full replay. It checks actual SQLi detection, the logging callback,
both request-phase completion sentinels, and final anomaly-score telemetry. To run the native integration test:

```bash
CONCEPT_ANALYSIS_CRS=runs/crs-4.10.0 python -m pytest src/http_attack_agent/concept_analysis/tests -q
```

Option B: build the included CPU-only WAF image. No application backend or listening port is used.
Only project source is copied into the image; the dataset is mounted when running it.

```bash
docker build -f src/http_attack_agent/concept_analysis/Dockerfile -t http-concept-waf .
docker run --rm http-concept-waf check-waf --crs /opt/crs
```

The image build also runs `check-waf`, so incompatible native libraries or CRS configuration
fail the build before dataset replay. The connector reads the PCRE-limit flag through
`TX:MSC_PCRE_LIMITS_EXCEEDED`, which also works with older libModSecurity 3 releases.

For rootless Podman in a Slurm job without a systemd user session, use the following instead
of the Docker build/check/replay commands. Run from the repository root after activating your
Python environment. Request at least four CPU cores; no GPU or sudo is required for these
commands when the cluster already supports rootless Podman.

```bash
set -e
RUN_DIR="${RUN_DIR:-$PWD/runs/secbert-concepts}"

podman --cgroup-manager=cgroupfs --events-backend=file build \
  --isolation=chroot \
  -f src/http_attack_agent/concept_analysis/Dockerfile \
  -t localhost/http-concept-waf:latest .

podman --cgroup-manager=cgroupfs --events-backend=file run \
  --rm --cgroups=disabled --network=none --pull=never \
  localhost/http-concept-waf:latest check-waf --crs /opt/crs

podman --cgroup-manager=cgroupfs --events-backend=file run \
  --rm --cgroups=disabled --network=none --pull=never --userns=keep-id \
  -v "$RUN_DIR:/work" \
  localhost/http-concept-waf:latest replay \
  --prepared /work/prepared --output /work/waf --crs /opt/crs --workers 4
```

`prepared/` must already contain a completed preparation stage. Reuse it after fixing a build
or preflight failure; do not repeat preparation. Keep the build and replay in the same job
initially because image storage can be node-local. `--pull=never` prevents an unsuccessful
build from being followed by an unrelated registry pull. Disabling container cgroup creation
does not remove Slurm's existing resource limits.

Native replay uses the ModSecurity C API directly. It executes request phases 1 and 2 in
DetectionOnly mode and records engine identity, the effective configuration, the complete CRS
tree hash, concept mapping, processing status, and elapsed time. Default paranoia level is 1.
Response rules are outside the request-only annotation scope.

Some native libinjection or `multiMatch` alerts omit rule tags
([upstream report](https://github.com/owasp-modsecurity/ModSecurity/issues/2754)). For each logged rule ID,
the connector supplements missing tags from the exact loaded CRS tree's rule catalog. This
keeps broad concepts, event-level joint concepts, and CAPEC mappings available on these library
versions. It does not create additional rule hits or change the recorded anomaly score. Enriched
events preserve `raw_line`, record the original `logged_tags`, list `tags_from_rule_metadata`,
and identify `rule_metadata_file`; the replay manifest records the CRS tree hash. Unknown rule
IDs receive no inferred tags. The SQLi/phase/score preflight remains mandatory.

## Full workflow using your existing SecBERT checkpoint

The checkpoint directory must contain `model.pt`, `embedding_head.pt`, `metadata.json`,
`backbone_config/`, and `tokenizer/`. Replace `runs/secbert-frozen` if your actual location differs.
No SecBERT retraining is performed. All new stages write into `runs/secbert-concepts/`.

```bash
python -m http_attack_agent.concept_analysis prepare \
  --dataset-config configs/dataset.srbh2020.yaml \
  --output runs/secbert-concepts/prepared

python -m http_attack_agent.concept_analysis replay \
  --prepared runs/secbert-concepts/prepared \
  --crs runs/crs-4.10.0 --workers 4 \
  --output runs/secbert-concepts/waf

python -m http_attack_agent.concept_analysis encode \
  --prepared runs/secbert-concepts/prepared \
  --checkpoint runs/secbert-frozen --batch-size 32 --device cuda \
  --output runs/secbert-concepts/encoded

python -m http_attack_agent.concept_analysis fit \
  --prepared runs/secbert-concepts/prepared \
  --annotations runs/secbert-concepts/waf \
  --encoded runs/secbert-concepts/encoded \
  --cap 20000 --permutations 20 --repeats 3 --conditional \
  --output runs/secbert-concepts/bank

python -m http_attack_agent.concept_analysis analyze \
  --prepared runs/secbert-concepts/prepared \
  --encoded runs/secbert-concepts/encoded \
  --bank-dir runs/secbert-concepts/bank \
  --checkpoint runs/secbert-frozen --device cuda \
  --max-samples 5000 --random-directions 10

python -m http_attack_agent.concept_analysis report \
  --bank-dir runs/secbert-concepts/bank
```

For the container option, replace only the replay command with:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/runs/secbert-concepts:/work" http-concept-waf replay \
  --prepared /work/prepared --output /work/waf --crs /opt/crs --workers 4
```

WAF and embedding export use the same prepared manifest but run independently. Completed stages
can be reused. A new stage output directory must be empty: incomplete stages are never silently
accepted or overwritten. An interrupted stage should be rerun into a fresh directory; mid-stage
resume is not implemented. Keep experiment outputs under `runs/`, which is already ignored by Git.

Preparation reproduces the current YAML's original train/validation/test membership before any
optional sampling. Encoding checks checkpoint input-field order, label order, dataset size,
and historical held-out request IDs when `test_embeddings.npz` is available. For old checkpoints
without a saved data hash/split manifest, these checks cannot establish full historical provenance.
Use the original CSV, row order, YAML split settings, and code revision.

`--conditional` additionally tests broad WAF concepts within each attack label. It can substantially
increase CPU time and may skip class/concept combinations with inadequate unique examples.
Conditional subprobes use one fit without permutation repetitions; the repetition options apply
to the main concept probes.
Use `--permutations 100 --repeats 5` for a more extensive stability experiment. Tail fractions are
empirical diagnostics; the report does not claim corrected multiple-testing significance.

## Request, rule, concept, and location alignment

Each CSV row retains its ID and exact model-input hash. The offline connector creates one
transaction for that row, collects its rule events, and attaches the original ID directly.
No tracking header is inserted into the model/WAF input.

| Evidence | Automatic annotation |
| --- | --- |
| Rule ID | Exact recorded CRS rule match |
| `attack-sqli` rule tag | SQLi detection concept |
| Rule `942100` | libinjection SQLi detection concept, not necessarily a UNION query |
| Rule `942140` | SQL database-name detection concept |
| Rule `942160` | SQL sleep/benchmark test detection concept |
| `ARGS:user` | Parameter named `user`; source traced to query/body when unambiguous |
| `REQUEST_HEADERS:User-Agent` | User-Agent field |
| `REQUEST_COOKIES:session` | Cookie member named `session` |
| `REQUEST_BODY` | Body collection |

Joint concepts are formed from the **same event**, so SQLi in arguments and XSS in a cookie do
not imply SQLi in that cookie. Summary/threshold rules are excluded. All annotations are automatic
after the once-defined mapping in `concepts.yaml`; `--concept-config` accepts a custom YAML mapping.
The default Python mapping also works when installed from a wheel.

Completed inspections use 0/1 concept labels. Missing phases, over-limit bodies, processing errors,
and unrepresentable request lines use **-1 (unknown)**, excluded from fitting and evaluation.
Zero means no observed match under this engine, enabled rules, and configuration, not semantic safety.

The CSV cannot reconstruct original duplicate headers, framing, omitted fields, or original bytes
that were already lost. Body Content-Length is synthesized and recorded. Connection addresses are
fixed synthetic values. Protocol-related labels therefore describe the reconstructed request.
The CAPEC attack targets remain separate and are never copied into WAF labels.

## Embedding analysis and interpretation limits

Embedding extraction uses the checkpoint's tokenizer, max length, masked-mean pooling, and weights.
Embeddings and logits are float32 memory-mapped arrays. The encoder and head are held fixed.
For the supplied frozen-backbone run, probe results concern pre-existing SecBERT representations
and the trained head, not security knowledge newly acquired by the encoder during head training.

Each concept uses a linear SVM. Its scaler is fitted on training data only, C and the decision
threshold are selected on validation data, and metrics are computed on held-out test data.
The saved affine separator and CAV are converted back into the original embedding coordinates.
SVM decision scores are not calibrated probabilities. Probe training is capped and balanced per
concept; the original test prevalence is retained.

Identical model inputs are deduplicated for probe fitting, and overlapping validation inputs are
excluded. The report includes both full test results and distinct test inputs absent from all
prepared training/validation inputs. It does not detect near duplicates or undo historical exposure.
A pilot samples the original splits, so its duplicate audit cannot cover unselected original rows.

TCAV uses the original head's logit gradients. A missing class is N/A, never replaced with all rows.
Interventions move a concept projection to its training-negative reference level, with norm-matched
random-direction controls. They do not erase every possible encoding of the concept, disentangle
correlated concepts, or establish causality. Capped TCAV sampling retains rare attack labels and is
explicitly identified as a sample rather than a population metric.

Fast tokenizers additionally save per-field visibility: unknown, absent/truncated, partly visible,
or fully visible. A matched field's visibility is not an exact matched-fragment attribution.
Exact character spans and transformation inversion are deliberately not fabricated.

## One request

Choose an ID from the prepared `row_ids.npy` or `requests.jsonl`:

```bash
python -m http_attack_agent.concept_analysis explain \
  --prepared runs/secbert-concepts/prepared \
  --annotations runs/secbert-concepts/waf \
  --encoded runs/secbert-concepts/encoded \
  --bank-dir runs/secbert-concepts/bank \
  --checkpoint runs/secbert-frozen --row-id row-123 \
  --output runs/secbert-concepts/request-row-123.json --device cuda
```

The JSON includes original probabilities and thresholds, actual rule evidence, observed concept
labels, probe scores, directional derivatives, intervention logit changes, and field visibility.
Add `--ablate-field body` to re-encode that request with an empty body and compare original logits.
This is a whole-field perturbation, which may change multiple concepts and the truncation window;
it is not an automatically isolated semantic edit.

## A100 / 32 GB host RAM planning estimate

The following is an **engineering estimate, not an A100 benchmark**. It assumes 907,815 requests,
one SecBERT pass at length 256, batch size 32, 4-8 available CPU cores, local SSD storage, 22
configured concepts, a 20,000-example cap per probe, 20 random-label runs, 3 CAV fits, and 5,000
requests for head analysis. CPU allocation, logging volume, rule hits, and SVM convergence matter.

| Stage | Approximate time | Main resource |
| --- | ---: | --- |
| CSV preparation and integrity checks | 2-10 minutes | CPU / SSD |
| Offline CRS replay | 20-180 minutes | CPU; A100 does not accelerate CRS |
| Embedding export | 10-40 minutes | A100 / tokenizer CPU |
| Probes, permutation diagnostics, conditional probes | 20-120 minutes | CPU |
| TCAV and direction interventions | 5-30 minutes | A100 / head batching |
| Report generation | 1-3 minutes | CPU |

Budget roughly **1-6 hours** for the default experiment after setup. A 100-permutation,
5-fit experiment can take **3-10 hours or more**, especially with conditional probes or fewer
CPU cores. Setup/download time is additional. These ranges are not upper bounds.
The embedding estimate corresponds to approximately 380-1,500 requests/second including
tokenization and export; actual throughput must be measured on your allocation.

Full embeddings occupy about 2.8 GB (2.6 GiB), read on demand. CSV streaming and bounded WAF
queues avoid loading every serialized request/log into RAM. 32 GB is a reasonable host-memory
budget with four WAF workers; reduce workers if the node's rules/other jobs consume substantial
memory. Reserve at least 20-40 GB of local disk; unusually verbose WAF logs may require more.

For a measured pilot, replace only the preparation output with
`--output runs/secbert-concepts-pilot/prepared --max-rows 20000`, then use the pilot directory
for replay, encode, and fit. Finally:

```bash
python -m http_attack_agent.concept_analysis estimate \
  --pilot runs/secbert-concepts-pilot --total-rows 907815
```

Replay and encoding manifests contain elapsed seconds and observed rows/second. The estimate
command extrapolates those measured rates and reports probe time separately because probe caps
and repetition counts, rather than total CSV rows alone, determine its cost. Sampled pilot
results are exploratory and must not be presented as the formal full-dataset experiment.

## Reverting this addition

`CHANGE_MANIFEST.json` records the original commit, pre-existing untracked evaluation directory,
original tracked-file hashes, and newly added files. This feature requires no changes outside
this directory. Reverting it consists of removing the files recorded as newly added, after
checking that no subsequent user edits would be lost. Do not use `git clean`, `git reset --hard`,
or delete the pre-existing `evaluation-summary/` directory.

Experiment artifacts are separate under the explicitly selected `runs/...` output directories;
keep or remove those independently. Existing trained checkpoints are not overwritten.

## Method and implementation references

- [Post-hoc Concept Bottleneck Models](https://arxiv.org/html/2205.15480v2)
- [TCAV](https://proceedings.mlr.press/v80/kim18d.html)
- [Official ModSecurity C API](https://github.com/owasp-modsecurity/ModSecurity/tree/v3/master/headers/modsecurity)
- [CRS 4.10.0 rule definitions](https://github.com/coreruleset/coreruleset/tree/v4.10.0/rules)
