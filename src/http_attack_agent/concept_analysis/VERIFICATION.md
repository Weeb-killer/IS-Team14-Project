# Verification record

Verified locally on Windows with Python 3.14 and CPU PyTorch. No A100 measurements are claimed.

| Check | Result |
| --- | --- |
| Existing regression suite plus new tests | 132 passed, 1 native integration test skipped |
| Focused tests | Concept mapping, event-level joint concepts, unknown-label handling, content overlap, independent held-out evaluation, immutable head parameters |
| Local Transformer integration | Tiny saved BERT checkpoint loaded without downloads; embeddings/logits exported; input order checked; long-body truncation recorded |
| Synthetic workflow | Probe fitting, repeated CAVs, conditional probes, original-head TCAV, interventions, one-request explanation, Markdown/CSV/PNG report |
| Real CSV preparation | Original 907,815 rows scanned; 1,000 sampled rows retained as 700 train / 150 validation / 150 test; request IDs and hashes saved |
| Static analysis | New package passes Ruff checks |
| Existing repository files | No tracked files changed; pre-existing evaluation results preserved |
| Direct CRS-versus-classifier comparison | Synthetic end-to-end reports, paired identical test subsets, unknown handling, content deduplication, taxonomy coverage, confidence intervals, and integrity checks |
| Validation isolation | Mutating only test targets changes test results but leaves selected operating thresholds unchanged |
| CRS decision semantics | Actual aggregate telemetry, not arbitrary rule hits or severity sums; terminal CAPEC mapping excludes ancestors |

The native ModSecurity/CRS integration test is intentionally conditional on
`CONCEPT_ANALYSIS_CRS`. This host has no usable native ModSecurity installation or Docker, so
the test was skipped. Real CRS behavior and A100 throughput must be checked on the Linux host
using `check-waf` and the measured pilot commands in the README. Replay itself runs the same
SQLi/phase/logging preflight before starting the dataset.

The synthetic demo does not substitute for the real integration test. The previously trained
SecBERT checkpoint resides on the user's training environment and was not present locally;
its full-dataset explanation results have not been generated here.

The existing evaluation metric tests emit two warnings for deliberately absent positive classes;
these are unrelated to the new feature. Test scratch directories and generated demo outputs
are kept under the already ignored `runs/` directory.
