# duckdb-jev

DuckDB scalar functions over TypeSafe AI's Jev model, so unstructured text
columns can be filtered and sorted like numeric ones.

**Status (2026-09-18): Phase 1 harness complete and tested. Calibration
numbers not yet produced: no API key is available in this environment.**
The corpus, client, metrics, gate and notebook all run; only the scoring
pass is blocked. See [Running it](#running-it).

---

## Why calibration is Phase 1 and not Phase 2

The product is `ORDER BY` over a semantic score. If the probabilities are
not calibrated, the sort key is a meaningless number and every query fails
*silently*: rows come back in an order, just not a defensible one. Nothing
errors, nothing looks wrong, and the result is wrong. So the calibration
harness is the first deliverable and the gate on everything after it.

`udf.register()` enforces this in code: it refuses to register the SQL
functions unless `results/results.json` records a passing gate.

## What is measured

Two families, because a gate on calibration alone is the wrong gate.

**Calibration** (Brier + Murphy decomposition, ECE, MCE) asks: is a stated
0.7 really 70%?

**Ranking** (Spearman, Kendall, AUC, pairwise inversion rate) asks: does
sorting by this put rows in the right order?

They come apart in both directions. Probabilities squashed into
[0.48, 0.52] but perfectly ordered give an ECE of 0.485 with zero
inversions: terrible calibration, flawless sort. (That exact case is a
test in `test_metrics.py`.) Conversely a model calibrated in aggregate can
still invert many individual pairs and produce a visibly wrong page of
results. `ORDER BY` depends on the second family; the original spec's gate
named only the first.

**Invariants** are the third thing measured, and the most defensible,
because they need **no ground-truth labels at all**:

| Invariant | What it catches |
|---|---|
| Negation symmetry: `P(q)` vs `1 - P(not q)` | Probabilities that move with phrasing rather than evidence. The `jev-1.13` jaggedness page explicitly disclaims this identity, which is what makes it worth measuring. |
| Score rubric ordinality | Levels are scored independently and the model never sees level numbers, so ordinality is imposed by *our* array order. A non-ordinal rubric yields a garbage sort key. |
| Choice option-order invariance | Position effects in the option list. |

Because these compare the model against itself, they are immune to the
label-provenance problem below. They are necessary, not sufficient:
passing negation symmetry does not imply calibration, but failing it means
the probabilities cannot be thresholded reliably, which kills
`WHERE prob > x` independently of calibration.

## Label provenance

This decides whether any of the numbers mean anything.

The corpus is **20 Newsgroups**, and each document's label is the
newsgroup its author chose to post it to: a human judgment, recorded by a
human, independent of this project. Hand-labeling the corpus ourselves
would have measured Jev's agreement with Claude and published it as
calibration, producing an authoritative-looking number worth nothing.

Selection constraints, in the order they bound the result:

1. Human-labeled by provenance, licensed for research use.
2. Single-factor labels, matching the spec's rule for questions.
3. Clear of Jev's documented weak spots. The jaggedness page for
   `jev-1.13` names math/counting, date comparison, and hex/RGB numeric
   representations. Topic membership touches none of them, so a
   miscalibration we measure is not secretly a counting failure.
4. **Labels span the probability range.** Sampling only obvious cases pins
   every prediction at 0 or 1, ECE comes out tiny, the gate passes
   vacuously, and nothing is learned.

Constraint 4 drives the stratified sampler: each probe draws clear
positives, topically adjacent **near misses**, and plainly unrelated
negatives, weighted toward the near misses. Default corpus is 360 rows
across 3 probes, inside the spec's 300-500 band.

The vendor's self-reported 67.8% agreement against averaged frontier
judgments is not used as a baseline anywhere here. Agreement with other
models is not calibration.

## Gate

Thresholds were fixed **before** any results were seen.

| Condition | Threshold | Rationale |
|---|---|---|
| ECE | ≤ 0.10 | A stated 0.8 that is really 0.7 is tolerable for ranking; wider and the probability is decorative. |
| Inversion rate | ≤ 0.15 | Past roughly one bad pair in six, a sorted page looks visibly wrong. |
| Resolution | > 0 | At or below zero, the model is not separating classes at all. A model predicting the base rate every time scores a respectable Brier and is useless for `ORDER BY`. |
| Negation asymmetry | ≤ 0.15 | Beyond this, phrasing moves the answer as much as evidence does. |

## Running it

Phase 1 needs **no new packages**: numpy, scipy, sklearn, matplotlib and
requests are already present. `duckdb` is a Phase 2 dependency only.

```sh
# one-time corpus download (~14MB, human-labeled, research licence)
mkdir -p ~/scikit_learn_data/20news_home
curl -L -A "Mozilla/5.0" -o /tmp/20news.tar.gz \
  http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar xzf /tmp/20news.tar.gz -C ~/scikit_learn_data/20news_home

# build the corpus
python3 harness/corpus.py ../../.data/jev-calibration/corpus.jsonl

# verify the harness with no API key and no spend
python3 harness/test_metrics.py     # 26 known-answer metric tests
python3 harness/test_pipeline.py    # end-to-end against a mock Jev server

# then, with a key:
export TYPESAFE_AI_API_KEY=...
python3 harness/run_calibration.py --pilot   # 10 rows + cost extrapolation
python3 harness/run_calibration.py           # full run
python3 harness/run_calibration.py --analyze-only   # recompute, no spend
```

Phase 2 additionally needs `pip install duckdb` (run it yourself; this
repo does not install packages autonomously).

## Execution layer

Built in Phase 1 rather than retrofitted in Phase 3, because all four are
cheaper to build now and the calibration run needs them anyway.

- **Batching.** Every question for a row goes in one request. Jev answers
  independent questions against one state in a single parallel pass, so
  the calibration run issues 1 request per row instead of 4, and pays for
  the state tokens once instead of four times. Verified by test: 60 rows →
  60 requests, 4 questions each.
- **Cache.** Content hash of `(model, state, questions)` → SQLite (WAL,
  thread-local connections). Verified: a replay run makes 0 requests and
  spends 0 tokens.
- **Cost ceiling.** Hard token budget that raises `BudgetExceeded` rather
  than degrading. `ORDER BY` over a large table is an easy way to spend
  real money by accident, so the failure mode is a loud stop.
- **Concurrency.** Bounded pool, exponential backoff with jitter,
  honouring `retry-after`. Retries 429/529/5xx; does not retry 401/422,
  which will not improve. Partial failures are collected and surfaced, not
  silently nulled.

Token usage is recorded per call from the first request, because rate
limits and per-token pricing are undocumented: the Phase 3 cost ceiling
can only be calibrated from usage we measure ourselves.

## SQL surface (Phase 2)

```
jev_bool(text, question)      -> STRUCT(value BOOLEAN, prob DOUBLE)
jev_choice(text, options[])   -> STRUCT(value VARCHAR, prob DOUBLE, confidence DOUBLE)
jev_score(text, rubric[])     -> STRUCT(score DOUBLE, confidence DOUBLE)
jev_score_val(text, rubric[]) -> DOUBLE
```

Structs, not bare values: returning NULL on low confidence poisons
`ORDER BY` unpredictably (DuckDB sorts NULLs last regardless of direction,
so low-confidence rows silently clump at one end), and a bare score hides
the uncertainty the caller needs. `jev_score_val` covers the case where
the caller has already decided to trust the score.

`jev_bool` exposes `prob` with no separate confidence because Noul returns
a probability and has no confidence field: `value` is `prob >= 0.5`.

### Score scale: the sharp edge

Score returns a probability-weighted mean over level **indices**, so an
`n`-level rubric spans `0..n-1`, **not** `0..1`. Therefore:

- `jev_score_val` output is **not comparable across different rubrics**.
- `ORDER BY` mixing rubrics is meaningless, and nothing in SQL will warn.

## Limitations

- **One corpus, one domain.** English newsgroup posts. Calibration is a
  property of model *and* domain; these numbers will not transfer to
  contracts, tickets or governance proposals without re-running.
- **Topic membership is an easy judgment**, so treat the results as an
  upper bound on harder ORDER BY workloads.
- **~120 rows per probe.** Ten-bin ECE is noisy at that size. Bins carry
  Wilson intervals and adaptive (equal-mass) binning is the default; read
  the intervals, not the third decimal.
- **20 Newsgroups labels are themselves noisy** (cross-posting, imperfect
  group choice), which inflates apparent miscalibration.
- **`jev-latest` is a moving target.** The `model` field from each
  response is recorded; these numbers attach to one version.

## Layout

```
harness/corpus.py           stratified corpus builder, provenance notes
harness/client.py           batching, cache, budget, retries, metering
harness/metrics.py          calibration + ranking + invariants + gate
harness/run_calibration.py  scoring run, analysis, reliability diagram
harness/udf.py              Phase 2 DuckDB functions (gated on Phase 1)
harness/test_metrics.py     known-answer tests for every metric
harness/test_pipeline.py    end-to-end test against a mock Jev server
notebook/calibration.ipynb  the publishable artifact
results/                    results.json + reliability.png (after a run)
```

Corpus and cached responses live in `.data/jev-calibration/` (gitignored):
raw response bodies can contain corpus text, and the cache is a build
artifact.
