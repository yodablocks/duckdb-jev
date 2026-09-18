"""Phase 1 calibration run.

Scores the corpus with Jev, computes the metrics, writes results.json and
a reliability diagram. Every response is cached, so a second run costs
nothing and the notebook is reproducible without re-spending.

Usage:
    export TYPESAFE_AI_API_KEY=...
    python3 run_calibration.py --pilot          # 10 rows, prints cost estimate
    python3 run_calibration.py                  # full run
    python3 run_calibration.py --analyze-only   # recompute from cache

Design note on batching: each row sends one request carrying every
question for that row (noul, the negated noul, choice and score). Jev
answers independent questions against one state in a single parallel
pass, so this is one request where the naive shape would be four, and the
state tokens are paid for once instead of four times.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from client import BudgetExceeded, JevClient, choice, has_api_key, noul, score
from corpus import PROBES
import metrics as M

DATA = Path(__file__).resolve().parents[3] / ".data" / "jev-calibration"
CORPUS = DATA / "corpus.jsonl"
CACHE = DATA / "responses.sqlite"
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Score rubric. Levels are ordered low -> high and each describes a
# concrete situation that stands on its own, because the model sees only
# the descriptions: never the level numbers, never the neighbours.
#
# The resulting score is a probability-weighted mean over level INDICES,
# so with 4 levels the range is 0..3, NOT 0..1. Scores from different
# rubrics are not comparable, and ORDER BY across mixed rubrics is
# meaningless.
RELEVANCE_RUBRIC = [
    "The message has nothing to do with the topic.",
    "The message mentions the topic only in passing or as an aside.",
    "The message discusses the topic as one of several subjects.",
    "The message is entirely and directly about the topic.",
]
SCORE_SCALE_MAX = len(RELEVANCE_RUBRIC) - 1

CHOICE_OPTIONS = {
    "space": "Spaceflight, astronomy, or space exploration.",
    "medicine": "Medicine, health, disease, or medical treatment.",
    "for_sale": "An offer to sell or trade a specific item.",
    "cars": "Cars, driving, or automotive maintenance.",
    "computer_graphics": "Computer graphics, image formats, or rendering.",
    "other": "None of the other options fit this message.",
}


def negate(question: str) -> str:
    """Build the logical complement mechanically from the same string.

    Hand-writing a separate negative sentence introduces a confound: if
    the measured asymmetry is large, we cannot tell "Jev violates the
    identity" from "my two strings were not actually complements". So the
    negation is derived from the positive question by construction, and
    the only difference between the framings is the inserted negation.

    Note this is still not a *guaranteed* complement in natural language;
    it is only as good as the transformation. Hence the third framing in
    questions_for, which triangulates.
    """
    q = question.strip().rstrip("?")
    for prefix in ("Is this message ", "Is this "):
        if q.startswith(prefix):
            return f"{prefix}NOT {q[len(prefix):]}?"
    return f"Is it NOT the case that: {q}?"


def questions_for(probe: str) -> dict:
    """Every judgment for one row, asked in a single request."""
    p = PROBES[probe]
    topic = {
        "space": "spaceflight, astronomy, or space exploration",
        "medical": "medicine, health, or medical treatment",
        "forsale": "offering an item for sale",
    }[probe]

    return {
        # Boolean -> jev_bool
        "bool": noul(p["question"]),
        # The negated twin, same request, derived mechanically from the
        # positive string. The jaggedness page disclaims
        # P(q) == 1 - P(not q), so this measures the violation instead of
        # assuming it away.
        "bool_negated": noul(negate(p["question"])),
        # A third framing, semantically equivalent to the positive but
        # worded differently. This separates two explanations for any
        # asymmetry: if paraphrase disagreement is as large as negation
        # disagreement, the model is sensitive to wording in general and
        # the negation result is not specifically about negation.
        "bool_paraphrase": noul(
            f"Does this message concern {topic}?"
        ),
        # Choice -> jev_choice
        "choice": choice(
            "Which single subject best describes what this message is about?",
            CHOICE_OPTIONS,
        ),
        # Score -> jev_score
        "score": score(
            f"How directly is this message about {topic}?",
            RELEVANCE_RUBRIC,
        ),
        # Score with the rubric REVERSED. Levels are scored independently
        # and the model never sees level numbers, so ordinality is imposed
        # entirely by our array order. If a reversed rubric does not mirror
        # the score, the rubric is not ordinal and every sort key built
        # from it is noise. One extra question in a request we are already
        # paying for.
        "score_reversed": score(
            f"How directly is this message about {topic}?",
            list(reversed(RELEVANCE_RUBRIC)),
        ),
    }


def question_set_hash() -> str:
    """Fingerprint of every question this run will ask.

    Cache keys include the questions, so editing a rubric or a probe
    question silently invalidates the whole cache and triggers a full
    re-spend. Stamping this into results.json makes that visible instead
    of surprising, and marks which question set a number belongs to.
    """
    import hashlib

    blob = json.dumps(
        {p: questions_for(p) for p in PROBES}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        sys.exit(f"No corpus at {CORPUS}. Run: python3 corpus.py {CORPUS}")
    return [json.loads(l) for l in CORPUS.open() if l.strip()]


def score_rows(rows, client, workers=4, use_cache=True):
    """Bounded pool. Partial failures are surfaced, never silently nulled."""
    out, errors = {}, {}

    def one(row):
        return row["row_id"], client.ask(
            row["text"], questions_for(row["probe"]), use_cache=use_cache
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, r): r for r in rows}
        done = 0
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                rid, resp = fut.result()
                out[rid] = resp
            except BudgetExceeded:
                for f in futures:
                    f.cancel()
                raise
            except Exception as exc:
                errors[row["row_id"]] = f"{type(exc).__name__}: {exc}"
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(rows)} "
                      f"({client.usage.total_tokens:,} tokens, "
                      f"{client.usage.cache_hits} cached)", flush=True)

    return out, errors


def analyze(rows, responses) -> dict:
    """Compute every metric, per primitive type and per probe."""
    report: dict = {
        "n_corpus": len(rows),
        "n_scored": len(responses),
        "question_set_hash": question_set_hash(),
        "score_scale": {
            "levels": len(RELEVANCE_RUBRIC),
            "range": [0, SCORE_SCALE_MAX],
            "note": "probability-weighted mean over level indices, not 0-1; "
                    "not comparable across different rubrics",
        },
    }

    by_id = {r["row_id"]: r for r in rows}
    paired = [(by_id[rid], responses[rid]) for rid in responses if rid in by_id]
    if not paired:
        return report

    def ans(resp, key):
        return resp.get("answers", {}).get(key, {})

    # ---- Boolean (noul) ----
    p_bool = np.array([ans(r, "bool").get("noul", np.nan) for _, r in paired])
    y = np.array([row["label"] for row, _ in paired], float)
    ok = ~np.isnan(p_bool)

    # Calibration is computed on confidently-labeled rows only. For the
    # ambiguous near-miss groups the "no" is arguable, so a correct 0.6
    # scored against a wrong False reads as miscalibration and could fail
    # the gate on label error rather than model error. Ranking keeps every
    # row: those hard cases are exactly where sort order matters, and
    # ranking only needs the pairs the labels do order.
    conf = ok & np.array([row.get("label_confident", True) for row, _ in paired])

    if ok.sum():
        report["boolean"] = {
            "n": int(ok.sum()),
            "n_calibration": int(conf.sum()),
            "n_excluded_ambiguous": int(ok.sum() - conf.sum()),
            "brier": M.brier(p_bool[conf], y[conf]),
            "decomposition": M.brier_decomposition(p_bool[conf], y[conf]),
            "ece": M.ece(p_bool[conf], y[conf], n_bins=10, adaptive=True),
            "ece_fixed_bins": M.ece(p_bool[conf], y[conf], n_bins=10, adaptive=False)["ece"],
            # Reported for comparison: if these diverge a lot, the exclusion
            # is doing heavy lifting and should be stated prominently.
            "ece_all_rows_incl_ambiguous": M.ece(p_bool[ok], y[ok], n_bins=10)["ece"],
            "ranking": M.rank_metrics(p_bool[ok], y[ok]),
            "calibration_note": "Brier/ECE exclude near-miss groups whose "
                                "negative label is arguable; ranking uses all rows.",
        }

        # Per-probe, because one bad probe can hide inside the average.
        report["boolean"]["by_probe"] = {}
        for probe in PROBES:
            m = np.array([row["probe"] == probe for row, _ in paired]) & ok
            if m.sum() > 10:
                report["boolean"]["by_probe"][probe] = {
                    "n": int(m.sum()),
                    "brier": M.brier(p_bool[m], y[m]),
                    "ece": M.ece(p_bool[m], y[m], n_bins=5)["ece"],
                    "ranking": M.rank_metrics(p_bool[m], y[m]),
                }

        # Per-stratum. The near_miss rows are the informative ones: if the
        # model is confident there, it is confidently wrong somewhere.
        report["boolean"]["by_stratum"] = {}
        for stratum in ("positive", "near_miss", "far"):
            m = np.array([row["stratum"] == stratum for row, _ in paired]) & ok
            if m.sum() > 5:
                report["boolean"]["by_stratum"][stratum] = {
                    "n": int(m.sum()),
                    "mean_prob": float(p_bool[m].mean()),
                    "base_rate": float(y[m].mean()),
                    "brier": M.brier(p_bool[m], y[m]),
                }

    # ---- negation invariant (needs no labels) ----
    p_neg = np.array([ans(r, "bool_negated").get("noul", np.nan) for _, r in paired])
    both = ok & ~np.isnan(p_neg)
    if both.sum():
        report["negation_invariant"] = M.negation_symmetry(p_bool[both], p_neg[both])

    # Paraphrase control. A semantically equivalent rewording should give
    # the same answer, so |P(q) - P(paraphrase)| is a floor for how much
    # wording alone moves this model. If it is comparable to the negation
    # violation, the negation result is general wording sensitivity rather
    # than anything specific to negation, and must be reported as such.
    p_par = np.array([ans(r, "bool_paraphrase").get("noul", np.nan) for _, r in paired])
    par = ok & ~np.isnan(p_par)
    if par.sum():
        diff = np.abs(p_bool[par] - p_par[par])
        report["paraphrase_control"] = {
            "n": int(par.sum()),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
            "p95_abs_diff": float(np.percentile(diff, 95)),
            "max_abs_diff": float(diff.max()),
            "note": "floor for wording sensitivity; compare against "
                    "negation_invariant.mean_abs_violation before attributing "
                    "asymmetry to negation specifically",
        }
        neg_v = report.get("negation_invariant", {}).get("mean_abs_violation")
        if neg_v is not None and diff.mean() > 0:
            report["paraphrase_control"]["negation_to_paraphrase_ratio"] = float(
                neg_v / diff.mean()
            )

    # ---- Choice ----
    ch_rows = [(row, r) for row, r in paired if row.get("choice_label")]
    if ch_rows:
        correct, confs = [], []
        for row, r in ch_rows:
            a = ans(r, "choice")
            if "choice" not in a:
                continue
            correct.append(1.0 if a["choice"] == row["choice_label"] else 0.0)
            confs.append(a.get("confidence", np.nan))
        correct, confs = np.array(correct), np.array(confs, float)
        m = ~np.isnan(confs)
        if m.sum():
            # For Choice, calibration means: does stated confidence predict
            # whether the pick was right?
            report["choice"] = {
                "n": int(m.sum()),
                "accuracy": float(correct[m].mean()),
                "brier_on_confidence": M.brier(confs[m], correct[m]),
                "ece_on_confidence": M.ece(confs[m], correct[m], n_bins=10),
                "decomposition": M.brier_decomposition(confs[m], correct[m]),
            }

    # ---- Score ----
    sc = np.array([ans(r, "score").get("score", np.nan) for _, r in paired])
    m = ~np.isnan(sc)
    if m.sum():
        # Brier/ECE do not apply to a continuous score. Rank metrics do,
        # and ranking is what ORDER BY depends on.
        report["score"] = {
            "n": int(m.sum()),
            "observed_range": [float(sc[m].min()), float(sc[m].max())],
            "mean": float(sc[m].mean()),
            "ranking_vs_label": M.rank_metrics(sc[m], y[m]),
            "note": "Brier/ECE omitted: they are classification metrics and "
                    "do not apply to a continuous score.",
        }
        # A stratum-ordered check: far < near_miss < positive should hold
        # if the rubric is genuinely ordinal.
        means = {}
        for stratum in ("far", "near_miss", "positive"):
            mm = np.array([row["stratum"] == stratum for row, _ in paired]) & m
            if mm.sum() > 5:
                means[stratum] = float(sc[mm].mean())
        report["score"]["stratum_means"] = means
        if len(means) == 3:
            report["score"]["stratum_monotonic"] = (
                means["far"] < means["near_miss"] < means["positive"]
            )

        # Rubric ordinality invariant. With the rubric reversed, a genuinely
        # ordinal scale must mirror: score_rev ~= SCALE_MAX - score. This is
        # the check that the array order we impose corresponds to something
        # the model actually perceives as ordered. Needs no ground truth.
        sc_rev = np.array(
            [ans(r, "score_reversed").get("score", np.nan) for _, r in paired]
        )
        mr = m & ~np.isnan(sc_rev)
        if mr.sum():
            mirrored = SCORE_SCALE_MAX - sc_rev[mr]
            resid = np.abs(sc[mr] - mirrored)
            report["score"]["rubric_ordinality"] = {
                "n": int(mr.sum()),
                "mean_abs_mirror_error": float(resid.mean()),
                "median_abs_mirror_error": float(np.median(resid)),
                "p95_abs_mirror_error": float(np.percentile(resid, 95)),
                # As a fraction of the full scale, so it is readable
                # independently of how many levels the rubric has.
                "mean_error_as_scale_fraction": float(
                    resid.mean() / SCORE_SCALE_MAX
                ),
                "correlation_with_mirror": (
                    float(np.corrcoef(sc[mr], mirrored)[0, 1])
                    if len(np.unique(sc[mr])) > 1
                    and len(np.unique(mirrored)) > 1
                    else float("nan")
                ),
                "note": "reversed rubric should mirror: "
                        "score_reversed ~= scale_max - score. Large error "
                        "means the rubric is not ordinal to the model and "
                        "any sort key built from it is noise.",
            }

    # ---- gate ----
    gate = M.Gate()
    b = report.get("boolean", {})
    gate.check(
        ece_val=b.get("ece", {}).get("ece"),
        inversion=b.get("ranking", {}).get("inversion_rate"),
        resolution=b.get("decomposition", {}).get("resolution"),
        negation=report.get("negation_invariant", {}).get("mean_abs_violation"),
        # Gated separately: jev_score_val is what ORDER BY sorts on, so a
        # Score that inverts pairs must stop Phase 2 on its own merits.
        score_inversion=report.get("score", {})
        .get("ranking_vs_label", {})
        .get("inversion_rate"),
        choice_ece=report.get("choice", {}).get("ece_on_confidence", {}).get("ece"),
    )
    report["gate"] = {
        "passed": gate.passed,
        "failures": gate.failures,
        "thresholds": {
            "max_ece": gate.max_ece,
            "max_inversion_rate": gate.max_inversion_rate,
            "min_resolution": gate.min_resolution,
            "max_negation_violation": gate.max_negation_violation,
        },
    }
    return report


def reliability_diagram(report, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = report.get("boolean", {}).get("ece", {}).get("bins", [])
    if not bins:
        return None

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), height_ratios=[3, 1], sharex=True
    )
    x = [b["mean_pred"] for b in bins]
    obs = [b["observed"] for b in bins]
    lo = [o - b["ci_low"] for o, b in zip(obs, bins)]
    hi = [b["ci_high"] - o for o, b in zip(obs, bins)]

    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect calibration")
    ax.errorbar(x, obs, yerr=[lo, hi], fmt="o-", color="#2b6cb0",
                capsize=3, lw=1.5, label="observed (95% Wilson CI)")
    ax.set_ylabel("observed frequency")
    ax.set_title(
        f"jev_bool reliability  |  ECE={report['boolean']['ece']['ece']:.3f}  "
        f"Brier={report['boolean']['brier']:.3f}  n={report['boolean']['n']}"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax2.bar(x, [b["n"] for b in bins], width=0.06, color="#a0aec0")
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="score 10 rows and estimate full-run cost")
    ap.add_argument("--analyze-only", action="store_true",
                    help="recompute metrics from cache, make no API calls")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=2_000_000)
    args = ap.parse_args()

    rows = load_corpus()
    if args.pilot:
        rows = rows[:10]
    elif args.limit:
        rows = rows[: args.limit]

    client = JevClient(CACHE, token_budget=args.token_budget)

    if args.analyze_only:
        responses = {}
        for r in rows:
            from client import Cache
            hit = client.cache.get(
                Cache.key(client.model, r["text"], questions_for(r["probe"]))
            )
            if hit:
                responses[r["row_id"]] = hit
        print(f"Loaded {len(responses)} cached responses.")
        errors = {}
    else:
        if not has_api_key():
            sys.exit(
                "No API key set.\n\n"
                "  export TYPESAFE_AI_API_KEY=...\n\n"
                "Phase 1 cannot produce calibration numbers without it. The "
                "harness and corpus are ready; only the scoring pass is blocked."
            )
        print(f"Scoring {len(rows)} rows ({args.workers} workers)...")
        responses, errors = score_rows(rows, client, args.workers)

    if errors:
        print(f"\n{len(errors)} rows failed (surfaced, not nulled):")
        for rid, e in list(errors.items())[:5]:
            print(f"  {rid}: {e}")

    u = client.usage
    print(f"\nUsage: {u.requests} requests, {u.cache_hits} cache hits, "
          f"{u.input_tokens:,} in + {u.output_tokens:,} out "
          f"= {u.total_tokens:,} tokens")

    if args.pilot and u.requests:
        per_row = u.total_tokens / u.requests
        full = load_corpus()
        print(f"\nPilot estimate: {per_row:,.0f} tokens/row "
              f"-> ~{per_row * len(full):,.0f} tokens for {len(full)} rows.")

    if not responses:
        print("\nNothing scored; no metrics to compute.")
        return

    report = analyze(rows, responses)
    report["usage"] = {
        "requests": u.requests,
        "cache_hits": u.cache_hits,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "failed_rows": len(errors),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    diagram = reliability_diagram(report, RESULTS / "reliability.png")

    print(f"\nWrote {RESULTS / 'results.json'}")
    if diagram:
        print(f"Wrote {diagram}")

    g = report.get("gate", {})
    print(f"\nGATE: {'PASS' if g.get('passed') else 'FAIL'}")
    for f in g.get("failures", []):
        print(f"  - {f}")
    if not g.get("passed"):
        print("\nPer the build spec: stop and rethink before Phase 2.")


if __name__ == "__main__":
    main()
