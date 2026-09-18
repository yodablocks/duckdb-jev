"""Phase 2: DuckDB scalar functions over Jev.

Gated on Phase 1. Registering these before the calibration gate passes
means semantic ORDER BY sorts by a number nobody has checked, which is the
exact silent failure the build spec is structured to prevent. `register`
refuses unless the gate passed or override=True.

Signatures follow the build spec. The struct return is settled: returning
NULL on low confidence poisons ORDER BY unpredictably (NULLs sort last in
DuckDB regardless of direction, so low-confidence rows silently clump at
one end), and returning a bare score hides the uncertainty the caller
needs to filter on. Structs expose both; jev_score_val covers the common
case where the caller has already decided to trust the score.

    jev_bool(text, question)      -> STRUCT(value BOOLEAN, prob DOUBLE)
    jev_choice(text, options[])   -> STRUCT(value VARCHAR, prob DOUBLE, confidence DOUBLE)
    jev_score(text, rubric[])     -> STRUCT(score DOUBLE, confidence DOUBLE)
    jev_score_val(text, rubric[]) -> DOUBLE

Note on jev_bool: Noul returns a probability with no separate confidence
field, so `prob` is the probability of yes and `value` is prob >= 0.5.
That matches the two-field struct in the spec.

Note on jev_score scale: Score is a probability-weighted mean over level
INDICES, so the range is 0..len(rubric)-1, not 0..1. Scores from different
rubrics are not comparable. ORDER BY across mixed rubrics is meaningless.

Known limitation, stated rather than hidden: each UDF call issues one
request for one row. A SELECT with three semantic columns over 100 rows
is 300 requests where batching would be 100, and the state tokens are
paid for once per question instead of once per row. This is the exact
per-row pattern the build spec flags as wasting Jev's architecture, and
it is why Phase 3 exists.

It is left as-is for Phase 2 deliberately. DuckDB scalar UDFs are called
per row, so batching requires either a vectorised UDF (`type="arrow"`)
or a two-pass pattern: pre-warm the cache with one batched pass, then let
the per-row UDFs read cached answers for free. The cache makes the second
approach work today, and `test_udf.py` reports the request count so the
cost is visible instead of surprising.

Requires duckdb, which is NOT needed for Phase 1:  pip install duckdb
"""

from __future__ import annotations

import json
from pathlib import Path

from client import JevClient, choice as q_choice, noul as q_noul, score as q_score


# Measured against the live API, not assumed.
#
# The build spec's signatures are jev_score(text, rubric[]) and
# jev_choice(text, options[]): no question parameter, unlike jev_bool.
# That reads fine and does not work. With a generic instruction, a rubric
# phrased around "the topic" never says WHICH topic, so Jev scores almost
# everything near the top of the scale:
#
#   "NASA confirmed the orbiter completed its lunar transfer burn."  2.76
#   "Selling my road bike, $400, barely used, local pickup only."    2.86
#
# The bike outranks the orbiter. ORDER BY over that is noise. Naming the
# judgment separates them completely:
#
#   instructions="How directly is this message about spaceflight?"
#   -> orbiter 3.0, bike 0.0
#
# So the two-argument forms are kept for spec compatibility but are a
# trap, and the *_q variants take the question explicitly. Put the
# judgment in the question; the rubric only describes the levels.
DEFAULT_SCORE_QUESTION = (
    "Rate this against the criteria. NOTE: no judgment was specified, so "
    "this generic instruction scores most inputs alike. Use jev_score_q "
    "or jev_score_val_q and state what to judge."
)
DEFAULT_CHOICE_QUESTION = (
    "Which option best describes this? NOTE: no judgment was specified. "
    "Use jev_choice_q and state what to judge."
)


def _gate_passed(results: Path) -> tuple[bool, str]:
    f = results / "results.json"
    if not f.exists():
        return False, f"no Phase 1 results at {f}"
    try:
        g = json.loads(f.read_text()).get("gate", {})
    except json.JSONDecodeError as e:
        return False, f"unreadable results.json: {e}"
    if not g:
        return False, "results.json has no gate section"
    return bool(g.get("passed")), "; ".join(g.get("failures", [])) or "gate passed"


def register(con, client: JevClient, results_dir: Path | None = None,
             override: bool = False):
    """Register the four functions on a DuckDB connection.

    Raises unless Phase 1's gate passed. Pass override=True only to
    experiment knowingly with uncalibrated output.
    """
    # Gate first, before anything else can fail for an unrelated reason: a
    # missing duckdb should not mask an unpassed calibration gate.
    results_dir = results_dir or Path(__file__).resolve().parents[1] / "results"
    passed, detail = _gate_passed(results_dir)
    if not passed and not override:
        raise RuntimeError(
            f"Phase 1 calibration gate has not passed ({detail}).\n"
            "Semantic ORDER BY over uncalibrated probabilities sorts by a "
            "number nobody has verified, and fails silently.\n"
            "Run the calibration first, or pass override=True to proceed "
            "knowingly."
        )

    import duckdb  # noqa: F401  (imported for a clear error if missing)

    def jev_bool(text: str, question: str) -> dict:
        if text is None or question is None:
            return {"value": None, "prob": None}
        r = client.ask(text, {"q": q_noul(question)})
        p = r["answers"]["q"]["noul"]
        return {"value": p >= 0.5, "prob": p}

    def _choice(text: str, options: list[str], question: str) -> dict:
        if text is None or not options:
            return {"value": None, "prob": None, "confidence": None}
        r = client.ask(
            text, {"q": q_choice(question, {o: None for o in options})}
        )
        a = r["answers"]["q"]
        return {
            "value": a["choice"],
            "prob": a.get("probabilities", {}).get(a["choice"]),
            "confidence": a.get("confidence"),
        }

    def jev_choice(text: str, options: list[str]) -> dict:
        return _choice(text, options, DEFAULT_CHOICE_QUESTION)

    def jev_choice_q(text: str, options: list[str], question: str) -> dict:
        return _choice(text, options, question)

    def _score(text: str, rubric: list[str], question: str) -> dict:
        if text is None or not rubric:
            return {"score": None, "confidence": None}
        r = client.ask(text, {"q": q_score(question, list(rubric))})
        a = r["answers"]["q"]
        return {"score": a["score"], "confidence": a.get("confidence")}

    def jev_score(text: str, rubric: list[str]) -> dict:
        return _score(text, rubric, DEFAULT_SCORE_QUESTION)

    def jev_score_q(text: str, rubric: list[str], question: str) -> dict:
        return _score(text, rubric, question)

    def jev_score_val(text: str, rubric: list[str]) -> float | None:
        """Bare score for the common ORDER BY case.

        Scale is 0..len(rubric)-1. Do not mix rubrics in one sort.
        """
        return _score(text, rubric, DEFAULT_SCORE_QUESTION)["score"]

    def jev_score_val_q(
        text: str, rubric: list[str], question: str
    ) -> float | None:
        """The form to actually use for ORDER BY. See DEFAULT_SCORE_QUESTION."""
        return _score(text, rubric, question)["score"]

    con.create_function(
        "jev_bool", jev_bool, ["VARCHAR", "VARCHAR"],
        "STRUCT(value BOOLEAN, prob DOUBLE)",
    )
    con.create_function(
        "jev_choice", jev_choice, ["VARCHAR", "VARCHAR[]"],
        "STRUCT(value VARCHAR, prob DOUBLE, confidence DOUBLE)",
    )
    con.create_function(
        "jev_score", jev_score, ["VARCHAR", "VARCHAR[]"],
        "STRUCT(score DOUBLE, confidence DOUBLE)",
    )
    con.create_function(
        "jev_score_val", jev_score_val, ["VARCHAR", "VARCHAR[]"], "DOUBLE",
    )
    # The question-bearing variants. These are the ones to use: see the
    # DEFAULT_SCORE_QUESTION note for why the two-argument forms score
    # almost everything alike.
    con.create_function(
        "jev_choice_q", jev_choice_q, ["VARCHAR", "VARCHAR[]", "VARCHAR"],
        "STRUCT(value VARCHAR, prob DOUBLE, confidence DOUBLE)",
    )
    con.create_function(
        "jev_score_q", jev_score_q, ["VARCHAR", "VARCHAR[]", "VARCHAR"],
        "STRUCT(score DOUBLE, confidence DOUBLE)",
    )
    con.create_function(
        "jev_score_val_q", jev_score_val_q,
        ["VARCHAR", "VARCHAR[]", "VARCHAR"], "DOUBLE",
    )
    return con
