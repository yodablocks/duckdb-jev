"""Phase 2: the SQL layer, executed against a real DuckDB connection.

Why this file exists: udf.py was written from the build spec and, until
now, had never run. The four `create_function` type signatures
(`STRUCT(value BOOLEAN, prob DOUBLE)`, `VARCHAR[]`, ...) were correct on
paper and had never been accepted by an actual DuckDB. Struct return
types and list parameters are exactly the API surface that fails on first
contact, so "designed" and "demonstrated" are different claims and only
this file can make the second one.

Runs against a mock Jev server by default, so it costs nothing and needs
no API key. Pass --live to hit the real API (cache-served rows are free).

Run:  python3 harness/test_udf.py [--live]
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import HTTPServer
from pathlib import Path

import client as C


def _mock_server() -> tuple[HTTPServer, str]:
    """Reuse the pipeline test's Jev mock so the contract stays in one place."""
    import test_pipeline as TP

    srv = HTTPServer(("127.0.0.1", 0), TP.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1/systemone"


ROWS = [
    ("r1", "NASA confirmed the orbiter completed its lunar transfer burn."),
    ("r2", "Selling my road bike, $400, barely used, local pickup only."),
    ("r3", "The patient presented with elevated blood pressure after therapy."),
    ("r4", "Does anyone know a good restaurant near the train station?"),
]

RUBRIC = [
    "The message has nothing to do with the topic.",
    "The message mentions the topic only in passing.",
    "The message discusses the topic as one of several subjects.",
    "The message is entirely and directly about the topic.",
]


def main() -> None:
    live = "--live" in sys.argv

    try:
        import duckdb
    except ImportError:
        sys.exit(
            "duckdb is not installed. Phase 2 needs it:\n\n"
            "    pip install duckdb\n\n"
            "Phase 1 (the calibration harness) does not."
        )

    print(f"duckdb {duckdb.__version__}   mode: {'LIVE API' if live else 'mock'}")

    srv = None
    if live:
        if not C.has_api_key():
            sys.exit("--live needs an API key; see README.")
    else:
        srv, endpoint = _mock_server()
        C.ENDPOINT = endpoint
        import os

        os.environ.setdefault("TYPESAFE_AI_API_KEY", "test-key")

    import udf

    tmp = Path(tempfile.mkdtemp())
    # Budget is a real ceiling even in tests: a bug that loops on rows
    # should stop, not bill.
    client = C.JevClient(tmp / "udf.sqlite", token_budget=200_000)
    con = duckdb.connect()

    # --- the gate must be enforced, not decorative ----------------------
    empty = tmp / "no-results"
    empty.mkdir()
    try:
        udf.register(con, client, results_dir=empty)
        raise AssertionError("register() ignored a missing Phase 1 gate")
    except RuntimeError as e:
        assert "gate has not passed" in str(e)
    print("PASS  register() refuses without a passing Phase 1 gate")

    # --- registration against a real connection -------------------------
    # The moment of truth for the four type signatures.
    udf.register(con, client)
    print("PASS  all four functions registered on a live DuckDB connection")

    con.execute("CREATE TABLE docs (id VARCHAR, body VARCHAR)")
    con.executemany("INSERT INTO docs VALUES (?, ?)", ROWS)

    # --- jev_bool: STRUCT(value BOOLEAN, prob DOUBLE) -------------------
    r = con.execute(
        "SELECT id, jev_bool(body, 'Is this message about space or astronomy?') AS j "
        "FROM docs ORDER BY id"
    ).fetchall()
    for _id, j in r:
        assert isinstance(j, dict), f"expected STRUCT, got {type(j)}"
        assert set(j) == {"value", "prob"}, j
        assert isinstance(j["value"], bool), j
        assert 0.0 <= j["prob"] <= 1.0, j
    print(f"PASS  jev_bool returns STRUCT(value, prob): {r[0][1]}")

    # Struct fields must be addressable in SQL, which is the whole point
    # of returning a struct rather than a bare value.
    got = con.execute(
        "SELECT id FROM docs "
        "WHERE (jev_bool(body, 'Is this message about space or astronomy?')).prob > 0.5"
    ).fetchall()
    print(f"PASS  struct field usable in WHERE: {[g[0] for g in got]}")

    # --- jev_choice: VARCHAR[] parameter --------------------------------
    r = con.execute(
        "SELECT id, jev_choice(body, ['space', 'medicine', 'for_sale', 'other']) AS j "
        "FROM docs ORDER BY id"
    ).fetchall()
    for _id, j in r:
        assert set(j) == {"value", "prob", "confidence"}, j
        assert isinstance(j["value"], str), j
    print(f"PASS  jev_choice accepts VARCHAR[] and returns STRUCT: {r[0][1]['value']!r}")

    # --- jev_score + the accessor ---------------------------------------
    rubric_sql = "[" + ", ".join(f"'{lv}'" for lv in RUBRIC) + "]"
    r = con.execute(
        f"SELECT id, jev_score(body, {rubric_sql}) AS j FROM docs ORDER BY id"
    ).fetchall()
    for _id, j in r:
        assert set(j) == {"score", "confidence"}, j
        # Score is a probability-weighted mean over level INDICES, so the
        # range is 0..len(rubric)-1, not 0..1.
        assert 0.0 <= j["score"] <= len(RUBRIC) - 1, j
    print(f"PASS  jev_score in 0..{len(RUBRIC)-1}: {r[0][1]}")

    # --- the headline claim: semantic ORDER BY --------------------------
    # Uses the question-bearing variant. The two-argument jev_score_val
    # cannot know WHAT to judge, and against the live API it scores almost
    # everything near the top (orbiter 2.76, bike-for-sale 2.86), so a
    # sort over it is noise. See DEFAULT_SCORE_QUESTION in udf.py.
    QUESTION = "How directly is this message about spaceflight or astronomy?"
    ordered = con.execute(
        f"""SELECT id, jev_score_val_q(body, {rubric_sql}, ?) AS relevance
            FROM docs
            ORDER BY relevance DESC""",
        [QUESTION],
    ).fetchall()
    scores = [row[1] for row in ordered]
    assert all(a >= b for a, b in zip(scores, scores[1:])), f"not sorted: {ordered}"
    print("PASS  semantic ORDER BY executed and is correctly sorted:")
    for _id, s in ordered:
        print(f"         {_id}  {s:.3f}")

    # Sorted is not the same as right: a constant column is also "sorted".
    # The space row must top the ranking and the off-topic rows must not,
    # or the sort key carries no signal. This is the assertion that
    # catches an under-specified question.
    rank = {row[0]: i for i, row in enumerate(ordered)}
    assert rank["r1"] == 0, f"space row should rank first: {ordered}"
    for other in ("r2", "r3", "r4"):
        assert rank["r1"] < rank[other], f"space did not outrank {other}: {ordered}"
    spread = max(scores) - min(scores)
    assert spread > 0.5, (
        f"scores span only {spread:.2f} of the {len(RUBRIC)-1}-point scale; "
        "the sort key is not discriminating (under-specified question?)"
    )
    print(f"PASS  ordering is semantically right, spread {spread:.2f} "
          f"of {len(RUBRIC)-1} points")

    # And the documented trap: the question-free form does NOT discriminate.
    flat = con.execute(
        f"SELECT jev_score_val(body, {rubric_sql}) AS s FROM docs"
    ).fetchall()
    flat_spread = max(x[0] for x in flat) - min(x[0] for x in flat)
    print(f"      (question-free jev_score_val spread: {flat_spread:.2f} "
          "- documented trap, not a failure)")

    # --- NULL handling, measured rather than assumed --------------------
    # DuckDB does not call a scalar UDF when an argument is NULL (default
    # null_handling), so the whole struct comes back NULL and the in-Python
    # guards in udf.py never execute on this path. No error, but also not
    # the NULL-filled struct the code anticipates.
    con.execute("INSERT INTO docs VALUES ('r5', NULL)")
    r = con.execute(
        "SELECT id, jev_bool(body, 'Is this about space?') AS j "
        "FROM docs WHERE id = 'r5'"
    ).fetchall()
    assert r[0][1] is None, f"expected NULL struct, got {r[0][1]!r}"
    print("PASS  NULL input returns NULL (UDF not invoked) and does not raise")

    # This is the empirical basis for the settled struct-return decision.
    # DuckDB defaults to NULLS LAST, so a NULL sort key lands at the end in
    # BOTH directions: it does not flip with ASC/DESC the way a real value
    # would. So a row whose score is NULL is not "ranked low", it is
    # silently parked at one end regardless of what the query asked for.
    # That is why the UDFs return a struct exposing confidence instead of
    # returning NULL when the model is unsure.
    seen = {}
    for direction in ("DESC", "ASC"):
        rows = con.execute(
            f"SELECT id, jev_score_val(body, {rubric_sql}) AS s "
            f"FROM docs ORDER BY s {direction}"
        ).fetchall()
        seen[direction] = rows[-1][0]
    assert seen["DESC"] == seen["ASC"] == "r5", seen
    print(f"PASS  NULL sorts last in BOTH directions ({seen}): "
          "demonstrates why NULL is not a usable sort key")

    # --- cost accounting ------------------------------------------------
    u = client.usage
    print(
        f"\nusage: {u.requests} requests, {u.cache_hits} cache hits, "
        f"{u.total_tokens:,} tokens"
    )
    # Documents the per-row pattern rather than hiding it: each UDF call is
    # one request, so N rows x M semantic columns is N*M requests. See the
    # note in udf.py about why Phase 3 batching matters.
    n_rows = len(ROWS)
    print(
        f"note: {u.requests} requests for {n_rows} rows across several "
        "columns confirms the per-row call pattern (see udf.py)."
    )

    if srv:
        srv.shutdown()
    print("\nall UDF tests passed")


if __name__ == "__main__":
    main()
