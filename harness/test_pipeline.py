"""End-to-end pipeline test against a mock Jev server.

Without an API key the scoring pass cannot run, so this substitutes a
local server that speaks the documented response contract. It proves the
harness works: request shape, cache, retries, budget ceiling, partial
failure handling and the full metric path. What it cannot prove is
anything about Jev itself.

The mock is deliberately miscalibrated (overconfident) so the gate has
something real to fail on, and we can see the gate actually fires.

Run: python3 tools/duckdb-jev/harness/test_pipeline.py
"""

from __future__ import annotations

import json
import random
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import client as C


class Handler(BaseHTTPRequestHandler):
    fail_next = 0          # simulate transient 529s
    request_log: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.request_log.append(body)

        if Handler.fail_next > 0:
            Handler.fail_next -= 1
            self.send_response(529)
            self.send_header("retry-after", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"overloaded"}')
            return

        state = body["state"]
        rng = random.Random(hash(state) & 0xFFFF)
        # Crude signal so metrics have something non-degenerate to chew on.
        positive = "space" in state.lower() or "nasa" in state.lower()
        base = 0.85 if positive else 0.15
        p = min(0.99, max(0.01, base + rng.uniform(-0.12, 0.12)))

        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                # Negated question gets an intentionally asymmetric answer,
                # so the invariant check has a violation to detect.
                answers[qid] = {"type": "noul",
                                "noul": round(1 - p + 0.08, 4) if "negat" in qid else round(p, 4)}
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                probs = {o: 0.05 for o in opts}
                probs[opts[0] if positive else opts[-1]] = 0.7
                total = sum(probs.values())
                probs = {k: round(v / total, 4) for k, v in probs.items()}
                pick = max(probs, key=probs.get)
                answers[qid] = {"type": "choice", "choice": pick,
                                "confidence": probs[pick],
                                "probabilities": probs}
            elif q["type"] == "score":
                n = len(q["criteria"])
                dist = {str(i): 0.1 for i in range(n)}
                dist[str(n - 1 if positive else 0)] = 0.7
                tot = sum(dist.values())
                dist = {k: v / tot for k, v in dist.items()}
                val = sum(int(k) * v for k, v in dist.items())
                answers[qid] = {"type": "score", "score": round(val, 4),
                                "confidence": round(max(dist.values()), 4),
                                "probabilities": {k: round(v, 4) for k, v in dist.items()},
                                "legend": {str(i): str(c) for i, c in enumerate(q["criteria"])}}

        payload = {"model": body["model"], "answers": answers,
                   "usage": {"input_tokens": len(state) // 4, "output_tokens": 12 * len(answers)}}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    C.ENDPOINT = f"http://127.0.0.1:{port}/v1/systemone"
    import os
    os.environ["TYPESAFE_AI_API_KEY"] = "test-key"

    import run_calibration as R
    R.CORPUS = R.DATA / "corpus.jsonl"

    tmp = Path(tempfile.mkdtemp())
    rows = R.load_corpus()[:60]

    # --- batching: one request per row, not one per question ---
    Handler.request_log.clear()
    cl = C.JevClient(tmp / "c.sqlite")
    cl._post.__func__  # noqa: B018
    responses, errors = R.score_rows(rows, cl, workers=4)
    assert len(responses) == 60, f"expected 60, got {len(responses)}"
    assert not errors, errors
    assert len(Handler.request_log) == 60, (
        f"batching broken: {len(Handler.request_log)} requests for 60 rows")
    assert len(Handler.request_log[0]["questions"]) == 4, "4 questions per request"
    print(f"PASS  batching: 60 rows -> {len(Handler.request_log)} requests, "
          f"4 questions each")

    # --- cache: second pass must make zero requests ---
    before = len(Handler.request_log)
    cl2 = C.JevClient(tmp / "c.sqlite")
    r2, _ = R.score_rows(rows, cl2, workers=4)
    assert len(Handler.request_log) == before, "cache did not prevent re-requests"
    assert len(r2) == 60 and cl2.usage.cache_hits == 60
    assert cl2.usage.total_tokens == 0, "cached run must spend nothing"
    print(f"PASS  cache: replay made 0 new requests, {cl2.usage.cache_hits} hits")

    # --- retry on 529 ---
    Handler.fail_next = 2
    cl3 = C.JevClient(tmp / "c3.sqlite", max_retries=5)
    resp = cl3.ask("a nasa space mission report about orbital mechanics",
                   R.questions_for("space"))
    assert "answers" in resp
    print("PASS  retry: recovered from 2x HTTP 529")

    # --- budget ceiling fails loud ---
    cl4 = C.JevClient(tmp / "c4.sqlite", token_budget=50)
    try:
        for r in rows:
            cl4.ask(r["text"], R.questions_for(r["probe"]), use_cache=False)
        raise AssertionError("budget ceiling did not fire")
    except C.BudgetExceeded as e:
        print(f"PASS  budget ceiling fired: {str(e)[:60]}...")

    # --- partial failure surfaced, not nulled ---
    class Boom(C.JevClient):
        def ask(self, state, questions, use_cache=True):
            if "nasa" in state.lower():
                raise RuntimeError("simulated row failure")
            return super().ask(state, questions, use_cache)

    b = Boom(tmp / "c5.sqlite")
    ok, errs = R.score_rows(rows, b, workers=2)
    print(f"PASS  partial failure: {len(ok)} ok, {len(errs)} surfaced as errors")
    assert len(ok) + len(errs) == 60

    # --- full metric path + gate ---
    report = R.analyze(rows, responses)
    assert "boolean" in report and "negation_invariant" in report
    assert "choice" in report and "score" in report
    assert report["score"]["observed_range"][1] <= R.SCORE_SCALE_MAX
    print(f"PASS  metrics computed: "
          f"ECE={report['boolean']['ece']['ece']:.3f} "
          f"Brier={report['boolean']['brier']:.3f} "
          f"inversion={report['boolean']['ranking']['inversion_rate']:.3f}")
    print(f"      negation violation="
          f"{report['negation_invariant']['mean_abs_violation']:.3f} "
          f"(mock injects 0.08 by construction)")
    print(f"      score range={report['score']['observed_range']} "
          f"of 0..{R.SCORE_SCALE_MAX}")

    assert "gate" in report
    print(f"PASS  gate evaluated: passed={report['gate']['passed']} "
          f"failures={report['gate']['failures']}")

    # --- diagram renders ---
    out = R.reliability_diagram(report, tmp / "rel.png")
    assert out and out.exists() and out.stat().st_size > 5000
    print(f"PASS  reliability diagram rendered ({out.stat().st_size:,} bytes)")

    srv.shutdown()
    print("\nall pipeline tests passed")


if __name__ == "__main__":
    main()
