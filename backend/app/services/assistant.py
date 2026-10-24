"""Rule-based, evidence-grounded analyst assistant. No LLM, no external calls: every answer is assembled from fields of
the stored record. Unknown questions get an honest 'not answerable from this record'."""
from __future__ import annotations


def ask(rec: dict, q: str) -> dict:
    ql = q.lower()
    a, ex, res = rec["assessment"], rec["explanation"], rec["results"]
    v = res.get("video") or res.get("image")
    au = res.get("audio")
    facts, hyps = [], []
    if any(k in ql for k in ("highest", "peak", "most suspicious", "which timestamp", "when")):
        if v and v.get("status") == "ok":
            fr = [f for f in v["frames"] if f["score"] is not None]
            if fr:
                t = max(fr, key=lambda f: f["score"]); facts.append(f"Highest frame score {t['score']:.3f} at t={t['t']}s (frame {t['index']}).")
        if au and au.get("status") == "ok":
            w = [x for x in au["windows"] if x["score"] is not None]
            if w:
                t = max(w, key=lambda x: x["score"]); facts.append(f"Highest audio window score {t['score']:.3f} at {t['start']:.2f}–{t['end']:.2f}s.")
    elif any(k in ql for k in ("disagree", "conflict", "differ")):
        facts += [f"Scenario {a['scenario']}: {a['headline']}"] + a.get("conflicts", [])
        facts += [f"{m}: {r['verdict']} (score {r['score']})" for m, r in a["modalities"].items()]
        hyps += ex["possible_explanations"]
    elif any(k in ql for k in ("limit", "caveat", "reliab", "trust")):
        facts += ex["limitations"]
    elif any(k in ql for k in ("check", "verify", "next", "additional")):
        facts += ex["verification_steps"]
    elif any(k in ql for k in ("why", "flag", "evidence", "support")):
        facts += ex["observed_evidence"]; facts += ex["model_interpretation"]; hyps += ex["possible_explanations"]
    else:
        return {"answer": "I can answer from this analysis record about: why it was flagged, the highest-scoring timestamp, "
                          "modality disagreement, evidence, limitations and suggested checks. I can't answer other questions "
                          "without inventing information.", "facts": [], "hypotheses": [], "source": "analysis record"}
    return {"answer": "\n".join(facts) or "No supporting data in this record.", "facts": facts, "hypotheses": hyps,
            "source": "analysis record (rule-based, no external LLM)"}
