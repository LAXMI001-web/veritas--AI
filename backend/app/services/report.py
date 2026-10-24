"""PDF + JSON forensic reports generated strictly from the stored analysis record."""
from __future__ import annotations

import io
import json
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402


def json_report(rec: dict) -> bytes:
    slim = json.loads(json.dumps(rec))
    for r in slim["results"].values():  # drop bulky base64 blobs; keep numbers
        if isinstance(r, dict):
            r.pop("spectrogram_png", None)
            for f in r.get("frames", []) or []:
                f.pop("thumb", None)
    return json.dumps({"report_type": "truthlens-ai-forensic-report", "version": 1, "machine_generated": True, "analysis": slim},
                      indent=2).encode()


def _chart(xs, ys, title, xlabel, thr) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=130)
    ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color="#2563eb")
    ax.axhline(thr, color="#dc2626", ls="--", lw=0.8)
    ax.set_ylim(0, 1); ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel("score (high = suspicious)", fontsize=7)
    ax.tick_params(labelsize=7); fig.tight_layout()
    b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig); b.seek(0)
    return b


def pdf_report(rec: dict) -> bytes:
    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("s", fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"TruthLens AI report {rec['id']}")
    P = lambda t, st=None: Paragraph(escape(str(t)), st or ss["BodyText"])  # noqa: E731
    a, pol = rec["assessment"], rec["policy"]
    S = [Paragraph("TruthLens AI — Forensic Analysis Report", ss["Title"]),
         P("Machine-generated assessment from uncalibrated classifiers. Not definitive forensic proof.", small), Spacer(1, 6)]
    rows = [["Analysis ID", rec["id"]], ["File", rec["filename"]], ["SHA-256", rec["sha256"]], ["Kind", rec["kind"]],
            ["Analyzed (UTC)", rec["created_at"]], ["Duration (s)", rec.get("duration_s")], ["Has audio track", rec["has_audio"]],
            ["Overall assessment", a["overall_label"]], ["Scenario", f"{a['scenario']} — {a['headline']}"]]
    t = Table([[P(x, small) for x in r] for r in rows], colWidths=[40 * mm, 130 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    S += [t, Spacer(1, 8), Paragraph("Modality assessments", ss["Heading2"])]
    mrows = [["Modality", "Verdict", "Score", "Reasons"]]
    for m, r in a["modalities"].items():
        mrows.append([m, r["verdict"], "n/a" if r["score"] is None else f"{r['score']:.3f}", "; ".join(r["reasons"])])
    t = Table([[P(x, small) for x in r] for r in mrows], colWidths=[22 * mm, 32 * mm, 18 * mm, 98 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    S += [t, Spacer(1, 6)]
    if a.get("conflicts"):
        S += [Paragraph("Model disagreements / conflicts", ss["Heading3"])] + [P("• " + c, small) for c in a["conflicts"]]
    v = rec["results"].get("video") or rec["results"].get("image")
    if v and v.get("status") == "ok" and rec["kind"] == "video":
        fr = [f for f in v["frames"] if f["score"] is not None]
        S += [Paragraph("Frame-level evidence", ss["Heading2"]),
              RLImage(_chart([f["t"] for f in fr], [f["score"] for f in fr], "FA3-CLIP frame scores", "time (s)", pol["frame_suspicious_cut"]), 170 * mm, 56 * mm),
              P(f"Sampling: {v['sampling']}. Aggregates: {v['aggregates']}. Duplicate frames excluded: {v['duplicate_frames']}.", small)]
    au = rec["results"].get("audio")
    if au and au.get("status") == "ok":
        w = [x for x in au["windows"] if x["score"] is not None]
        if w:
            S += [Paragraph("Audio segment evidence", ss["Heading2"]),
                  RLImage(_chart([x["start"] for x in w], [x["score"] for x in w], "CNN+Conformer window scores", "window start (s)", pol["frame_suspicious_cut"]), 170 * mm, 56 * mm),
                  P(f"Aggregates: {au['aggregates']}. Silent windows skipped: {au['silent_windows']}.", small)]
    tr, cc = rec["results"].get("transcript"), rec["results"].get("text_crosscheck")
    if tr is not None:
        S.append(Paragraph("Transcript and voice-vs-wording check", ss["Heading2"]))
        if tr.get("status") != "ok":
            S.append(P(f"Transcript {tr.get('status')}: {tr.get('reason')}", small))
        else:
            tc = rec["results"].get("text_check", {})
            S.append(P(f"Engine: {tr['engine']}. {tr['note']}", small))
            S.append(P(f"Wording check ({tc.get('method')}): {tc.get('verdict')}, risk score {tc.get('score')}. " + " ".join(tc.get("reasons", [])), small))
            if cc:
                S.append(P(f"Cross-check: {cc['headline']} " + " ".join(cc.get("explanation", [])), small))
            for sg in (cc or {}).get("segments", tr["segments"]):
                flag = " [VOICE/WORDING DISAGREE]" if sg.get("mismatch") else ""
                S.append(P(f"{sg['start']}–{sg['end']} s | voice {sg.get('audio_class', 'n/a')} ({sg.get('audio_score')}) | wording {sg.get('text_class', 'n/a')}{flag}: {sg['text']}", small))
    ivs = (v or {}).get("intervals", []) + (au or {}).get("intervals", [])
    if ivs:
        S += [Paragraph("Suspicious intervals (sampled evidence)", ss["Heading3"])] + [P(f"• {i['start']}–{i['end']} s, n={i['n']}, max={i['max']}", small) for i in ivs]
    ig = rec.get("integrity")
    if ig and "metadata" in ig:
        S += [Paragraph("Integrity inspection", ss["Heading2"]), P("Metadata: " + json.dumps(ig["metadata"], default=str)[:1500], small)]
        for o in ig["observations"]:
            S.append(P("• " + o, small))
        S.append(P("Heuristic indicators: " + json.dumps(ig["heuristic_indicators"], default=str)[:1500], small))
        S.append(P("Provenance: " + ig["provenance"]["note"], small))
    ex = rec["explanation"]
    for title, key in (("Observed evidence", "observed_evidence"), ("Model interpretation", "model_interpretation"),
                       ("Possible explanations", "possible_explanations"), ("Limitations", "limitations"),
                       ("Suggested verification steps", "verification_steps")):
        S.append(Paragraph(title, ss["Heading3"]))
        S += [P("• " + x, small) for x in ex[key]] or [P("—", small)]
    S += [Paragraph("Processing configuration & models", ss["Heading2"]),
          P("Policy: " + json.dumps(pol), small), P("Models: " + json.dumps(rec["models"], default=str), small)]
    if rec.get("warnings"):
        S += [P("Warnings: " + "; ".join(rec["warnings"]), small)]
    S += [Spacer(1, 8), P("Confidence note: scores are uncalibrated classifier outputs (high = more suspicious). "
                          "'Likely authentic' does not mean proven authentic. " + rec["disclaimer"], small)]
    doc.build(S)
    return buf.getvalue()
