"""Evidence-grounded explanation. Every sentence is built from fields in the analysis record; nothing is inferred
from a score alone. Output is split into: observed evidence, model interpretation, possible explanations,
limitations, verification steps."""
from __future__ import annotations

SCEN_TEXT = {
    "A": "Both modalities scored below the suspicion band. This does not prove authenticity.",
    "B": "The video model did not find strong suspicious evidence in the sampled frames, while the audio model "
         "scored the audio as suspicious. This is a mixed-authenticity warning.",
    "C": "The video model scored sampled frames as suspicious while the audio model did not find comparable "
         "suspicious patterns. The whole video is not assumed to be fake and no specific technique is inferred.",
    "D": "Suspicious scores appeared in both modalities. The models ran independently, so this does not show that "
         "the two streams were manipulated together.",
    "E": "At least one modality fell in the inconclusive band or lacked enough evidence, so no overall verdict is forced.",
    "F": "The file has no audio track; audio is unavailable (not counted as real or fake) and only the video is assessed.",
    "G": "One analysis failed or was unavailable; the successful results are shown and the overall result is incomplete.",
}


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def explain(rec: dict) -> dict:
    a = rec["assessment"]
    res = rec["results"]
    pol = rec["policy"]
    observed, interp, possible, limits, verify = [], [], [], [], []
    thr, m = pol["suspicious_threshold"], pol["inconclusive_margin"]

    v = res.get("video") or res.get("image")
    if v and v.get("status") == "ok":
        kind = "video" if "video" in res else "image"
        fr = [f for f in v["frames"] if f.get("score") is not None]
        observed.append(f"FA3-CLIP scored {v['scored']} {'sampled frame(s)' if kind == 'video' else 'image'}"
                        f"{'' if not v.get('failed_frames') else ' (%d failed to decode)' % v['failed_frames']}.")
        if kind == "video" and fr:
            top = max(fr, key=lambda f: f["score"])
            low = min(fr, key=lambda f: f["score"])
            observed.append(f"Highest frame score {_f(top['score'])} at t={_f(top['t'])}s (frame {top['index']}); "
                            f"lowest {_f(low['score'])} at t={_f(low['t'])}s.")
            observed.append(f"Aggregates — mean {_f(v['aggregates']['mean'])}, median {_f(v['aggregates']['median'])}, "
                            f"fraction ≥{pol['frame_suspicious_cut']}: {_f(v['aggregates']['fraction'])}.")
            cm = v.get("crop_modes", {})
            observed.append(f"Crop: {cm.get('face', 0)} face crop(s), {cm.get('full_frame', 0)} full frame(s), {cm.get('no_face', 0)} frame(s) with no detectable face (not scored).")
            ev = v.get("evidence") or {}
            if ev:
                observed.append(f"Face found in {ev.get('candidate_frames_with_face')} of {ev.get('candidate_frames')} candidate frames"
                                + (f"; scored frames span {ev['temporal_coverage']:.0%} of the video." if ev.get("temporal_coverage") is not None else "."))
            if v.get("topk_used"):
                observed.append(f"Primary score = mean of the {v['topk_used']} highest-scoring frames (so a manipulated minority is not averaged away).")
            for iv in v.get("intervals", []):
                observed.append(f"Sampled-frame interval {_f(iv['start'])}–{_f(iv['end'])}s with {iv['n']} consecutive "
                                f"frames ≥{pol['interval_min_score']} (max {_f(iv['max'])}).")
            if v.get("duplicate_frames"):
                observed.append(f"{v['duplicate_frames']} sampled frame(s) were near-duplicates of the previous one and excluded from aggregation.")
        elif fr:
            observed.append(f"Fake-class probability {_f(fr[0]['score'], 3)} (logits {fr[0]['logits']}).")
        interp.append(f"The {v['aggregation_method']} score {_f(v['score'])} was compared with a suspicion threshold of "
                      f"{thr} ± {m} (policy).")
        limits.append("FA3-CLIP is a single-frame classifier: it has no temporal modelling and its softmax output is "
                      "an uncalibrated score, not a probability of fakeness.")
        limits.append("The FFT/saliency views show model inputs or sensitivity, not proof of manipulation.")
    au = res.get("audio")
    if au and au.get("status") == "ok":
        sc = [w for w in au["windows"] if w["score"] is not None]
        observed.append(f"CNN+Conformer scored {au['scored']} window(s) of ≈{au['window_seconds']}s "
                        f"({au['silent_windows']} silent window(s) skipped) over {_f(au['duration_s'], 1)}s of audio.")
        if sc:
            top = max(sc, key=lambda w: w["score"])
            observed.append(f"Highest window score {_f(top['score'])} at {_f(top['start'])}–{_f(top['end'])}s; "
                            f"mean {_f(au['aggregates']['mean'])}, median {_f(au['aggregates']['median'])}.")
        for iv in au.get("intervals", []):
            observed.append(f"Audio interval {_f(iv['start'])}–{_f(iv['end'])}s with {iv['n']} consecutive windows ≥{pol['interval_min_score']}.")
        interp.append(f"The audio {au['aggregation_method']} score {_f(au['score'])} was compared with the same policy band.")
        limits.append("The audio model was designed around ≈2 s inputs and its score is uncalibrated; overlapping "
                      "windows are correlated, and it cannot identify a speaker, a cloning method, or a manipulation type.")
    for mod, r in a["modalities"].items():
        if r["verdict"] in ("unavailable", "failed", "not_present"):
            observed.append(f"{mod.capitalize()}: {r['verdict'].replace('_', ' ')} — {'; '.join(r['reasons'])}")
        for w in r.get("warnings", []) or []:
            observed.append(f"{mod.capitalize()} warning: {w}")
        if r["verdict"] == "inconclusive":
            interp.append(f"{mod.capitalize()} is inconclusive: {'; '.join(r['reasons'])}")
    if a["scenario"] in SCEN_TEXT:
        interp.append(f"Scenario {a['scenario']}: {SCEN_TEXT[a['scenario']]}")
    for c in a.get("conflicts", []):
        observed.append(f"Conflict: {c}")
    if a.get("secondary_scenario") == "I":
        interp.append("Scenario I: high scores are concentrated in sampled intervals; these are sampled-evidence "
                      "intervals, not precise manipulation boundaries.")
    # possible explanations (neutral, never asserted)
    if a["scenario"] == "B":
        possible += ["Audio replacement, voice conversion or synthesis, audio editing, heavy compression/noise, or a "
                     "recording condition outside the audio model's training data (none of these is established)."]
    if a["scenario"] == "C":
        possible += ["Visual manipulation of some frames, low-quality or unusual footage, or domain shift in the face "
                     "classifier (none of these is established)."]
    if a["scenario"] == "D":
        possible += ["Genuinely manipulated streams, or both models reacting to compression/quality artefacts."]
    if a["scenario"] == "A":
        possible += ["Authentic media, or manipulation of a type the models were not trained to detect."]
    if a["scenario"] in ("E",):
        possible += ["Short/poor-quality media, too few frames or audio windows, or genuinely ambiguous content."]
    limits += ["Scores come from classifiers whose accuracy on unseen manipulation types, codecs, or languages has not "
               "been measured in this project unless the Model Evaluation page shows results.",
               "'Likely authentic' does not mean the media is proven authentic."]
    verify += ["Check the original source and upload chain; look for a trusted provenance record (e.g. C2PA).",
               "Review the highlighted timestamps manually (lip-sync, lighting, blink/mouth detail, audio breaths and room tone).",
               "Re-run on the highest-quality copy available; compare with other copies using the Media Comparison page."]
    if a["scenario"] in ("B", "C", "D"):
        verify += ["Ask for an independent verification (reverse search, contact the claimed source)."]
    return {"observed_evidence": observed, "model_interpretation": interp, "possible_explanations": possible,
            "limitations": limits, "verification_steps": verify}
