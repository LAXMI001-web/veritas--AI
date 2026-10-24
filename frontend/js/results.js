"use strict";
/* ---------- result rendering: every value shown comes from the stored analysis record ---------- */
const OVERALL_CLASS = { likely_authentic: "ok", suspected_manipulation: "bad", mixed_authenticity: "warn", inconclusive: "warn", analysis_incomplete: "na" };
const VERDICT_CLASS = { likely_authentic: "ok", suspected: "bad", inconclusive: "warn", unavailable: "na", failed: "na", not_present: "na" };
const VERDICT_TEXT = { likely_authentic: "Likely authentic", suspected: "Suspected manipulation", inconclusive: "Inconclusive", unavailable: "Unavailable", failed: "Failed", not_present: "No audio track" };
const SCENARIOS = [
  ["A", "Video likely authentic + audio likely authentic", "Likely authentic (with warning that this is not proof)"],
  ["B", "Video likely authentic + audio suspected", "Mixed-authenticity warning"],
  ["C", "Video suspected + audio likely authentic", "Mixed-authenticity warning"],
  ["D", "Video suspected + audio suspected", "Suspected manipulation (assessed independently per modality)"],
  ["E", "One or both modalities inconclusive", "Inconclusive"],
  ["F", "Video has no audio track", "Video-only verdict; audio marked unavailable"],
  ["G", "One modality failed or model unavailable", "Analysis incomplete; valid results still shown"],
  ["H", "Models disagree / evidence conflicts", "Conflict flagged; no forced confident verdict"],
  ["I", "Suspicious intervals but not whole video", "Sampled-frame / window evidence shown, not precise boundaries"],
];
const badge = (cls, text) => el("span", { class: "badge b-" + cls, text });

function audioHealthNote(m) {   // shown wherever the audio model is used, when its probe says it is a constant classifier
  const h = m && m.audio && m.audio.health; if (!h || h.status !== "suspect") return null;
  return el("div", { class: "note", style: "border-color:var(--warn)" }, el("strong", { text: "⚠ Audio model looks non-discriminative. " }), h.message,
    " Until this is fixed, audio verdicts are shown as “Inconclusive” instead of “Likely authentic”/“Suspected”.");
}
function bullets(items) { return items && items.length ? el("ul", { class: "ev" }, items.map(t => el("li", { text: t }))) : el("div", { class: "mut small", text: "—" }); }

function modalityCard(name, v, res) {
  const kids = [el("div", { class: "row" }, el("h3", { text: name, style: "margin:0" }), badge(VERDICT_CLASS[v.verdict] || "na", VERDICT_TEXT[v.verdict] || v.verdict))];
  if (v.score !== null && v.score !== undefined) {
    const uninterpreted = v.model_unvalidated || v.model_degenerate;
    kids.push(el("div", { class: "score", style: `color:${uninterpreted ? "var(--na)" : scoreColor(v.score)}` }, fmt(v.score, 3)));
    if (uninterpreted) kids.push(el("div", { class: "small mut", text: "Raw model output, not interpreted (see reason below)." }));
    kids.push(el("div", { class: "mut small" }, tip("suspiciousness score", TIPS.score), " · ", tip("uncalibrated", TIPS.uncal), ` · ${res.aggregation_method} aggregation · ${res.model || ""}`));
  }
  kids.push(bullets(v.reasons)); (v.warnings || []).forEach(w => kids.push(el("div", { class: "note", text: "⚠ " + w })));
  if (res && res.aggregates) kids.push(el("div", { class: "small mut", text: `mean ${fmt(res.aggregates.mean)} · median ${fmt(res.aggregates.median)} · fraction≥cut ${fmt(res.aggregates.fraction)} · top-k ${fmt(res.aggregates.topk)} · n=${res.scored}` }));
  return el("div", { class: "card" }, kids);
}

function matrixTable(active, secondary) {
  return el("div", { class: "tblwrap" }, el("table", { class: "matrix" },
    el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: "Scenario" }), el("th", { text: "Overall outcome" }))),
    el("tbody", {}, SCENARIOS.map(([c, d, o]) => el("tr", { class: (c === active || c === secondary) ? "active" : "", "aria-current": (c === active) ? "true" : null },
      el("td", { text: c + (c === active ? " ◀" : c === secondary ? " ◀ (also)" : "") }), el("td", { text: d }), el("td", { text: o }))))));
}

/* synchronized evidence timeline: lane 1 = video frame scores, lane 2 = audio waveform, lane 3 = audio window scores */
function timelineCanvas(rec, onPick, selT) {
  const v = rec.results.video, au = rec.results.audio, dur = rec.duration_s || (au && au.duration_s) || 1;
  const c = el("canvas", { role: "img", tabindex: 0, "aria-label": "Synchronized evidence timeline. Use the tables below for the same data in text form." });
  const draw = () => {
    const { g, w, h } = setup(c, 250), L = 70, R = 8, X = t => L + (t / dur) * (w - L - R);
    g.font = "11px system-ui"; g.fillStyle = cssv("--mut");
    ["Video frames", "Audio wave", "Audio windows"].forEach((t, i) => g.fillText(t, 4, 40 + i * 70));
    for (let i = 0; i < 3; i++) { g.strokeStyle = cssv("--line"); g.strokeRect(L, 14 + i * 70, w - L - R, 56); }
    for (let s = 0; s <= dur; s += Math.max(1, Math.round(dur / 8))) { g.fillStyle = cssv("--mut"); g.fillText(s + "s", X(s) - 6, h - 4); }
    const iv = [...((v && v.intervals) || []).map(i => [i, 0]), ...((au && au.intervals) || []).map(i => [i, 2])];
    iv.forEach(([i, lane]) => { g.fillStyle = "rgba(240,97,109,.22)"; g.fillRect(X(i.start), 14 + lane * 70, Math.max(X(i.end) - X(i.start), 3), 56); });
    if (v && v.status === "ok") v.frames.forEach(f => { if (f.t === null) return; const x = X(f.t);
      if (f.score === null) { g.strokeStyle = cssv("--na"); g.beginPath(); g.moveTo(x - 3, 20); g.lineTo(x + 3, 64); g.stroke(); return; }
      g.fillStyle = scoreColor(f.score); g.fillRect(x - 3, 14 + 56 * (1 - f.score), 6, 56 * f.score + 1); });
    if (au && au.status === "ok") {
      const wf = au.waveform, n = wf.max.length; g.strokeStyle = cssv("--accent"); g.beginPath();
      for (let i = 0; i < n; i++) { const x = X(i / n * au.duration_s); g.moveTo(x, 98 - wf.max[i] * 26); g.lineTo(x, 98 - wf.min[i] * 26); } g.stroke();
      au.windows.forEach(wd => { const x0 = X(wd.start), x1 = X(wd.end); if (wd.score === null) { g.fillStyle = "rgba(135,147,165,.25)"; g.fillRect(x0, 154, x1 - x0, 56); return; }
        g.fillStyle = scoreColor(wd.score); g.globalAlpha = .75; g.fillRect(x0, 154 + 56 * (1 - wd.score), x1 - x0, 56 * wd.score + 1); g.globalAlpha = 1; });
    } else if (!au || au.status !== "ok") { g.fillStyle = cssv("--mut"); g.fillText(!rec.has_audio ? "No audio track" : au ? `Audio ${au.status}` : "", L + 8, 98); }
    if (selT !== null && selT !== undefined) { g.strokeStyle = cssv("--text"); g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(selT), 8); g.lineTo(X(selT), 216); g.stroke(); }
    c.onclick = e => { const r = c.getBoundingClientRect(); const t = ((e.clientX - r.left) / r.width * w - L) / (w - L - R) * dur; onPick(Math.min(Math.max(t, 0), dur)); };
  };
  requestAnimationFrame(draw); window.addEventListener("resize", draw, { once: true }); c.redraw = draw; return c;
}

function inspectorPanel(rec, t) {
  const box = el("div", { class: "card" }, el("h3", { text: "Selected timestamp" }));
  if (t === null || t === undefined) { box.append(el("div", { class: "mut", text: "Click the timeline (or a frame) to inspect the frame, audio window and scores at that moment." })); return box; }
  box.append(el("div", { text: `t = ${fmt(t, 2)} s`, style: "font-weight:700" }));
  const v = rec.results.video;
  if (v && v.status === "ok") { const f = v.frames.filter(f => f.t !== null).sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t))[0];
    if (f) box.append(el("div", { class: "row" }, f.thumb ? el("img", { src: "data:image/jpeg;base64," + f.thumb, width: 160, alt: `Sampled frame at ${fmt(f.t)} s` }) : null,
      el("div", {}, el("div", { text: `Nearest sampled frame ${f.index} @ ${fmt(f.t)} s` }), el("div", { text: `FA3-CLIP score: ${fmt(f.score, 3)} (${f.crop || ""})` }),
        el("div", { class: "mut small", text: f.error ? "Frame error: " + f.error : f.duplicate_of_prev ? "Near-duplicate of previous frame (excluded from aggregation)" : "" })))); }
  const au = rec.results.audio;
  if (au && au.status === "ok") { const ws = au.windows.filter(w => w.start <= t && w.end >= t);
    box.append(el("div", { class: "small", text: ws.length ? "Audio windows covering this time: " + ws.map(w => `${fmt(w.start)}–${fmt(w.end)} s → ${w.silent ? "silent (not scored)" : fmt(w.score, 3)}`).join("; ") : "No audio window covers this time." })); }
  const inIv = [...((v && v.intervals) || []), ...((au && au.intervals) || [])].filter(i => t >= i.start && t <= i.end);
  box.append(el("div", { class: "small", text: inIv.length ? `Inside a sampled suspicious interval (${inIv.map(i => fmt(i.start) + "–" + fmt(i.end) + " s").join(", ")}). This is sampled evidence, not a precise manipulation boundary.` : "Not inside a flagged interval." }));
  return box;
}

function timelineSection(rec) {
  const wrap = el("div", { class: "card" }); let sel = null;
  const insp = el("div"), cv = { c: null };
  const pick = t => { sel = t; clear(insp).append(inspectorPanel(rec, t)); cv.c.redraw = null; mount(); };
  const mount = () => { const old = cv.c; cv.c = timelineCanvas(rec, pick, sel); if (old) old.replaceWith(cv.c); };
  cv.c = timelineCanvas(rec, pick, sel); insp.append(inspectorPanel(rec, null));
  wrap.append(el("h2", { text: "Evidence timeline" }), el("div", { class: "mut small", text: "Top: FA3-CLIP frame scores (bar height = score; grey tick = undecodable frame). Middle: audio waveform. Bottom: audio window scores. Red shading = sampled suspicious intervals. Media is not retained, so playback is unavailable; sampled frames stand in." }), cv.c, insp);
  return wrap;
}

function frameGallery(rec, onSel) {
  const v = rec.results.video || rec.results.image; if (!v || v.status !== "ok") return null;
  const cut = rec.policy.frame_suspicious_cut;
  return el("div", { class: "card" }, el("h2", { text: rec.kind === "video" ? "Sampled frames" : "Image" }),
    el("div", { class: "frames" }, v.frames.map(f => el("button", { class: "frame" + (f.score !== null && f.score >= cut ? " sus" : ""), style: "background:var(--panel2);color:inherit", onclick: () => onSel && onSel(f.t), "aria-label": `Frame ${f.index}, score ${fmt(f.score)}` },
      f.thumb ? el("img", { src: "data:image/jpeg;base64," + f.thumb, alt: "" }) : el("div", { class: "mut", text: "no image" }),
      el("div", { style: `color:${scoreColor(f.score)};font-weight:700`, text: f.score === null ? (f.error || "no score") : fmt(f.score, 3) }),
      el("div", { class: "mut", text: f.t !== null ? fmt(f.t, 1) + " s" : "" }), f.duplicate_of_prev ? el("div", { class: "mut", text: "duplicate" }) : null))),
    v.frames.length > 1 ? el("details", {}, el("summary", { text: "Frame score chart" }), (() => { const c = el("canvas", { role: "img", "aria-label": "Frame scores over time" }); requestAnimationFrame(() => lineChart(c, v.frames.filter(f => f.t !== null).map(f => ({ x: f.t, y: f.score })), { thr: cut, xlabel: "time (s)", onPick: onSel })); return c; })()) : null,
    el("div", { class: "note", text: "FA3-CLIP scores each frame independently (no temporal modelling). " + (v.note_crop || "") + (v.sampling ? ` Sampling: ${v.sampling.strategy}, ${v.sampling.sampled || v.frames.length} of ${v.sampling.container_frame_count} frames requested ${v.sampling.requested}; failed ${v.failed_frames}, duplicates ${v.duplicate_frames}.` : "") }));
}

function audioPanel(rec) {
  const a = rec.results.audio; if (!a || a.status !== "ok") return null;
  const c = el("canvas", { role: "img", "aria-label": "Audio window scores" });
  requestAnimationFrame(() => lineChart(c, a.windows.map(w => ({ x: (w.start + w.end) / 2, y: w.score })), { thr: rec.policy.frame_suspicious_cut, xlabel: "window centre (s)", xmax: a.duration_s }));
  return el("div", { class: "card" }, el("h2", { text: "Audio evidence" }),
    el("div", { class: "mut small", text: `${fmt(a.duration_s, 1)} s analysed in ${a.windows.length} windows of ${a.window_seconds} s (hop ${a.hop_seconds} s); ${a.silent_windows} silent window(s) skipped.` }),
    el("h3", { text: "Mel spectrogram (model input representation)" }), el("img", { class: "full", src: "data:image/png;base64," + a.spectrogram_png, alt: "Mel spectrogram of the analysed audio" }),
    el("h3", { text: "Window scores" }), c, el("div", { class: "note", text: a.note }),
    el("details", {}, el("summary", { text: "Window table" }), el("div", { class: "tblwrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["start", "end", "score", "note"].map(h => el("th", { text: h })))),
      el("tbody", {}, a.windows.map(w => el("tr", {}, el("td", { text: fmt(w.start) }), el("td", { text: fmt(w.end) }), el("td", { text: w.score === null ? "—" : fmt(w.score, 3) }), el("td", { text: w.silent ? "silent, not scored" : w.padded ? "zero-padded (short)" : "" }))))))));
}

function explanationPanel(rec, aid) {
  const ex = rec.explanation, S = (cls, title, items) => el("div", { class: cls }, el("h3", { text: title }), bullets(items));
  const ans = el("div", { class: "note", style: "display:none" }), q = el("input", { type: "text", placeholder: "Ask about this analysis, e.g. “Which timestamp has the highest score?”", "aria-label": "Question for the analyst assistant", style: "flex:1;min-width:220px" });
  const ask = async () => { if (!q.value.trim()) return; try { const r = await api(`/analyses/${aid}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q.value }) });
    ans.style.display = "block"; clear(ans).append(el("div", { style: "white-space:pre-wrap", text: r.answer }), el("div", { class: "small", text: "Source: " + r.source })); } catch (e) { toast(e.message, true); } };
  q.addEventListener("keydown", e => e.key === "Enter" && ask());
  return el("div", { class: "card" }, el("h2", { text: "Explanation" }), el("div", { class: "grid g2" },
    S("sec-obs", "1 · Observed evidence", ex.observed_evidence), S("sec-int", "2 · Model interpretation", ex.model_interpretation),
    S("sec-pos", "3 · Possible explanations (not established)", ex.possible_explanations), S("sec-lim", "4 · Limitations & uncertainty", ex.limitations)),
    el("h3", { text: "Suggested verification steps" }), bullets(ex.verification_steps),
    el("h3", { text: "Analyst assistant (answers only from this record; no external AI)" }), el("div", { class: "row" }, q, el("button", { class: "btn sec", onclick: ask, text: "Ask" })), ans);
}

function integrityPanel(ig) {
  if (!ig) return null; if (ig.error) return el("div", { class: "card" }, el("h2", { text: "Integrity inspection" }), el("div", { class: "note", text: ig.error }));
  const m = ig.metadata, flat = (o, p = "") => Object.entries(o || {}).flatMap(([k, v]) => v && typeof v === "object" && !Array.isArray(v) ? flat(v, p + k + ".") : [[p + k, Array.isArray(v) ? JSON.stringify(v) : v]]);
  return el("div", { class: "card" }, el("h2", { text: "Media integrity inspection" }),
    el("h3", { text: "Metadata observations" }), el("div", { class: "tblwrap" }, el("table", {}, el("tbody", {}, flat(m).filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => el("tr", {}, el("td", { text: k, class: "mut" }), el("td", { text: String(v), class: "mono" })))))),
    bullets(ig.observations),
    el("h3", { text: "Heuristic anomaly indicators (review pointers, not classifications)" }), el("pre", { class: "mono", style: "white-space:pre-wrap;margin:0", text: JSON.stringify(ig.heuristic_indicators, null, 1) }),
    el("h3", { text: "Provenance" }), el("div", { class: "note", text: `C2PA/JUMBF marker present: ${ig.provenance.manifest_marker_present ? "yes" : "no"} · verified: NO. ${ig.provenance.note}` }),
    el("h3", { text: "Unavailable checks" }), bullets(ig.unavailable_checks));
}


/* ---------- transcript + voice-vs-wording (shared by file results and the live page) ---------- */
const SEV_CLASS = { ok: "ok", info: "na", warn: "warn", bad: "bad" };
const AUDIO_CLS = { suspected: ["bad", "voice: suspicious"], authentic: ["ok", "voice: no strong flags"], inconclusive: ["warn", "voice: inconclusive"], unknown: ["na", "voice: not scored"] };
const TEXT_CLS = { suspicious: ["bad", "wording: suspicious"], mild: ["warn", "wording: mild indicators"], ordinary: ["ok", "wording: ordinary"] };
const TEXT_VERDICT = { suspicious_wording: ["bad", "Suspicious wording"], mild_indicators: ["warn", "Mild indicators"], ordinary_wording: ["ok", "Ordinary wording"], inconclusive: ["warn", "Inconclusive"] };
function highlighted(text, inds) {   // phrase matches wrapped in <mark>; built with DOM nodes, never innerHTML
  const out = []; let i = 0;
  for (const m of [...(inds || [])].sort((a, b) => a.pos - b.pos)) {
    if (m.pos < i) continue; out.push(text.slice(i, m.pos)); out.push(el("mark", { title: m.category, text: text.slice(m.pos, m.pos + m.phrase.length) })); i = m.pos + m.phrase.length;
  }
  out.push(text.slice(i)); return out;
}
function segRow(sg) {
  const a = AUDIO_CLS[sg.audio_class] || AUDIO_CLS.unknown, t = TEXT_CLS[sg.text_class] || TEXT_CLS.ordinary;
  return el("div", { class: "seg" + (sg.mismatch ? " mism" : "") },
    el("div", { class: "segt mono", text: `${fmt(sg.start, 1)}–${fmt(sg.end, 1)} s` }),
    el("div", { class: "segb" },
      el("div", { class: "segtext" }, highlighted(sg.text, sg.indicators)),
      el("div", { class: "row small" }, badge(a[0], a[1] + (sg.audio_score === null || sg.audio_score === undefined ? "" : ` (${fmt(sg.audio_score, 2)})`)), badge(t[0], t[1]),
        sg.mismatch ? badge("warn", "⚠ voice and wording disagree") : null,
        sg.low_confidence ? el("span", { class: "mut", text: "low transcription confidence" }) : null)));
}
function crosscheckBanner(cc) {
  if (!cc) return null;
  return el("div", { class: "banner " + (SEV_CLASS[cc.severity] || "na") }, el("h3", { style: "margin:0 0 4px", text: "Voice vs. spoken words" }), el("div", { text: cc.headline }), bullets(cc.explanation));
}
function transcriptPanel(rec) {
  const tr = rec.results.transcript; if (!tr) return null;
  const root = el("div", { class: "card" }, el("h2", { text: "Transcript — what is being said" }));
  if (tr.status !== "ok") { root.append(el("div", { class: "note", text: `Transcript ${tr.status}: ${tr.reason || ""} The audio analysis above is unaffected.` })); return root; }
  const tc = rec.results.text_check || {}, cc = rec.results.text_crosscheck, tv = TEXT_VERDICT[tc.verdict] || ["na", tc.verdict || "n/a"];
  const b = crosscheckBanner(cc); b && root.append(b);
  root.append(el("div", { class: "row", style: "margin:10px 0" }, el("strong", { text: "Wording check:" }), badge(tv[0], tv[1]), tc.score !== null && tc.score !== undefined ? el("span", { class: "mut small", text: `risk score ${fmt(tc.score, 2)} · ${tc.word_count} words · transcription confidence ${fmt(tc.asr_confidence, 2)}` }) : null), bullets(tc.reasons));
  root.append(el("div", { class: "seglist", role: "list" }, (cc ? cc.segments : tr.segments).map(sg => segRow(sg))));
  if ((tr.segments || []).length === 0) root.append(el("div", { class: "mut", text: "No speech was detected in the audio track." }));
  const inds = tc.indicators || [];
  if (inds.length) root.append(el("details", {}, el("summary", { text: `Matched phrases (${inds.length})` }), bullets(inds.map(i => `“${i.phrase}” — ${i.category}`))));
  root.append(el("div", { class: "note", text: `${tr.engine}. ${tr.note} The wording check is a transparent phrase-matching heuristic, not a trained or validated model, and it cannot tell whether a voice is synthetic: it only shows whether the words and the audio model's verdict look consistent. Genuine calls can contain these phrases and scammers can avoid them.` }));
  return root;
}

function renderResult(rec, { compact } = {}) {
  const a = rec.assessment, root = el("div"), cls = OVERALL_CLASS[a.overall] || "na";
  root.append(el("div", { class: "banner " + cls }, el("h2", { text: a.overall_label }),
    el("div", { text: a.headline }), el("div", { class: "mut small", text: `${rec.filename} · ${rec.kind} · analysis ${rec.id} · ${rec.created_at} · scenario ${a.scenario}${a.secondary_scenario ? "+" + a.secondary_scenario : ""}` }),
    el("div", { class: "small", text: "“Likely authentic” does not mean the media is proven authentic; scores are model outputs, not proof. " }), (rec.warnings || []).map(w => el("div", { class: "small", text: "⚠ " + w }))));
  const mods = el("div", { class: "grid g2" });
  for (const [k, v] of Object.entries(a.modalities)) mods.append(modalityCard(k[0].toUpperCase() + k.slice(1), v, rec.results[k] || rec.results.image || {}));
  root.append(mods);
  (a.conflicts || []).length && root.append(el("div", { class: "card", style: "border-color:var(--warn)" }, el("h3", { text: "⚠ Conflicting evidence" }), bullets(a.conflicts)));
  const cb = crosscheckBanner(rec.results.text_crosscheck); cb && root.append(cb);
  if (rec.results.sync) { const s = rec.results.sync; root.append(el("div", { class: "card" }, el("h3", { text: "Audio–visual speech-activity check (experimental)" }), el("div", { text: s.status === "ok" ? `Correlation ${s.best_correlation} at lag ${s.best_lag_s}s (face found in ${(s.face_detected_fraction * 100).toFixed(0)}% of samples). ${s.interpretation}` : `${s.status}: ${s.reason || ""}` }), el("div", { class: "note", text: s.note || "Not a validated lip-sync detector. Synchronization anomalies can come from ordinary editing or dubbing." }))); }
  if (rec.kind === "video") root.append(el("div", { class: "card" }, el("h2", { text: "Multimodal authenticity matrix" }), matrixTable(a.scenario, a.secondary_scenario)));
  if (rec.kind === "video") root.append(timelineSection(rec));
  const g = frameGallery(rec, null); g && root.append(g);
  const ap = audioPanel(rec); ap && root.append(ap);
  const tp = transcriptPanel(rec); tp && root.append(tp);
  root.append(explanationPanel(rec, rec.id));
  const ip = integrityPanel(rec.integrity); ip && root.append(ip);
  root.append(el("div", { class: "card" }, el("h3", { text: "Models & configuration" }), el("div", { class: "small mono", text: JSON.stringify({ models: rec.models, policy: rec.policy, timings: rec.timings }, null, 1), style: "white-space:pre-wrap" })));
  root.append(el("div", { class: "row" },
    el("a", { class: "btn", href: `/api/analyses/${rec.id}/report.pdf`, download: "", onclick: e => dl(e, `/analyses/${rec.id}/report.pdf`, `truthlens_${rec.id}.pdf`), text: "Download PDF report" }),
    el("a", { class: "btn sec", href: `/api/analyses/${rec.id}/report.json`, onclick: e => dl(e, `/analyses/${rec.id}/report.json`, `truthlens_${rec.id}.json`), text: "Download JSON" }),
    el("a", { class: "btn sec", href: `#/timeline/${rec.id}`, text: "Open in timeline" }),
    el("button", { class: "btn danger", onclick: async () => { if (!confirm("Delete this stored analysis? (Uploaded media was never stored.)")) return; try { await api(`/analyses/${rec.id}`, { method: "DELETE" }); toast("Analysis deleted"); location.hash = "#/history"; } catch (e) { toast(e.message, true); } }, text: "Delete analysis" })));
  root.append(el("p", { class: "mut small", text: rec.disclaimer }));
  return root;
}
async function dl(e, path, name) {   // authenticated download (works when an API token is required)
  e.preventDefault(); try { const h = TOKEN ? { Authorization: "Bearer " + TOKEN } : {}; const r = await fetch("/api" + path, { headers: h }); if (!r.ok) throw new Error("Download failed");
    const b = await r.blob(), u = URL.createObjectURL(b), a = el("a", { href: u, download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 5000); } catch (x) { toast(x.message, true); }
}

/* ---------- upload → job → result flow ---------- */
function analyzeWidget({ expect, label, hint }) {
  const out = el("div"), status = el("div", { "aria-live": "polite" });
  const run = async file => {
    clear(out); const bar = el("i"), msg = el("span", { text: `Uploading ${file.name}…` }), cancelBtn = el("button", { class: "btn sec", text: "Cancel", disabled: true });
    status.replaceChildren(el("div", { class: "card" }, el("div", { class: "row" }, el("span", { class: "spin" }), msg, cancelBtn), el("div", { class: "bar", style: "margin-top:8px" }, bar)));
    const form = new FormData(); form.append("file", file); form.append("expect", expect || "");
    try {
      const { job_id } = await upload("/analyze", form, f => { bar.style.width = (f * 30) + "%"; });
      cancelBtn.disabled = false; cancelBtn.onclick = () => api(`/jobs/${job_id}/cancel`, { method: "POST" });
      const j = await pollTask("jobs", job_id, j => { bar.style.width = (30 + j.progress * 70) + "%"; msg.textContent = `${file.name}: ${j.message}`; });
      clear(status);
      if (j.status === "done") { const rec = await api(`/analyses/${j.analysis_id}`); out.append(renderResult(rec)); }
      else status.append(el("div", { class: "card", style: "border-color:var(--bad)" }, el("h3", { text: j.status === "cancelled" ? "Cancelled" : "Analysis failed" }), el("p", { text: j.error || "" }),
        el("div", { class: "row" }, el("button", { class: "btn", text: "Try again", onclick: () => run(file) }))));
    } catch (e) { clear(status).append(el("div", { class: "card", style: "border-color:var(--bad)" }, el("h3", { text: "Error" }), el("p", { text: e.message }), el("button", { class: "btn", text: "Retry", onclick: () => run(file) }))); }
  };
  const root = el("div", {}, dropzone({ accept: ACCEPT[expect] || "image/*,audio/*,video/*", multiple: false, label: label, hint, onFiles: fs => run(fs[0]) }), status, out);
  root.run = run; return root;
}
async function renderAnalysisPage(container, id) {
  const rec = await api(`/analyses/${id}`); container.append(el("h1", { text: "Analysis" }), renderResult(rec));
}
