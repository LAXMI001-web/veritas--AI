"use strict";
const H = (t, sub) => [el("h1", { text: t }), sub ? el("p", { class: "mut", text: sub }) : null];
const PRIVACY = "Privacy: uploaded media is processed in a temporary folder and deleted when the analysis ends. Only the analysis result (scores, small frame thumbnails, metadata) is stored locally in SQLite until you delete it or the retention period passes. Nothing is sent to external services.";
const CONF_NOTE = "Model confidence ≠ authenticity: a score says how strongly a classifier reacted to the input, not whether the media is genuine. Scores here are uncalibrated.";

/* 1 Home */
register({ id: "home", title: "Home & Dashboard", group: "Analyze", async render(c) {
  const [st, models, list] = await Promise.all([api("/stats"), api("/models"), api("/analyses?limit=8")]);
  const ready = [models.fa3, models.audio, models.asr].filter(m => m && m.available).length, total = [models.fa3, models.audio, models.asr].filter(Boolean).length;
  const chip = (dot, n, label) => el("div", { class: "chip" }, el("span", { class: "dot " + dot }), el("b", { text: n }), el("span", { class: "mut", text: label }));
  c.append(el("section", { class: "hero" },
    el("h1", {}, "Verify what you see and hear — ", el("em", { text: "with evidence" }), ", not ", el("b", { text: "guesses" }), "."),
    el("p", { text: "Multimodal deepfake intelligence: image, audio and video are analysed independently, compared under a transparent policy, and reported with honest uncertainty." }),
    el("div", { class: "row" }, el("a", { class: "btn warm", href: "#/multimodal", text: "Analyze a video" }), el("a", { class: "btn", href: "#/live", text: "Start live detection" }), el("a", { class: "btn sec", href: "#/evaluation", text: "Model evaluation" })),
    el("div", { class: "chips" }, chip(ready === total ? "ok" : ready ? "warn" : "bad", `${ready}/${total}`, "models ready"), chip("", st.total, "analyses stored"),
      chip(models.audio && models.audio.validated === false ? "warn" : "ok", models.audio && models.audio.validated === false ? "unvalidated" : "validated", "audio model"))));
  const tiles = el("div", { class: "grid g3" });
  [["image", "Image detection", "FA3-CLIP scores the image as in the original app."], ["audio", "Audio detection", "Windowed CNN scoring across the whole recording."], ["video", "Video detection", "Face frames across the timeline + the audio track."]].forEach(([k, t, d]) =>
    tiles.append(el("a", { class: "card tile " + k, href: "#/" + k }, el("div", { class: "tk" }, el("span", { text: t }), icon(k, "")), el("div", { class: "tb" }, el("p", { text: d }), el("span", { class: "btn sec", text: "Open →" })))));
  c.append(tiles);
  const quick = analyzeWidget({ expect: "", label: "Quick analysis — drop any image, audio or video", hint: "The file type is detected from its content" });
  c.append(el("h2", { text: "Quick analysis" }), quick);
  const hn = audioHealthNote(models); hn && c.append(hn);
  c.append(el("h2", { text: "Model status" }), el("div", { class: "grid g2" }, [["FA3-CLIP", models.fa3], [(models.audio && (models.audio.model || models.audio.name)) || "Audio model", models.audio]].map(([n, m]) =>
    el("div", { class: "card" }, el("div", { class: "row" }, el("h3", { text: n, style: "margin:0" }), badge(m.available ? "ok" : "bad", m.available ? "ready" : "unavailable")), el("div", { class: "mut small", text: m.reason || [m.params ? `${m.params.toLocaleString()} parameters` : null, m.input, m.class_mapping ? `class order: fake=${m.class_mapping.fake}, real=${m.class_mapping.real}` : null].filter(Boolean).join(" · ") || "ready" })))));
  c.append(el("h2", { text: "Summary of stored results" }));
  if (!st.total) c.append(el("div", { class: "card empty", text: "No analyses yet. Upload a file above to get started." }));
  else c.append(el("div", { class: "grid g3" }, Object.entries(st.by_overall).map(([k, v]) => el("div", { class: "card" }, el("div", { class: "score", text: v }), el("div", { class: "mut", text: k.replace(/_/g, " ") })))));
  c.append(el("h2", { text: "Recent analyses" }), historyTable(list.items, false));
  c.append(el("div", { class: "note", text: CONF_NOTE + " " + PRIVACY }), el("p", {}, el("a", { href: "#/evaluation", text: "Research & evaluation" }), " · ", el("a", { href: "#/settings", text: "Documentation" })));
}});
/* 2-4 single-modality */
for (const [k, t, d] of [["image", "Image Detection", "FA3-CLIP scores the whole image exactly as in the original FA3 app (resize + center crop 224)."],
  ["audio", "Audio Detection", "CNN+Conformer scores overlapping ≈2.05 s windows across the whole recording."],
  ["video", "Video Detection", "Frames are sampled and scored by FA3-CLIP; the audio track (if any) is scored separately by CNN+Conformer."]])
  register({ id: k, title: t, group: "Analyze", async render(c) { if (k !== "image") { try { const n = audioHealthNote(await api("/models")); n && c.append(n); } catch (e) {} } c.append(...H(t, d), analyzeWidget({ expect: k, label: `Upload ${k}`, hint: "Max size per server settings" }), el("div", { class: "note", text: CONF_NOTE })); } });
/* 5 Multimodal */
register({ id: "multimodal", title: "Multimodal Analysis", group: "Analyze", render(c) {
  c.append(...H("Multimodal Analysis", "Upload a video: audio and video are analysed independently, then compared under a transparent, configurable policy."),
    el("div", { class: "card" }, el("h3", { text: "Authenticity matrix" }), matrixTable(null, null)), analyzeWidget({ expect: "video", label: "Upload a video (with or without audio)" }), el("div", { class: "note", text: CONF_NOTE })); } });
/* 6 Cross-modal conflict */
async function pickAnalysis(c, id, build, kinds) {
  const list = (await api("/analyses?limit=100")).items.filter(i => !kinds || kinds.includes(i.kind)); const holder = el("div");
  const sel = el("select", { "aria-label": "Choose analysis" }, el("option", { value: "", text: list.length ? "— choose an analysis —" : "No analyses yet" }), list.map(i => el("option", { value: i.id, text: `${i.filename} · ${i.overall.replace(/_/g, " ")} · ${i.created_at.slice(0, 16)}` })));
  const show = async v => { clear(holder); if (v) { try { build(holder, await api(`/analyses/${v}`)); } catch (e) { holder.append(el("div", { class: "note", text: e.message })); } } };
  sel.onchange = () => { history.replaceState(null, "", `#/${PAGES.find(p => p.build === build)?.id || ""}/${sel.value}`); show(sel.value); };
  c.append(el("div", { class: "row" }, sel), holder); if (id && list.some(i => i.id === id)) { sel.value = id; show(id); } else if (!id && list.length) { sel.value = list[0].id; show(list[0].id); }
  if (!list.length) holder.append(el("div", { class: "card empty" }, "Analyse a video first: ", el("a", { href: "#/multimodal", text: "Multimodal Analysis" })));
}
const buildConflict = (h, rec) => {
  const a = rec.assessment; h.append(el("div", { class: "banner " + (OVERALL_CLASS[a.overall] || "na") }, el("h2", { text: a.overall_label }), el("div", { text: a.headline })));
  h.append(el("div", { class: "grid g2" }, Object.entries(a.modalities).map(([k, v]) => modalityCard(k[0].toUpperCase() + k.slice(1), v, rec.results[k] || {}))));
  h.append(el("div", { class: "card" }, el("h3", { text: "Why the modalities agree or differ" }), bullets(rec.explanation.model_interpretation), el("h3", { text: "Conflicts detected" }), bullets(a.conflicts.length ? a.conflicts : ["No internal or cross-modal conflict flagged by the policy."]),
    el("h3", { text: "Possible explanations (none established)" }), bullets(rec.explanation.possible_explanations)), el("div", { class: "card" }, el("h3", { text: "Where the evidence is" }), bullets(rec.explanation.observed_evidence)),
    el("div", { class: "card" }, el("h3", { text: "All scenarios" }), matrixTable(a.scenario, a.secondary_scenario)), el("div", { class: "note", text: "No claim such as “voice cloning” or “face swap” is made: the models cannot identify manipulation techniques." }));
};
register({ id: "conflicts", title: "Cross-Modal Conflicts", group: "Analyze", build: buildConflict, async render(c, [id]) { c.append(...H("Cross-Modal Conflict Analysis", "Explains agreement and disagreement between audio and video findings using the stored model outputs.")); await pickAnalysis(c, id, buildConflict, ["video"]); } });
/* 7 Timeline */
const buildTimeline = (h, rec) => { h.append(rec.kind === "video" || rec.kind === "audio" ? timelineSection(rec) : el("div", { class: "note", text: "Timeline applies to video and audio." })); const g = frameGallery(rec, null); g && h.append(g); const a = audioPanel(rec); a && h.append(a); };
register({ id: "timeline", title: "Evidence Timeline", group: "Analyze", build: buildTimeline, async render(c, [id]) { c.append(...H("Evidence Timeline", "Synchronized frame scores, audio waveform, window scores and suspicious intervals.")); await pickAnalysis(c, id, buildTimeline, ["video", "audio"]); } });

/* 8 Comparison */
register({ id: "compare", title: "Media Comparison", group: "Forensics", render(c) {
  c.append(...H("Media Comparison Lab", "Compare two files of the same kind, and test how scores react to benign transformations."));
  const out = el("div"), fa = { f: null }, fb = { f: null };
  const mk = (slot, name) => { const lab = el("div", { class: "mut small", text: "No file" }); return el("div", { class: "card" }, el("h3", { text: name }), dropzone({ accept: "image/*,audio/*,video/*", multiple: false, label: `Choose file ${name}`, onFiles: f => { slot.f = f[0]; lab.textContent = f[0].name; } }), lab); };
  const go = el("button", { class: "btn", text: "Compare", onclick: async () => {
    if (!fa.f || !fb.f) return toast("Choose both files", true); const form = new FormData(); form.append("file_a", fa.f); form.append("file_b", fb.f); clear(out).append(el("span", { class: "spin" }), " working…");
    try { const { task_id } = await upload("/compare", form); const t = await pollTask("tasks", task_id, t => out.firstChild && (out.lastChild.textContent = ` ${t.message || "working"} ${(t.progress * 100) | 0}%`));
      clear(out); t.status === "done" ? out.append(renderComparison(t.result)) : out.append(el("div", { class: "card", style: "border-color:var(--bad)", text: t.error })); } catch (e) { clear(out).append(el("div", { class: "card", text: e.message })); } } });
  c.append(el("div", { class: "grid g2" }, mk(fa, "A"), mk(fb, "B")), el("div", { class: "row" }, go), out);
  const rob = el("div"), rf = { f: null }, rl = el("div", { class: "mut small", text: "No file" });
  c.append(el("h2", { text: "Robustness test (benign transformations)" }), el("div", { class: "card" }, el("p", { class: "mut", text: "Applies resizing/JPEG (image), resampling/MP3/volume (audio) or re-encoding/downscaling (video) to a copy and compares scores. Your original is never modified." }),
    dropzone({ accept: "image/*,audio/*,video/*", multiple: false, label: "Choose a file to stress-test", onFiles: f => { rf.f = f[0]; rl.textContent = f[0].name; } }), rl,
    el("button", { class: "btn", style: "margin-top:8px", text: "Run robustness test", onclick: async () => { if (!rf.f) return toast("Choose a file", true); const form = new FormData(); form.append("file", rf.f); clear(rob).append(el("span", { class: "spin" }), " working…");
      try { const { task_id } = await upload("/robustness", form); const t = await pollTask("tasks", task_id); clear(rob); t.status === "done" ? rob.append(renderRobust(t.result)) : rob.append(el("div", { class: "card", text: t.error })); } catch (e) { clear(rob).append(el("div", { text: e.message })); } } })), rob);
}});
function renderRobust(r) {
  return el("div", { class: "card" }, el("h3", { text: `Stability: ${r.stability || "n/a"}` }), el("div", { text: `Original score ${fmt(r.original.score, 3)} (${r.original.verdict})` }),
    el("div", { class: "tblwrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["Transformation", "Score", "Δ vs original", "Verdict"].map(h => el("th", { text: h })))),
      el("tbody", {}, r.transforms.map(t => el("tr", {}, el("td", { text: t.label }), el("td", { text: t.error ? t.error : fmt(t.score, 3) }), el("td", { text: fmt(t.delta, 3) }), el("td", { text: t.verdict || "" })))))),
    r.verdict_changes.length ? el("div", { class: "note", text: "Verdict changed under: " + r.verdict_changes.join(", ") }) : null, el("div", { class: "note", text: r.note }));
}
function renderComparison(r) {
  const root = el("div"), lab = k => `${r[k].filename} → ${r[k].overall_label}`;
  root.append(el("div", { class: "grid g2" }, [["a", "A"], ["b", "B"]].map(([k, n]) => el("div", { class: "card" }, el("h3", { text: "File " + n }), el("div", { text: lab(k) }), el("div", { class: "mut small", text: `visual ${fmt(r[k].video_score, 3)} · audio ${fmt(r[k].audio_score, 3)}` }),
    el("a", { href: `#/analysis/${r[k + "_id"] || r[k].id}`, text: "Open full analysis" })))));
  root.append(el("div", { class: "card" }, el("h3", { text: "Score change (B − A)" }), el("div", { text: `visual: ${fmt(r.score_delta.visual, 3)} · audio: ${fmt(r.score_delta.audio, 3)}` })));
  if (r.kind === "image") { root.append(el("div", { class: "grid g2" }, [["thumb_a", "A"], ["thumb_b", "B"]].map(([k, n]) => el("div", { class: "card" }, el("h3", { text: n }), r[k] ? el("img", { class: "full", src: "data:image/jpeg;base64," + r[k], alt: "Image " + n }) : el("div", { class: "mut", text: "no preview" })))));
    if (r.image_difference) root.append(el("div", { class: "card" }, el("h3", { text: "Difference map (B resized to A if needed)" }), el("div", { class: "mut small", text: `mean abs diff ${r.image_difference.mean_abs_diff}, max ${r.image_difference.max_abs_diff}${r.image_difference.resized_b_to_a ? " · B was resized" : ""}` }), el("img", { class: "full", src: "data:image/png;base64," + r.image_difference.diff_png, alt: "Pixel difference heat map" }))); }
  if (r.kind === "audio") root.append(el("div", { class: "grid g2" }, [["a", "A"], ["b", "B"]].map(([k, n]) => el("div", { class: "card" }, el("h3", { text: "Spectrogram " + n }), el("img", { class: "full", src: "data:image/png;base64," + r["spectrogram_" + k], alt: "Spectrogram " + n })))));
  if (r.kind === "video" || r.kind === "audio") { const c = el("canvas", { role: "img", "aria-label": "Score comparison over time" }); const A = r.kind === "video" ? r.frames_a : r.windows_a.map(w => [w.start, w.score]), B = r.kind === "video" ? r.frames_b : r.windows_b.map(w => [w.start, w.score]);
    requestAnimationFrame(() => { lineChart(c, A.map(([x, y]) => ({ x: x ?? 0, y })), { xlabel: "time (s) — A (colored dots)" }); const g = c.getContext("2d"), dpr = devicePixelRatio || 1, w = c.clientWidth; g.save(); g.scale(dpr, dpr); const mx = Math.max(...A.map(p => p[0] ?? 0), ...B.map(p => p[0] ?? 0), 1); g.strokeStyle = cssv("--text"); g.setLineDash([3, 3]); g.beginPath(); B.forEach(([x, y], i) => { const X = 38 + (x ?? 0) / mx * (w - 46), Y = 8 + (1 - (y ?? 0)) * 148; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke(); g.restore(); });
    root.append(el("div", { class: "card" }, el("h3", { text: "Scores over time: A (solid) vs B (dashed white)" }), c)); }
  if (r.metadata_diff.length) root.append(el("div", { class: "card" }, el("h3", { text: "Metadata differences" }), el("div", { class: "tblwrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["field", "A", "B"].map(h => el("th", { text: h })))), el("tbody", {}, r.metadata_diff.map(d => el("tr", {}, el("td", { text: d.field }), el("td", { text: String(d.a ?? "—"), class: "mono" }), el("td", { text: String(d.b ?? "—"), class: "mono" }))))))));
  root.append(el("div", { class: "note", text: r.note + " Synchronized playback is unavailable because uploaded media is not retained." })); return root;
}
/* 9 Forensic inspection */
register({ id: "inspect", title: "Forensic Inspection", group: "Forensics", render(c) {
  const out = el("div"); c.append(...H("Forensic Inspection", "Metadata, codec, duration, repeated-frame and discontinuity indicators, and provenance markers — no deepfake classification."),
    dropzone({ accept: "image/*,audio/*,video/*", multiple: false, label: "Upload a file to inspect", onFiles: async ([f]) => { const form = new FormData(); form.append("file", f); clear(out).append(el("span", { class: "spin" }), " inspecting…");
      try { const { task_id } = await upload("/inspect", form); const t = await pollTask("tasks", task_id); clear(out); out.append(t.status === "done" ? integrityPanel(t.result) : el("div", { class: "card", text: t.error })); } catch (e) { clear(out).append(el("div", { class: "card", text: e.message })); } } }), out,
    el("div", { class: "note", text: "Missing metadata or compression artefacts are never treated as evidence of fakery. Provenance is reported as unverified because no signature verification is implemented." })); } });
/* 10 Batch */
register({ id: "batch", title: "Batch Analysis", group: "Forensics", render(c) {
  const out = el("div"); c.append(...H("Batch Analysis", "Upload up to 50 files. Each is queued, processed with per-file progress, and stored in history."),
    dropzone({ accept: "image/*,audio/*,video/*", multiple: true, label: "Drop multiple files", onFiles: async fs => { const form = new FormData(); fs.forEach(f => form.append("files", f)); clear(out).append(el("span", { class: "spin" }), " uploading…");
      try { const { batch_id } = await upload("/batch", form); trackBatch(out, batch_id); } catch (e) { clear(out).append(el("div", { class: "card", text: e.message })); } } }), out); } });
async function trackBatch(out, bid) {
  for (;;) { const b = await api(`/batches/${bid}`); clear(out);
    out.append(el("div", { class: "card" }, el("h3", { text: `Batch ${bid}` }), el("div", { class: "row" }, Object.entries(b.counts).map(([k, v]) => badge(k === "done" ? "ok" : k === "failed" ? "bad" : "na", `${k}: ${v}`)), Object.entries(b.summary).map(([k, v]) => badge(OVERALL_CLASS[k] || "na", `${k.replace(/_/g, " ")}: ${v}`))),
      el("div", { class: "tblwrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["File", "Status", "Progress", ""].map(h => el("th", { text: h })))), el("tbody", {}, b.jobs.map(j => el("tr", {},
        el("td", { text: j.filename }), el("td", { text: j.status + (j.error ? ": " + j.error : "") }), el("td", {}, el("div", { class: "bar" }, el("i", { style: `width:${(j.progress * 100) | 0}%` }))),
        el("td", {}, j.analysis_id ? el("a", { href: "#/analysis/" + j.analysis_id, text: "Open" }) : null, " ", ["queued", "running"].includes(j.status) ? el("button", { class: "btn sec", text: "Cancel", onclick: () => api(`/jobs/${j.id}/cancel`, { method: "POST" }) }) : null,
          j.status === "failed" ? el("button", { class: "btn sec", text: "Retry", onclick: async () => { try { await api(`/jobs/${j.id}/retry`, { method: "POST" }); } catch (e) { toast(e.message, true); } } }) : null)))))),
      b.finished ? el("a", { class: "btn", href: `/api/batches/${bid}/report`, onclick: e => dl(e, `/batches/${bid}/report`, `batch_${bid}.json`), text: "Download batch report", style: "margin-top:10px" }) : null));
    if (b.finished || !document.body.contains(out)) return; await sleep(1000); } }
/* 11 Reports & 12 History */
function historyTable(items, actions) {
  if (!items.length) return el("div", { class: "card empty", text: "Nothing here yet." });
  return el("div", { class: "card tblwrap" }, el("table", {}, el("thead", {}, el("tr", {}, ["File", "Kind", "Outcome", "Scenario", "When", actions ? "Actions" : ""].map(h => el("th", { text: h })))),
    el("tbody", {}, items.map(i => el("tr", {}, el("td", {}, el("a", { href: "#/analysis/" + i.id, text: i.filename })), el("td", { text: i.kind }), el("td", {}, badge(OVERALL_CLASS[i.overall] || "na", i.overall.replace(/_/g, " "))),
      el("td", { text: i.scenario || "" }), el("td", { text: i.created_at.replace("T", " ").slice(0, 16) }),
      actions ? el("td", {}, el("div", { class: "row" }, el("a", { href: `/api/analyses/${i.id}/report.pdf`, onclick: e => dl(e, `/analyses/${i.id}/report.pdf`, `truthlens_${i.id}.pdf`), text: "PDF" }), el("a", { href: `/api/analyses/${i.id}/report.json`, onclick: e => dl(e, `/analyses/${i.id}/report.json`, `truthlens_${i.id}.json`), text: "JSON" }),
        el("button", { class: "btn danger", text: "Delete", onclick: async ev => { if (!confirm("Delete this stored analysis?")) return; try { await api(`/analyses/${i.id}`, { method: "DELETE" }); ev.target.closest("tr").remove(); toast("Deleted"); } catch (e) { toast(e.message, true); } } }))) : null)))));
}
function listPage(id, title, sub, withFilters) {
  register({ id, title, group: "Records", async render(c) {
    c.append(...H(title, sub)); const holder = el("div"), q = el("input", { type: "search", placeholder: "Search filename, analysis ID or hash", "aria-label": "Search" }),
      kind = el("select", { "aria-label": "Kind" }, ["", "image", "audio", "video"].map(v => el("option", { value: v, text: v || "all kinds" }))),
      ov = el("select", { "aria-label": "Outcome" }, ["", "likely_authentic", "suspected_manipulation", "mixed_authenticity", "inconclusive", "analysis_incomplete"].map(v => el("option", { value: v, text: v ? v.replace(/_/g, " ") : "all outcomes" })));
    const load = async () => { const p = new URLSearchParams({ q: q.value, kind: kind.value, overall: ov.value, limit: 200 }); const r = await api("/analyses?" + p); clear(holder).append(historyTable(r.items, true)); };
    let tm; q.oninput = () => { clearTimeout(tm); tm = setTimeout(load, 250); }; kind.onchange = ov.onchange = load; c.append(el("div", { class: "row" }, q, kind, ov), holder, el("div", { class: "note", text: PRIVACY })); await load(); } });
}
listPage("reports", "Forensic Reports", "Download a PDF or JSON report generated from each stored analysis.");
listPage("history", "Analysis History", "Search, reopen or delete stored results. Deleting removes the stored result; uploaded media is never retained.");
/* 13 Evaluation */
function metricsBlock(name, m) {
  if (!m || !m.available) return el("div", { class: "card" }, el("h3", { text: name }), el("div", { class: "mut", text: m?.reason || "Unavailable" }));
  const cf = m.confusion, cv = el("canvas", { role: "img", "aria-label": name + " ROC curve" }); requestAnimationFrame(() => m.roc.length && rocChart(cv, m.roc));
  return el("div", { class: "card" }, el("h3", { text: name }), el("div", { class: "grid g3" }, ["accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "coverage"].map(k => el("div", {}, el("div", { class: "mut small", text: k.toUpperCase().replace("_", "-") }), el("div", { class: "score", style: "font-size:1.3rem", text: fmt(m[k], 3) })))),
    el("div", { class: "small mut", text: `n=${m.n} (unscored ${m.n_failed_or_unscored}) · threshold ${m.threshold}` }), el("div", { class: "grid g2" }, el("table", {}, el("thead", {}, el("tr", {}, el("th"), el("th", { text: "pred fake" }), el("th", { text: "pred real" }))), el("tbody", {}, el("tr", {}, el("th", { text: "actual fake" }), el("td", { text: cf.tp }), el("td", { text: cf.fn })), el("tr", {}, el("th", { text: "actual real" }), el("td", { text: cf.fp }), el("td", { text: cf.tn })))), cv));
}
register({ id: "evaluation", title: "Model Evaluation", group: "Research", async render(c) {
  c.append(...H("Model Evaluation & Research Dashboard", "Metrics come only from labeled test data you provide. None are shown without it."));
  const s = await api("/evaluation"), holder = el("div");
  c.append(el("div", { class: "card" }, el("h3", { text: "Dataset on the server" }), el("pre", { class: "mono", style: "margin:0", text: JSON.stringify(s.dataset_counts, null, 1) }), el("div", { class: "note", text: "Required layout: " + s.requirements }),
    el("button", { class: "btn", style: "margin-top:8px", text: "Run evaluation", onclick: async () => { clear(holder).append(el("span", { class: "spin" }), " evaluating…"); try { const { task_id } = await api("/evaluation/run", { method: "POST" }); const t = await pollTask("tasks", task_id, t => holder.lastChild && (holder.lastChild.textContent = ` ${(t.progress * 100) | 0}% ${t.message}`)); clear(holder); t.status === "done" ? showEval(holder, t.result) : holder.append(el("div", { text: t.error })); } catch (e) { clear(holder).append(el("div", { text: e.message })); } } })), holder);
  if (s.latest) showEval(holder, s.latest);
  else holder.append(el("div", { class: "card empty", text: "No evaluation has been run. Latency, resource use and accuracy metrics are marked unavailable until labeled data is provided." }));
}});
function showEval(h, r) {
  clear(h); h.append(el("div", { class: "note", text: `Run ${r.created} · status: ${r.status}. ` + (r.notes || []).join(" ") }));
  if (r.status !== "ok") return;
  for (const [mod, m] of Object.entries(r.modalities)) { h.append(el("h2", { text: mod })); if (!m.available) { h.append(el("div", { class: "card mut", text: m.reason })); continue; }
    h.append(metricsBlock(mod + " — primary", m.primary), el("div", { class: "small mut", text: `latency mean ${m.latency_s.mean}s, p95 ${m.latency_s.p95}s` }));
    if (m.frame_aggregation) { h.append(el("h3", { text: "Video frame-aggregation comparison" }), el("div", { class: "grid g2" }, Object.entries(m.frame_aggregation).map(([k, v]) => metricsBlock(k, v)))); }
    if (m.ablation) { h.append(el("h3", { text: "Ablation (same test split)" })); m.ablation.available === false ? h.append(el("div", { class: "card mut", text: m.ablation.reason })) : h.append(el("div", { class: "grid g2" }, Object.entries(m.ablation).filter(([k, v]) => v && typeof v === "object").map(([k, v]) => metricsBlock(k.replace(/_/g, " "), v)))); } }
}
/* 14 Settings & docs */
register({ id: "settings", title: "Settings & Documentation", group: "Research", async render(c) {
  const [cfg, models] = await Promise.all([api("/config"), api("/models")]); c.append(...H("Settings & Documentation"));
  const form = el("form", { class: "card" }, el("h3", { text: "Decision policy (applies to new analyses)" })), inputs = {};
  for (const [k, v] of Object.entries(cfg.policy)) { const inp = typeof v === "boolean" ? el("input", { type: "checkbox", checked: v }) : (k.endsWith("aggregation")) ? el("select", {}, ["mean", "median", "fraction", "topk"].map(o => el("option", { value: o, selected: o === v, text: o }))) : k === "video_face_crop" ? el("select", {}, ["auto", "off"].map(o => el("option", { value: o, selected: o === v, text: o }))) : el("input", { type: "number", step: "any", value: v });
    inputs[k] = inp; form.append(el("label", { class: "row", style: "justify-content:space-between;margin:4px 0" }, el("span", { text: k }), inp)); }
  form.append(el("button", { class: "btn", type: "submit", text: "Save policy" }), el("div", { class: "note", style: "margin-top:8px", text: "Defaults are conventional placeholders and have NOT been calibrated on labeled data. Choose thresholds and aggregation on a validation split (see Model Evaluation)." }));
  form.onsubmit = async e => { e.preventDefault(); const body = {}; for (const [k, i] of Object.entries(inputs)) body[k] = i.type === "checkbox" ? i.checked : i.tagName === "SELECT" || i.type === "text" ? i.value : Number(i.value);
    try { await api("/policy", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); toast("Policy saved"); } catch (er) { toast(er.message, true); } };
  c.append(form, el("div", { class: "card" }, el("h3", { text: "Privacy & storage" }), bullets([`Uploaded media stored: ${cfg.privacy.uploaded_media_stored ? "yes" : "no — deleted after each analysis"}`, `Results stored: yes, in ${cfg.privacy.stored_where}; retention ${cfg.privacy.retention_days} days`, `Frame thumbnails kept in results: ${cfg.privacy.store_thumbnails ? "yes (disable with STORE_THUMBNAILS=false)" : "no"}`, "No external services or LLM APIs are called; the analyst assistant is rule-based.", "No face/speaker identification is performed; Haar face detection is used only to crop/locate faces."]),
    el("p", { class: "small mut", text: `Limits: ${cfg.limits.max_upload_mb} MB, ${cfg.limits.max_media_seconds} s, ${cfg.limits.rate_limit_per_min} POST requests/min/IP.` })),
    el("div", { class: "card" }, el("h3", { text: "Models" }), el("pre", { class: "mono", style: "white-space:pre-wrap", text: JSON.stringify(models, null, 2) })),
    el("div", { class: "card" }, el("h3", { text: "How to read a result" }), bullets(["Scores are oriented so that higher = more suspicious, and are uncalibrated classifier outputs (not probabilities of being fake).", "“Likely authentic” never means proven authentic; “Suspected manipulation” never means proven fake.", "FA3-CLIP judges frames independently; the audio model judges ≈2 s windows. Neither models time, identity or manipulation technique.", "Heuristic integrity indicators and the experimental sync check are review pointers only.", "Model weights were trained on datasets whose coverage of your media is unknown; unseen manipulation types may be missed or over-flagged."])));
}});
/* boot */
window.addEventListener("DOMContentLoaded", async () => { buildNav(); await refreshPills(); route(); });
