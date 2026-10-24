"use strict";
/* Live detection: microphone (+ optional webcam). Chunks are recorded in the browser, sent to /api/live/*, scored with the
   same models/policy as file analysis, and discarded server-side. Nothing here is simulated: every number shown is the
   server's response. */
(function () {
  const CHUNK_CHOICES = [4, 5, 6, 8];
  const FRAME_EVERY_MS = 3000;
  const pickMime = () => ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"].find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";

  register({ id: "live", title: "Live Detection", group: "Analyze", async render(c) {
    c.append(...H("Live Detection", "Speak into your microphone (and optionally show your webcam). Audio is scored in ~2 s windows by the CNN+Conformer, transcribed by Whisper, and the words are compared with the voice verdict in near real time."));
    const m = await api("/models");
    const insecure = !(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    const S = { sid: null, stream: null, rec: null, stopping: false, timers: [], queue: [], busy: false, frameBusy: false, frameOn: false, audioPts: [], videoPts: [], log: [], ctx: null, raf: 0, chunkN: 0 };

    const statusBox = el("div", { "aria-live": "polite" }), banner = el("div"), ccBox = el("div"), scores = el("div", { class: "grid g2" }), logBox = el("div", { class: "seglist", style: "max-height:380px", role: "log", "aria-live": "polite" }),
      textBox = el("div"), chartA = el("canvas", { role: "img", "aria-label": "Live audio chunk scores" }), chartV = el("canvas", { role: "img", "aria-label": "Live webcam frame scores" }), meterBar = el("i"),
      cam = el("video", { class: "cam", muted: true, playsinline: true, autoplay: true, hidden: true, "aria-label": "Webcam preview" }), canvasCap = document.createElement("canvas");
    cam.muted = true;
    const camChk = el("input", { type: "checkbox", id: "camchk" }), chunkSel = el("select", { id: "chunksel", "aria-label": "Chunk length" }, CHUNK_CHOICES.map(n => el("option", { value: n, text: `${n} s chunks`, selected: n === 5 })));
    const startBtn = el("button", { class: "btn", text: "● Start live detection" }), stopBtn = el("button", { class: "btn danger", text: "Stop", disabled: true }), dlBtn = el("button", { class: "btn sec", text: "Download session log (JSON)", disabled: true });

    const chip = (label, v) => el("span", { class: "pill " + (v && v.available ? "ok" : "bad"), title: (v && v.reason) || "", text: `${label}: ${v && v.available ? "ready" : "unavailable"}` });
    c.append(el("div", { class: "row" }, chip("Audio model", m.audio), chip("Speech-to-text", m.asr), chip("FA3-CLIP (webcam)", m.fa3)));
    if (insecure) c.append(el("div", { class: "note", style: "border-color:var(--bad)", text: "This browser context cannot access the microphone. Open the app via http://localhost:8000 or over HTTPS (browsers block microphone/camera on plain http to other hosts)." }));
    { const hn = audioHealthNote(m); hn && c.append(hn); }
    if (!m.audio.available) c.append(el("div", { class: "note", style: "border-color:var(--bad)", text: "Audio model unavailable: " + (m.audio.reason || "") }));
    if (!m.asr.available) c.append(el("div", { class: "note", text: "Live scoring will work but transcripts are unavailable: " + (m.asr.reason || "") }));
    c.append(el("div", { class: "card" }, el("div", { class: "row" }, startBtn, stopBtn, dlBtn,
      el("label", { class: "row small", for: "chunksel" }, "Chunk length", chunkSel),
      el("label", { class: "row small", for: "camchk" }, camChk, "Also analyze webcam frames (FA3-CLIP)" + (m.fa3.available ? "" : " — FA3 weights not installed, frames will not be scored"))),
      el("div", { class: "row small", style: "margin-top:10px" }, el("span", { class: "mut", text: "Mic level" }), el("div", { class: "meter", style: "flex:1;min-width:120px" }, meterBar)), statusBox));
    c.append(el("div", { class: "note", text: "Privacy: audio chunks and webcam frames are processed in temporary files that are deleted immediately; only scores and transcript text are held in server memory for this session. Nothing is added to the analysis history and no external service is used. Short live windows give noisier scores than a full-file analysis, and a transcript can be cut mid-word at chunk boundaries." }));

    const layout = el("div", { class: "livebox" });
    const left = el("div"), right = el("div");
    left.append(banner, ccBox, el("div", { class: "card" }, el("h3", { text: "Live transcript" }), logBox), textBox);
    right.append(cam, scores, el("div", { class: "card" }, el("h3", { text: "Audio chunk scores" }), chartA, el("div", { class: "mut small", text: "x = seconds into the session; dashed line = suspicion threshold; high = more suspicious." })), el("div", { class: "card", hidden: true, id: "vcard" }, el("h3", { text: "Webcam frame scores" }), chartV));
    layout.append(left, right); c.append(layout);
    banner.append(el("div", { class: "card empty", text: "Start a session to see a live assessment." }));

    const setStatus = (msg, spin, err) => statusBox.replaceChildren(el("div", { class: "row small", style: "margin-top:8px" }, spin ? el("span", { class: "rec" }) : null, el("span", { class: err ? "" : "mut", style: err ? "color:var(--bad)" : "", text: msg })));

    function applyRoll(j) {
      const a = j.assessment, cls = OVERALL_CLASS[a.overall] || "na";
      banner.replaceChildren(el("div", { class: "banner " + cls }, el("h2", { text: a.overall_label + " (rolling " + j.rolling_seconds + " s)" }), el("div", { text: a.headline }),
        el("div", { class: "mut small", text: `${j.audio_seconds_total} s of audio, ${j.frames_total} webcam frame(s) received · scenario ${a.scenario}` }),
        el("div", { class: "small", text: "“Likely authentic” is not proof; live scores are uncalibrated and noisy." }), bullets(a.conflicts || [])));
      const tiles = [];
      for (const [k, v] of Object.entries(a.modalities)) tiles.push(el("div", { class: "card" }, el("div", { class: "row" }, el("strong", { text: k }), badge(VERDICT_CLASS[v.verdict] || "na", VERDICT_TEXT[v.verdict] || v.verdict)),
        v.score !== null && v.score !== undefined ? el("div", { class: "score", style: `color:${scoreColor(v.score)}`, text: fmt(v.score, 3) }) : el("div", { class: "mut", text: "no score yet" }), el("div", { class: "mut small", text: (v.reasons || [])[0] || "" })));
      scores.replaceChildren(...tiles);
      const cc = a.text_crosscheck; ccBox.replaceChildren(...(cc ? [crosscheckBanner(cc)] : []));
      const tc = j.text_check;
      if (tc && tc.status === "ok") { const tv = TEXT_VERDICT[tc.verdict] || ["na", tc.verdict]; textBox.replaceChildren(el("div", { class: "card" }, el("div", { class: "row" }, el("strong", { text: "Wording check (rolling):" }), badge(tv[0], tv[1]), tc.score !== null && tc.score !== undefined ? el("span", { class: "mut small", text: `risk ${fmt(tc.score, 2)} · ${tc.word_count} words` }) : null), bullets(tc.reasons),
        (tc.indicators || []).length ? el("div", { class: "small mut", text: "Matched: " + tc.indicators.map(i => `“${i.phrase}”`).join(", ") }) : null, el("div", { class: "note", text: "Phrase-matching heuristic, not a trained model; it does not say whether the voice is synthetic." }))); }
      else if (tc && tc.reason) textBox.replaceChildren(el("div", { class: "note", text: tc.reason }));
    }
    function drawCharts() {
      lineChart(chartA, S.audioPts, { thr: POLICY_THR, xlabel: "session time (s)", h: 160 });
      if (S.videoPts.length) { $("#vcard").hidden = false; lineChart(chartV, S.videoPts, { thr: POLICY_THR, xlabel: "session time (s)", h: 160 }); }
    }
    let POLICY_THR = 0.5; try { POLICY_THR = (await api("/config")).policy.frame_suspicious_cut; } catch (e) {}

    async function sendAudio(blob, tStart) {
      const form = new FormData(); form.append("file", blob, "chunk." + (blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm"));
      const t0 = performance.now(); S.chunkN++;
      try {
        const j = await api(`/live/${S.sid}/audio`, { method: "POST", body: form });
        if (S.stopped) return;
        const ch = j.chunk; S.audioPts.push({ x: (ch.start + ch.end) / 2, y: ch.score });
        for (const sg of j.transcript.segments) { S.log.push(sg); logBox.append(segRow(sg)); }
        logBox.scrollTop = logBox.scrollHeight;
        S.last = j; applyRoll(j); drawCharts();
        setStatus(`Listening… chunk ${S.chunkN} (${fmt(ch.duration_s, 1)} s) scored in ${((performance.now() - t0) / 1000).toFixed(1)} s${ch.silent ? " — silence" : ""}${j.transcript.status === "unavailable" ? " — transcript unavailable" : ""}`, true);
        if (j.transcript.status === "unavailable" && !S.warnedT) { S.warnedT = true; toast("Transcript unavailable: " + (j.transcript.reason || ""), true); }
      } catch (e) { if (!S.stopped) { setStatus("Chunk failed: " + e.message, true, true); if (/session/i.test(e.message)) stop(); } }
    }
    async function pump() {
      if (S.busy) return; S.busy = true;
      while (S.queue.length) {
        if (S.queue.length > 3) { S.queue.splice(0, S.queue.length - 2); toast("The server cannot keep up — skipped older chunks. Try longer chunks or a smaller Whisper model.", true); }
        const it = S.queue.shift(); await sendAudio(it.blob, it.t);
      }
      S.busy = false;
    }
    function recordCycle() {
      if (S.stopping) return;
      const mime = pickMime(), chunks = [];
      const rec = new MediaRecorder(new MediaStream(S.stream.getAudioTracks()), mime ? { mimeType: mime } : undefined); S.rec = rec;
      rec.ondataavailable = e => e.data && e.data.size && chunks.push(e.data);
      rec.onstop = () => { const blob = new Blob(chunks, { type: rec.mimeType || mime || "audio/webm" }); if (blob.size > 400 && !S.stopped) { S.queue.push({ blob }); pump(); } };
      rec.start(); S.timers.push(setTimeout(() => { if (rec.state !== "inactive") rec.stop(); recordCycle(); }, Number(chunkSel.value) * 1000));
    }
    async function grabFrame() {
      if (!S.frameOn || S.frameBusy || S.stopped || !cam.videoWidth) return; S.frameBusy = true;
      try {
        const w = Math.min(640, cam.videoWidth), h = Math.round(w * cam.videoHeight / cam.videoWidth); canvasCap.width = w; canvasCap.height = h; canvasCap.getContext("2d").drawImage(cam, 0, 0, w, h);
        const blob = await new Promise(r => canvasCap.toBlob(r, "image/jpeg", 0.85)); const form = new FormData(); form.append("file", blob, "frame.jpg");
        const j = await api(`/live/${S.sid}/frame`, { method: "POST", body: form }); if (S.stopped) return;
        if (j.frame.status === "ok") S.videoPts.push({ x: (Date.now() - S.t0) / 1000, y: j.frame.score });
        else { S.frameOn = false; toast("Webcam frames not scored: " + (j.frame.reason || j.frame.status), true); }
        S.last = j; applyRoll(j); drawCharts();
      } catch (e) { setStatus("Frame failed: " + e.message, true, true); } finally { S.frameBusy = false; }
    }
    function meter() {
      if (!S.analyser) return; const d = new Uint8Array(S.analyser.fftSize); S.analyser.getByteTimeDomainData(d);
      let mx = 0; for (const v of d) mx = Math.max(mx, Math.abs(v - 128)); meterBar.style.width = Math.min(100, mx / 128 * 160) + "%"; S.raf = requestAnimationFrame(meter);
    }

    async function start() {
      if (insecure) return toast("Microphone access is not available in this context.", true);
      startBtn.disabled = true; clear(logBox); clear(textBox); clear(ccBox); S.audioPts = []; S.videoPts = []; S.log = []; S.chunkN = 0; S.stopped = false; S.stopping = false; S.warnedT = false; S.queue = []; $("#vcard").hidden = true;
      try {
        S.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: camChk.checked ? { width: { ideal: 640 }, facingMode: "user" } : false });
      } catch (e) { setStatus("Could not access the " + (camChk.checked ? "microphone/camera" : "microphone") + ": " + e.message, false, true); startBtn.disabled = false; return; }
      try { const r = await api("/live/start", { method: "POST" }); S.sid = r.session_id; }
      catch (e) { S.stream.getTracks().forEach(t => t.stop()); setStatus("Could not start a session: " + e.message, false, true); startBtn.disabled = false; return; }
      S.t0 = Date.now(); stopBtn.disabled = false; dlBtn.disabled = true; chunkSel.disabled = camChk.disabled = true;
      try { S.ctx = new (window.AudioContext || window.webkitAudioContext)(); S.analyser = S.ctx.createAnalyser(); S.ctx.createMediaStreamSource(S.stream).connect(S.analyser); meter(); } catch (e) {}
      if (camChk.checked) { cam.hidden = false; cam.srcObject = S.stream; S.frameOn = true; S.timers.push(setInterval(grabFrame, FRAME_EVERY_MS)); }
      setStatus("Listening… speak now. First result after one chunk.", true); recordCycle();
    }
    function stop() {
      if (S.stopped) return; S.stopping = true; S.stopped = true; S.timers.forEach(t => { clearTimeout(t); clearInterval(t); }); S.timers = [];
      try { if (S.rec && S.rec.state !== "inactive") S.rec.stop(); } catch (e) {}
      if (S.stream) S.stream.getTracks().forEach(t => t.stop()); cam.srcObject = null; cam.hidden = true;
      cancelAnimationFrame(S.raf); S.analyser = null; meterBar.style.width = "0"; try { S.ctx && S.ctx.close(); } catch (e) {}
      if (S.sid) api(`/live/${S.sid}`, { method: "DELETE" }).catch(() => {}); S.sid = null;
      startBtn.disabled = false; stopBtn.disabled = true; chunkSel.disabled = camChk.disabled = false; dlBtn.disabled = !S.last;
      setStatus("Session ended. Nothing was saved on the server; use “Download session log” to keep the results.", false);
    }
    startBtn.onclick = start; stopBtn.onclick = stop;
    dlBtn.onclick = () => { const blob = new Blob([JSON.stringify({ exported: new Date().toISOString(), machine_generated: true, note: "Live-session scores are uncalibrated model outputs, not proof.", final_rolling_assessment: S.last && S.last.assessment, text_check: S.last && S.last.text_check, audio_chunk_scores: S.audioPts, webcam_frame_scores: S.videoPts, transcript_segments: S.log }, null, 2)], { type: "application/json" });
      const a = el("a", { href: URL.createObjectURL(blob), download: "truthlens_live_session.json" }); document.body.append(a); a.click(); a.remove(); };
    window.addEventListener("hashchange", function once() { stop(); window.removeEventListener("hashchange", once); });
  } });
  const i = PAGES.findIndex(p => p.id === "video"); const live = PAGES.pop(); PAGES.splice(i + 1, 0, live);
})();
