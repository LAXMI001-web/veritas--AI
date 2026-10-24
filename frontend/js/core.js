"use strict";
/* ---------- DOM helpers (all dynamic text goes through textContent: never innerHTML with data) ---------- */
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const $ = (s, r = document) => r.querySelector(s);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(d);
const clear = n => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
function toast(msg, err) { const t = el("div", { class: "toast" + (err ? " err" : ""), text: msg }); $("#toasts").append(t); setTimeout(() => t.remove(), 6000); }
function tip(label, text) { return el("span", { class: "tip", tabindex: 0, title: text, "aria-label": `${label}: ${text}` }, label); }
const TIPS = {
  score: "Model output oriented so that higher = more suspicious. It is NOT a probability that the media is fake.",
  uncal: "Uncalibrated: the number has not been mapped to real-world likelihoods using validation data.",
  window: "A ~2.05 s slice of audio, the input size of the CNN+Conformer.",
  fft: "High-frequency FFT map: an input representation used by FA3-CLIP, not proof of manipulation."
};

/* ---------- API ---------- */
let TOKEN = ""; try { TOKEN = sessionStorage.getItem("veritas_token") || ""; } catch (e) {}
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {}); if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
  const r = await fetch("/api" + path, Object.assign({}, opts, { headers }));
  if (r.status === 401) { const t = prompt("This server requires an API token:"); if (t) { TOKEN = t; try { sessionStorage.setItem("veritas_token", t); } catch (e) {} return api(path, opts); } }
  let data = null; try { data = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((data && (data.message || data.detail)) || `Request failed (${r.status})`);
  return data;
}
function upload(path, form, onProgress) {   // XHR so we can show upload progress
  return new Promise((res, rej) => {
    const x = new XMLHttpRequest(); x.open("POST", "/api" + path);
    if (TOKEN) x.setRequestHeader("Authorization", "Bearer " + TOKEN);
    x.upload.onprogress = e => e.lengthComputable && onProgress && onProgress(e.loaded / e.total);
    x.onload = () => { let d = null; try { d = JSON.parse(x.responseText); } catch (e) {}
      x.status >= 200 && x.status < 300 ? res(d) : rej(new Error((d && (d.message || d.detail)) || `Upload failed (${x.status})`)); };
    x.onerror = () => rej(new Error("Network error during upload")); x.send(form);
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function pollTask(kind, id, onProg) {
  for (;;) {
    const j = await api(`/${kind}/${id}`); onProg && onProg(j);
    if (["done", "failed", "cancelled"].includes(j.status)) return j; await sleep(600);
  }
}

/* ---------- upload widget ---------- */
function dropzone({ accept, multiple, label, onFiles, hint }) {
  const input = el("input", { type: "file", accept, multiple, hidden: true, "aria-hidden": "true" });
  const z = el("div", { class: "drop", role: "button", tabindex: 0, "aria-label": label },
    el("div", { text: label, style: "font-weight:600" }), el("div", { class: "mut small", text: hint || "Drag & drop, or click / press Enter to browse" }), input);
  const pick = fs => { fs = [...fs]; if (fs.length) onFiles(multiple ? fs : [fs[0]]); };
  z.addEventListener("click", () => input.click());
  z.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { pick(input.files); input.value = ""; });
  z.addEventListener("dragover", e => { e.preventDefault(); z.classList.add("over"); });
  z.addEventListener("dragleave", () => z.classList.remove("over"));
  z.addEventListener("drop", e => { e.preventDefault(); z.classList.remove("over"); pick(e.dataTransfer.files); });
  return z;
}
const ACCEPT = { image: "image/*", audio: "audio/*,.wav,.mp3,.flac,.ogg,.m4a", video: "video/*,.mp4,.mov,.mkv,.webm,.avi" };
const MAXMB_DEFAULT = 200;

/* ---------- charts (canvas, real data only) ---------- */
function setup(c, h) { const dpr = window.devicePixelRatio || 1, w = c.clientWidth || 600; c.width = w * dpr; c.height = h * dpr; c.style.height = h + "px";
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h); return { g, w, h }; }
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function scoreColor(s) { return s === null || s === undefined ? cssv("--na") : s >= 0.6 ? cssv("--bad") : s <= 0.4 ? cssv("--ok") : cssv("--warn"); }
function lineChart(canvas, pts, { thr = 0.5, xlabel = "", xmax, h = 180, onPick } = {}) {
  const { g, w } = setup(canvas, h), L = 38, R = 8, T = 8, B = 24; const xs = pts.map(p => p.x); const x0 = Math.min(...xs, 0), x1 = xmax ?? Math.max(...xs, 1);
  const X = x => L + (x - x0) / ((x1 - x0) || 1) * (w - L - R), Y = y => T + (1 - y) * (h - T - B);
  g.strokeStyle = cssv("--line"); g.fillStyle = cssv("--mut"); g.font = "11px system-ui"; g.lineWidth = 1;
  for (const v of [0, .25, .5, .75, 1]) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(v.toFixed(2), 2, Y(v) + 4); }
  g.setLineDash([5, 4]); g.strokeStyle = cssv("--bad"); g.beginPath(); g.moveTo(L, Y(thr)); g.lineTo(w - R, Y(thr)); g.stroke(); g.setLineDash([]);
  g.fillStyle = cssv("--mut"); g.fillText(xlabel, w / 2 - 20, h - 4);
  const ok = pts.filter(p => p.y !== null && p.y !== undefined);
  g.strokeStyle = cssv("--accent"); g.lineWidth = 1.5; g.beginPath(); ok.forEach((p, i) => i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))); g.stroke();
  ok.forEach(p => { g.fillStyle = scoreColor(p.y); g.beginPath(); g.arc(X(p.x), Y(p.y), 3.5, 0, 7); g.fill(); });
  if (onPick) canvas.onclick = e => { const r = canvas.getBoundingClientRect(), x = (e.clientX - r.left - L) / (r.width - L - R) * (x1 - x0) + x0; onPick(x); };
}
function rocChart(canvas, pts) {
  const { g, w, h } = setup(canvas, 220), S = Math.min(w, 220) - 30; g.strokeStyle = cssv("--line"); g.strokeRect(28, 8, S, S);
  g.setLineDash([4, 4]); g.beginPath(); g.moveTo(28, 8 + S); g.lineTo(28 + S, 8); g.stroke(); g.setLineDash([]);
  g.strokeStyle = cssv("--accent"); g.lineWidth = 2; g.beginPath(); pts.forEach(([f, t], i) => { const x = 28 + f * S, y = 8 + (1 - t) * S; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  g.fillStyle = cssv("--mut"); g.font = "11px system-ui"; g.fillText("FPR →", 28 + S / 2 - 12, h - 2); g.fillText("TPR", 0, 20);
}

/* ---------- router ---------- */
const PAGES = [];   // {id,title,group,render(container,params)}
function register(p) { PAGES.push(p); }
const ICONS = {   // feather-style 24px stroke icons (decorative)
  home: "M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10", image: "M4 5h16v14H4zM8 10a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM4 17l5-5 4 4 3-3 4 4",
  audio: "M12 3v18M8 8v8M4 11v2M16 7v10M20 10v4", video: "M3 6h12v12H3zM15 10l6-3v10l-6-3",
  multimodal: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5", live: "M12 12m-2 0a2 2 0 104 0 2 2 0 10-4 0M7 7a7 7 0 000 10M17 7a7 7 0 010 10",
  conflicts: "M12 3l10 18H2zM12 10v5M12 18h.01", timeline: "M3 12h4l3-7 4 14 3-7h4", compare: "M8 3H3v18h5M16 3h5v18h-5M12 4v16",
  inspect: "M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3", batch: "M4 6h16M4 12h16M4 18h10", evaluation: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  reports: "M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V8zM14 3v5h5M8 13h8M8 17h5", history: "M3 12a9 9 0 109-9 9 9 0 00-6.4 2.6L3 8M3 3v5h5M12 7v5l3 2",
  settings: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12a7 7 0 00-.1-1.2l2-1.5-2-3.4-2.4 1a7 7 0 00-2-1.2L14 3h-4l-.5 2.7a7 7 0 00-2 1.2l-2.4-1-2 3.4 2 1.5A7 7 0 005 12c0 .4 0 .8.1 1.2l-2 1.5 2 3.4 2.4-1a7 7 0 002 1.2L10 21h4l.5-2.7a7 7 0 002-1.2l2.4 1 2-3.4-2-1.5c.1-.4.1-.8.1-1.2z"
};
function icon(id, cls = "ico") {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("class", cls); s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path"); p.setAttribute("d", ICONS[id] || ICONS.home); s.append(p); return s;
}
function buildNav() {
  const nav = clear($("#nav")); let grp = null;
  for (const p of PAGES) { if (p.group !== grp) { grp = p.group; nav.append(el("div", { class: "grp", text: grp })); }
    nav.append(el("a", { href: "#/" + p.id, "data-id": p.id }, icon(p.id), p.title)); }
}
async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean), id = parts[0] || "home", rest = parts.slice(1);
  const page = PAGES.find(p => p.id === id) || (id === "analysis" ? { id: "analysis", title: "Analysis", render: (c, a) => renderAnalysisPage(c, a[0]) } : PAGES[0]);
  document.querySelectorAll("nav a").forEach(a => a.removeAttribute("aria-current"));
  const cur = $(`nav a[data-id="${page.id === "analysis" ? "history" : page.id}"]`); if (cur) cur.setAttribute("aria-current", "page");
  const main = clear($("#main")); $("#nav").classList.remove("open"); $("#menu-btn").setAttribute("aria-expanded", "false");
  try { await page.render(main, rest); } catch (e) { main.append(el("div", { class: "card" }, el("h2", { text: "Something went wrong" }), el("p", { text: e.message }))); }
  main.focus({ preventScroll: true }); window.scrollTo(0, 0);
}
async function refreshPills() {
  try { const m = await api("/models"), box = clear($("#model-pills"));
    for (const [k, v] of Object.entries({ "FA3-CLIP (image/video)": m.fa3, [`${(m.audio && (m.audio.model || m.audio.name)) || "Audio model"} (audio)`]: m.audio, "Speech-to-text": m.asr }).filter(([, v]) => v))
      { const sus = v.available && v.health && v.health.status === "suspect";
        box.append(el("span", { class: "pill " + (!v.available ? "bad" : sus ? "warn" : "ok"), title: v.reason || (sus ? v.health.message : ""), text: `${k}: ${!v.available ? "unavailable" : sus ? "ready — output unreliable" : "ready"}` })); }
  } catch (e) {}
}
window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  $("#menu-btn").addEventListener("click", () => { const o = $("#nav").classList.toggle("open"); $("#menu-btn").setAttribute("aria-expanded", o); });
});
