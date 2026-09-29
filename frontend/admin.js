const $ = (s) => document.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
};
let token = store.get("token");

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (!(opts.body instanceof FormData)) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = "Bearer " + token;
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  let data = null; try { data = await r.json(); } catch {}
  if (!r.ok) { const e = new Error((data && data.detail) || r.statusText); e.status = r.status; throw e; }
  return data;
}
function td(text) { const c = document.createElement("td"); c.textContent = text == null ? "" : text; return c; }
function bars(el, obj) {
  el.innerHTML = ""; const total = Object.values(obj).reduce((a, b) => a + b, 0) || 1;
  Object.entries(obj).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
    const row = document.createElement("div"); row.className = "bar";
    const l = document.createElement("span"); l.style.width = "90px"; l.textContent = k;
    const tr = document.createElement("div"); tr.className = "track";
    const f = document.createElement("div"); f.className = "fill"; f.style.width = (100 * v / total) + "%"; tr.appendChild(f);
    const n = document.createElement("span"); n.textContent = `${v} (${(100 * v / total).toFixed(0)}%)`;
    row.append(l, tr, n); el.appendChild(row);
  });
}

async function load() {
  let s;
  try { s = await api("/api/admin/stats"); }
  catch (e) { $("#loginBox").style.display = "block"; $("#dash").style.display = "none"; if (e.status !== 401 && e.status !== 403) $("#err").textContent = e.message; return; }
  $("#loginBox").style.display = "none"; $("#dash").style.display = "block";
  $("#sDocs").textContent = s.documents.total;
  $("#sDocsLang").textContent = Object.entries(s.documents.by_language).map(([k, v]) => `${k.toUpperCase()}: ${v}`).join("  ");
  $("#sChunks").textContent = s.index_chunks;
  $("#sQ").textContent = `${s.questions.today} / ${s.questions.this_week}`;
  $("#sRT").textContent = (s.average_response_time_ms / 1000).toFixed(2) + " s";
  $("#sUsers").textContent = s.users;
  $("#sFb").textContent = `👍 ${s.feedback.positive}   👎 ${s.feedback.negative}`;
  bars($("#qLang"), s.questions.by_language); bars($("#dLang"), s.documents.by_language); bars($("#qStatus"), s.questions.by_status);
  const pd = $("#perDay"); pd.innerHTML = ""; const mx = Math.max(1, ...s.questions.per_day.map((d) => d.questions));
  s.questions.per_day.forEach((d) => {
    const c = document.createElement("div"); c.className = "c";
    const b = document.createElement("div"); b.style.height = (110 * d.questions / mx) + "px"; b.title = d.questions;
    c.append(b, document.createTextNode(`${d.date.slice(5)} (${d.questions})`)); pd.appendChild(c);
  });
  loadDocs(); loadLogs();
}
async function loadDocs() {
  const q = $("#docSearch").value.trim();
  const docs = await api("/api/documents" + (q ? "?q=" + encodeURIComponent(q) : ""));
  const tb = $("#docs"); tb.innerHTML = "";
  docs.forEach((d) => {
    const tr = document.createElement("tr");
    const t = document.createElement("td");
    if (d.source_url) { const a = document.createElement("a"); a.href = d.source_url; a.target = "_blank"; a.textContent = d.title; t.appendChild(a); } else t.textContent = d.title;
    const del = document.createElement("button"); del.className = "danger"; del.textContent = "Delete";
    del.onclick = async () => { if (!confirm("Delete from index: " + d.title + "?")) return; await api("/api/documents/" + encodeURIComponent(d.doc_id), { method: "DELETE" }); load(); };
    const c = document.createElement("td"); c.appendChild(del);
    tr.append(t, td(d.language), td(d.category), td(d.n_chunks), c); tb.appendChild(tr);
  });
}
async function loadLogs() {
  const rows = await api("/api/admin/logs?limit=100");
  const tb = $("#logs"); tb.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.append(td(new Date(r.timestamp + "Z").toLocaleString()), td(r.question), td(r.language), td(r.status), td(r.confidence),
      td(r.response_time_ms), td(r.sources), td(r.retrieved_chunks), td(r.feedback == null ? "" : r.feedback > 0 ? "👍" : "👎"));
    tb.appendChild(tr);
  });
}
$("#loginBtn").onclick = async () => {
  try {
    const r = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ email: $("#email").value, password: $("#password").value }) });
    if (r.role !== "admin") { $("#err").textContent = "This account is not an admin"; return; }
    token = r.access_token; store.set("token", token); store.set("role", r.role); load();
  } catch (e) { $("#err").textContent = e.message; }
};
$("#upBtn").onclick = async () => {
  const f = $("#upFile").files[0]; if (!f) { $("#upMsg").textContent = "Choose a file"; return; }
  const fd = new FormData(); fd.append("file", f); fd.append("title", $("#upTitle").value || f.name.replace(/\.[^.]+$/, ""));
  if ($("#upLang").value) fd.append("language", $("#upLang").value); fd.append("category", $("#upCat").value || "admin-upload");
  $("#upMsg").textContent = "Uploading and indexing...";
  try { const r = await api("/api/documents", { method: "POST", body: fd }); $("#upMsg").textContent = `Added: ${r.title} — ${r.pages} pages, ${r.chunks} chunks (${r.language})`; load(); }
  catch (e) { $("#upMsg").textContent = "Error: " + e.message; }
};
$("#reload").onclick = async () => { try { const r = await api("/api/admin/reload", { method: "POST" }); alert(r.models_loaded ? "Models reloaded" : r.error); } catch (e) { alert(e.message); } };
$("#docSearch").oninput = () => loadDocs();
load();
