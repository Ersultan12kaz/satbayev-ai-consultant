const $ = (s) => document.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
};
let lang = store.get("lang") || "ru";
let token = store.get("token");
let role = store.get("role");
let conversationId = null;
let registerMode = false;

function t(k) { return (I18N[lang] && I18N[lang][k]) || I18N.ru[k] || k; }

function applyLang() {
  document.documentElement.lang = lang === "kk" ? "kk" : lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll(".langs button").forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
  const ex = $("#examples"); ex.innerHTML = "";
  t("examplesList").forEach((q) => {
    const b = document.createElement("button"); b.className = "chip"; b.textContent = q;
    b.onclick = () => { $("#q").value = q; ask(); }; ex.appendChild(b);
  });
  renderAuth();
}

function renderAuth() {
  const btn = $("#authBtn");
  btn.textContent = token ? t("logout") : t("login");
  let adm = document.getElementById("adminLink");
  if (token && role === "admin") {
    if (!adm) {
      adm = document.createElement("a"); adm.id = "adminLink"; adm.href = "admin.html"; adm.className = "ghost";
      adm.style.textDecoration = "none"; btn.before(adm);
    }
    adm.textContent = t("admin");
  } else if (adm) adm.remove();
  loadHistory();
}

async function api(path, opts = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  if (token) headers.Authorization = "Bearer " + token;
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  let data = null;
  try { data = await r.json(); } catch {}
  if (!r.ok) throw new Error((data && (data.detail && (data.detail.msg || data.detail))) || r.statusText);
  return data;
}

function addMsg(who, text) {
  $("#welcome") && $("#welcome").remove();
  const row = document.createElement("div"); row.className = "msg " + who;
  const b = document.createElement("div"); b.className = "bubble"; b.textContent = text;
  row.appendChild(b); $("#messages").appendChild(row);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return b;
}

function renderAnswer(bubble, res) {
  bubble.textContent = res.answer;
  bubble.classList.remove("typing");
  if (res.sources && res.sources.length) {
    const s = document.createElement("div"); s.className = "sources";
    const h = document.createElement("b"); h.textContent = t("sources") + ":"; s.appendChild(h);
    res.sources.forEach((src) => {
      const a = document.createElement(src.url ? "a" : "span"); a.className = "src";
      if (src.url) { a.href = src.url; a.target = "_blank"; a.rel = "noopener"; }
      a.textContent = `📄 [${src.n}] ${src.title} — ${t("page")} ${src.page}` + (src.lang && src.lang !== res.language ? ` (${src.lang.toUpperCase()})` : "");
      s.appendChild(a);
    });
    bubble.appendChild(s);
  }
  const m = document.createElement("div"); m.className = "meta";
  if (res.status === "answered") {
    const badge = document.createElement("span"); badge.className = "badge " + res.confidence;
    badge.textContent = `${t("confidence")}: ${t("conf")[res.confidence] || res.confidence}`;
    m.appendChild(badge);
  }
  m.appendChild(document.createTextNode(`${t("time")}: ${(res.response_time_ms / 1000).toFixed(2)} s · ${res.language.toUpperCase()} `));
  const fb = document.createElement("span"); fb.className = "fb";
  fb.appendChild(document.createTextNode(" " + t("helpful")));
  [["👍", 1], ["👎", -1]].forEach(([icon, v]) => {
    const bt = document.createElement("button"); bt.textContent = icon;
    bt.onclick = async () => {
      fb.querySelectorAll("button").forEach((x) => x.classList.remove("sel")); bt.classList.add("sel");
      try { await api(`/api/messages/${res.message_id}/feedback`, { method: "POST", body: JSON.stringify({ value: v }) }); } catch {}
    };
    fb.appendChild(bt);
  });
  m.appendChild(fb);
  bubble.appendChild(m);
  if ($("#showDebug").checked && res.retrieved) {
    const d = document.createElement("div"); d.className = "sources";
    if (res.question_type) {
      const u = document.createElement("span"); u.className = "src";
      u.textContent = "🔎 " + Object.entries(res.question_type).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")
        + (res.concepts && res.concepts.length ? " · " + res.concepts.join(", ") : "");
      d.appendChild(u);
    }
    const h = document.createElement("b"); h.textContent = t("retrieved") + ":"; d.appendChild(h);
    res.retrieved.forEach((r, i) => {
      const x = document.createElement("span"); x.className = "src";
      x.textContent = `${i + 1}. ${r.title} (${t("page")} ${r.page}, ${r.lang}) — dense ${r.dense}, bm25 ${r.bm25}, hybrid ${r.hybrid}, rerank ${r.rerank}`;
      d.appendChild(x);
    });
    bubble.appendChild(d);
  }
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

async function ask(ev) {
  if (ev) ev.preventDefault();
  const q = $("#q").value.trim();
  if (q.length < 2) return;
  $("#q").value = ""; $("#send").disabled = true;
  addMsg("user", q);
  const bubble = addMsg("bot", t("thinking")); bubble.classList.add("typing");
  try {
    const res = await api("/api/chat", { method: "POST", body: JSON.stringify({
      question: q, conversation_id: conversationId, mode: $("#mode").value, top_k: +$("#topk").value }) });
    conversationId = res.conversation_id;
    renderAnswer(bubble, res);
    loadHistory();
  } catch (e) {
    bubble.classList.remove("typing"); bubble.textContent = `${t("error")}: ${e.message}`;
  } finally { $("#send").disabled = false; $("#q").focus(); }
}

async function loadHistory() {
  const h = $("#history");
  if (!token) { h.innerHTML = ""; const d = document.createElement("div"); d.className = "hist-item"; d.textContent = t("loginForHistory"); h.appendChild(d); return; }
  try {
    const list = await api("/api/conversations");
    h.innerHTML = "";
    list.forEach((c) => {
      const d = document.createElement("div"); d.className = "hist-item"; d.textContent = c.title; d.title = c.title;
      d.onclick = () => openConversation(c.id); h.appendChild(d);
    });
  } catch { }
}

async function openConversation(id) {
  const c = await api(`/api/conversations/${id}`);
  conversationId = c.id; $("#messages").innerHTML = "";
  c.messages.forEach((m) => {
    addMsg("user", m.question);
    renderAnswer(addMsg("bot", ""), { answer: m.answer, sources: m.sources, status: m.status, confidence: m.confidence,
      language: m.language, response_time_ms: m.response_time_ms, message_id: m.id });
  });
}

function openAuth() {
  registerMode = false; updateAuthModal(); $("#authErr").textContent = ""; $("#authModal").classList.add("open");
}
function updateAuthModal() {
  $("#authTitle").textContent = registerMode ? t("register") : t("login");
  $("#authSubmit").textContent = registerMode ? t("register") : t("login");
  $("#authSwitch").textContent = registerMode ? t("haveAccount") : t("register");
  $("#fullName").style.display = registerMode ? "block" : "none";
}
async function submitAuth() {
  const body = { email: $("#email").value.trim(), password: $("#password").value };
  if (registerMode) body.full_name = $("#fullName").value.trim();
  try {
    const r = await api(registerMode ? "/api/auth/register" : "/api/auth/login", { method: "POST", body: JSON.stringify(body) });
    token = r.access_token; role = r.role; store.set("token", token); store.set("role", role);
    $("#authModal").classList.remove("open"); renderAuth();
  } catch (e) { $("#authErr").textContent = e.message; }
}

document.querySelectorAll(".langs button").forEach((b) => b.onclick = () => { lang = b.dataset.lang; store.set("lang", lang); applyLang(); });
$("#authBtn").onclick = () => {
  if (token) { token = null; role = null; store.set("token", null); store.set("role", null); conversationId = null; renderAuth(); }
  else openAuth();
};
$("#authSubmit").onclick = submitAuth;
$("#authSwitch").onclick = () => { registerMode = !registerMode; updateAuthModal(); };
$("#authClose").onclick = () => $("#authModal").classList.remove("open");
$("#composer").addEventListener("submit", ask);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); });
applyLang();
