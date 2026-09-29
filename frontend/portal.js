/* Student portal page with the AI consultant (our RAG models) built in. */
const $ = (s) => document.querySelector(s);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v === null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
};
let lang = store.get("lang") || "kk";
let token = store.get("token");
let role = store.get("role");
let userName = store.get("userName") || "";
let conversationId = null;
let registerMode = false;

const T = {
  kk: { portal: "Студент порталы", vision: "Нашар көретіндерге арналған нұсқа", visionOff: "Қалыпты нұсқа",
    bachelor: "Бакалавриат", guest: "Қонақ", login: "Кіру", logout: "Шығу", register: "Тіркелу", close: "Жабу",
    fullName: "Аты-жөні", password: "Құпия сөз", haveAccount: "Аккаунт бар ма? Кіру",
    newChat: "Жаңа сұхбат", noNotif: "Сізде жаңа хабарлама жоқ", cta: "AI-кеңесшімен сөйлесу", fab: "AI-кеңесші",
    botTitle: "Satbayev AI-кеңесші", botSub: "Ресми құжаттар бойынша жауап береді · KZ / RU / EN",
    mode: "Іздеу режимі", debug: "Табылған үзінділерді көрсету", showSrc: "Дереккөздерді көрсету", admin: "Әкімші панелі",
    placeholder: "Сұрағыңызды жазыңыз...", thinking: "Ресми құжаттардан іздеп жатырмын...",
    sources: "Дереккөздер", confidence: "Сенімділік", page: "бет", time: "жауап уақыты", helpful: "Пайдалы ма?",
    conf: { high: "жоғары", medium: "орташа", low: "төмен" }, error: "Қате", retrieved: "Табылған үзінділер",
    ask: "AI-дан сұрау", services: "Сервистер", student: "Студентке",
    hello: "Сәлеметсіз бе! Мен Satbayev University-дің AI-кеңесшісімін. Университеттің 750-ден астам ресми құжаты мен бірінші курсқа арналған анықтама-кітапша бойынша жауап беремін. Сұрағыңызды қазақ, орыс немесе ағылшын тілінде жазыңыз.",
    docs: "құжат", chunks: "фрагмент", langs3: "тіл",
  },
  ru: { portal: "Портал студента", vision: "Версия для слабовидящих", visionOff: "Обычная версия",
    bachelor: "Бакалавриат", guest: "Гость", login: "Вход", logout: "Выход", register: "Регистрация", close: "Закрыть",
    fullName: "Имя", password: "Пароль", haveAccount: "Уже есть аккаунт? Войти",
    newChat: "Новый диалог", noNotif: "У Вас нет новых уведомлений", cta: "Спросить AI-консультанта", fab: "AI-консультант",
    botTitle: "Satbayev AI-консультант", botSub: "Отвечает по официальным документам · KZ / RU / EN",
    mode: "Режим поиска", debug: "Показать найденные фрагменты", showSrc: "Показывать источники", admin: "Админ-панель",
    placeholder: "Напишите вопрос...", thinking: "Ищу в официальных документах...",
    sources: "Источники", confidence: "Уверенность", page: "стр.", time: "время ответа", helpful: "Полезно?",
    conf: { high: "высокая", medium: "средняя", low: "низкая" }, error: "Ошибка", retrieved: "Найденные фрагменты",
    ask: "Спросить AI", services: "Сервисы", student: "Студенту",
    hello: "Здравствуйте! Я AI-консультант Satbayev University. Отвечаю по более чем 750 официальным документам университета и справочнику первокурсника. Пишите вопрос на казахском, русском или английском.",
    docs: "документов", chunks: "фрагментов", langs3: "языка",
  },
  en: { portal: "Student portal", vision: "Version for the visually impaired", visionOff: "Normal version",
    bachelor: "Bachelor", guest: "Guest", login: "Log in", logout: "Log out", register: "Sign up", close: "Close",
    fullName: "Name", password: "Password", haveAccount: "Have an account? Log in",
    newChat: "New chat", noNotif: "You have no new notifications", cta: "Ask the AI consultant", fab: "AI consultant",
    botTitle: "Satbayev AI consultant", botSub: "Answers from the official documents · KZ / RU / EN",
    mode: "Search mode", debug: "Show retrieved chunks", showSrc: "Show sources", admin: "Admin panel",
    placeholder: "Type your question...", thinking: "Searching the official documents...",
    sources: "Sources", confidence: "Confidence", page: "p.", time: "response time", helpful: "Helpful?",
    conf: { high: "high", medium: "medium", low: "low" }, error: "Error", retrieved: "Retrieved chunks",
    ask: "Ask AI", services: "Services", student: "For students",
    hello: "Hello! I am the Satbayev University AI consultant. I answer from more than 750 official university documents and the freshman guide. Ask in Kazakh, Russian or English.",
    docs: "documents", chunks: "fragments", langs3: "languages",
  },
};
const t = (k) => (T[lang] && T[lang][k]) || T.kk[k] || k;

const EXAMPLES = {
  kk: ["Жатақханаға орналасу үшін қандай құжаттар керек?", "Шәкіртақы алуға кімнің құқығы бар?",
       "Академиялық демалысты қалай рәсімдеуге болады?", "Емтихан бағасына апелляция қалай беріледі?", "Ректор кім?"],
  ru: ["Какие документы нужны для заселения в общежитие?", "Кто имеет право на получение стипендии?",
       "Как оформить академический отпуск?", "Как подать апелляцию на оценку экзамена?", "Кто ректор университета?"],
  en: ["What documents are needed to move into the dormitory?", "Who is eligible for a scholarship?",
       "How do I take academic leave?", "How can I appeal an exam grade?", "Who is the rector?"],
};

const SLIDES = [
  { kk: ["Модуль: AI-кеңесші", "Енді порталда университеттің ресми құжаттары бойынша жауап беретін виртуалды кеңесші бар.", "✅ Студенттерге — сұраққа 1 секундта жауап, дереккөзі (құжат пен бет) көрсетіледі.", "✅ Университетке — студенттерге қызмет көрсету орталығына түсетін қайталанатын сұрақтар азаяды."],
    ru: ["Модуль: AI-консультант", "Теперь в портале есть виртуальный консультант, который отвечает по официальным документам университета.", "✅ Для студентов — ответ за 1 секунду с источником (документ и страница).", "✅ Для университета — меньше повторяющихся вопросов в ЦОС."],
    en: ["Module: AI consultant", "The portal now has a virtual consultant that answers from the official university documents.", "✅ For students — an answer in a second with the source (document and page).", "✅ For the university — fewer repeated questions to the service center."] },
  { kk: ["RAG технологиясы", "Сұрақ → тілді анықтау → құжаттардан іздеу (эмбеддинг + BM25) → reranker → жауап құрастыру.", "Барлық модельді біз университет құжаттарымен өзіміз үйреттік — дайын модель немесе сыртқы API жоқ."],
    ru: ["Технология RAG", "Вопрос → определение языка → поиск по документам (эмбеддинги + BM25) → reranker → сборка ответа.", "Все модели обучены нами на документах университета — без готовых моделей и внешних API."],
    en: ["RAG technology", "Question → language detection → search (embeddings + BM25) → reranker → answer composition.", "All models were trained by us on the university documents — no pre-trained models or external APIs."] },
  { kk: ["Үш тілде", "Қазақша, орысша, ағылшынша сұраңыз — тіпті қазақ әріптерінсіз («жатакхана калай алам») немесе ауызекі тілде («обшага», «академка») жазсаңыз да түсінеді."],
    ru: ["На трех языках", "Спрашивайте на казахском, русском или английском — понимает и разговорные слова («общага», «академка») и текст без казахских букв."],
    en: ["Three languages", "Ask in Kazakh, Russian or English — it also understands colloquial words and Kazakh typed without special letters."] },
  { kk: ["Ойдан шығармайды", "Жауаптағы әр сөйлем ресми құжаттан алынады. Құжаттарда жоқ сұраққа «ақпарат табылмады» деп жауап береді, университетке қатысы жоқ сұрақтардан бас тартады."],
    ru: ["Без выдумок", "Каждое предложение ответа взято из официального документа. Если информации нет — честно сообщает об этом и не отвечает на вопросы не по теме университета."],
    en: ["No hallucinations", "Every sentence comes from an official document. If the information is missing it says so, and it declines off-topic questions."] },
];

/* ---------------------------------------------------------------- page */
let slideIdx = 0, slideTimer = null;
function renderSlides() {
  const s = $("#slides"), d = $("#dots"); s.innerHTML = ""; d.innerHTML = "";
  SLIDES.forEach((sl, i) => {
    const [title, ...lines] = sl[lang];
    const el = document.createElement("div"); el.className = "slide" + (i === slideIdx ? " on" : "");
    const txt = document.createElement("div"); txt.className = "txt";
    const h = document.createElement("h4"); h.textContent = title; txt.appendChild(h);
    lines.forEach((l) => { const p = document.createElement("p"); p.textContent = l; txt.appendChild(p); });
    const art = document.createElement("div"); art.className = "art";
    art.innerHTML = `<svg viewBox="0 0 120 150" width="120" height="150"><circle cx="60" cy="42" r="30" fill="#fff"/><rect x="38" y="30" width="44" height="22" rx="11" fill="#12306b"/><circle cx="50" cy="41" r="5" fill="#5ec8ff"/><circle cx="70" cy="41" r="5" fill="#5ec8ff"/><rect x="30" y="76" width="60" height="58" rx="16" fill="#fff"/><rect x="44" y="92" width="32" height="26" rx="4" fill="#12306b"/><text x="60" y="110" text-anchor="middle" font-size="12" font-weight="700" fill="#5ec8ff">AI</text></svg>`;
    el.append(txt, art); s.appendChild(el);
    const dot = document.createElement("button"); dot.className = i === slideIdx ? "on" : ""; dot.setAttribute("aria-label", `slide ${i + 1}`);
    dot.onclick = () => { slideIdx = i; renderSlides(); }; d.appendChild(dot);
  });
  clearTimeout(slideTimer);
  slideTimer = setTimeout(() => { slideIdx = (slideIdx + 1) % SLIDES.length; renderSlides(); }, 7000);
}

async function renderStats() {
  try {
    const h = await (await fetch("/api/health")).json();
    const box = $("#stats"); box.innerHTML = "";
    [[h.documents || "750+", t("docs")], [h.chunks || 0, t("chunks")], [3, t("langs3")]].forEach(([v, l]) => {
      const d = document.createElement("div"); d.className = "stat";
      const b = document.createElement("b"); b.textContent = typeof v === "number" ? v.toLocaleString() : v;
      const s = document.createElement("span"); s.textContent = l; d.append(b, s); box.appendChild(d);
    });
  } catch {}
}

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
  document.querySelectorAll(".langs button").forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
  $("#q").placeholder = t("placeholder");
  $("#password").placeholder = t("password"); $("#fullName").placeholder = t("fullName");
  $("#visionBtn span").textContent = document.documentElement.classList.contains("vision") ? t("visionOff") : t("vision");
  renderSlides(); renderStats(); renderAuth(); renderQuick();
  const first = $("#messages").firstElementChild;
  if (!first) addMsg("bot hello", t("hello"));
  else if (first.classList.contains("hello")) first.textContent = t("hello");
}

function renderAuth() {
  $("#uname").textContent = token ? (userName || "Студент") : t("guest");
  $("#avatar").textContent = token ? initials(userName || "С") : "?";
  $("#authLabel").textContent = token ? t("logout") : t("login");
  $("#adminLink").style.display = token && role === "admin" ? "inline" : "none";
}
const initials = (n) => n.split(/\s+/).filter(Boolean).slice(0, 2).map((x) => x[0].toUpperCase()).join("");

function renderQuick() {
  const box = $("#quick"); box.innerHTML = "";
  EXAMPLES[lang].forEach((q) => {
    const b = document.createElement("button"); b.textContent = q; b.onclick = () => ask(q); box.appendChild(b);
  });
}

/* ---------------------------------------------------------------- assistant */
async function api(path, opts = {}) {
  const headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  if (token) headers.Authorization = "Bearer " + token;
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  let data = null;
  try { data = await r.json(); } catch {}
  if (!r.ok) throw new Error((data && (data.detail && (data.detail.msg || data.detail))) || r.statusText);
  return data;
}

function newChat() {
  conversationId = null; $("#messages").innerHTML = ""; addMsg("bot hello", t("hello")); $("#q").focus();
}

function addMsg(who, text) {
  const b = document.createElement("div"); b.className = "msg " + who; b.textContent = text;
  $("#messages").appendChild(b); $("#messages").scrollTop = $("#messages").scrollHeight;
  return b;
}

function renderAnswer(bubble, res) {
  bubble.classList.remove("typing");
  bubble.textContent = res.answer;
  const srcBox = document.getElementById("showSources");
  if (srcBox && srcBox.checked && res.sources && res.sources.length) {
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
    badge.textContent = `${t("confidence")}: ${t("conf")[res.confidence] || res.confidence}`; m.appendChild(badge);
  }
  m.appendChild(document.createTextNode(`${t("time")}: ${(res.response_time_ms / 1000).toFixed(2)} s · ${String(res.language).toUpperCase()} · ${t("helpful")}`));
  [["👍", 1], ["👎", -1]].forEach(([icon, v]) => {
    const bt = document.createElement("button"); bt.textContent = icon;
    bt.onclick = async () => {
      m.querySelectorAll("button").forEach((x) => x.classList.remove("sel")); bt.classList.add("sel");
      try { await api(`/api/messages/${res.message_id}/feedback`, { method: "POST", body: JSON.stringify({ value: v }) }); } catch {}
    };
    m.appendChild(bt);
  });
  bubble.appendChild(m);
  if ($("#showDebug").checked && res.retrieved) {
    const d = document.createElement("div"); d.className = "sources";
    if (res.question_type) {
      const u = document.createElement("span"); u.className = "src";
      u.textContent = "🔎 " + Object.entries(res.question_type).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")
        + (res.concepts && res.concepts.length ? " · " + res.concepts.join(", ") : ""); d.appendChild(u);
    }
    const h = document.createElement("b"); h.textContent = t("retrieved") + ":"; d.appendChild(h);
    res.retrieved.forEach((r, i) => {
      const x = document.createElement("span"); x.className = "src";
      x.textContent = `${i + 1}. ${r.title} (${t("page")} ${r.page}, ${r.lang}) — dense ${r.dense}, bm25 ${r.bm25}, rerank ${r.rerank}`;
      d.appendChild(x);
    });
    bubble.appendChild(d);
  }
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

async function ask(text) {
  const q = (text || $("#q").value).trim();
  if (q.length < 2) return;
  $("#q").value = ""; $("#send").disabled = true;
  addMsg("user", q);
  const bubble = addMsg("bot typing", t("thinking"));
  try {
    const res = await api("/api/chat", { method: "POST", body: JSON.stringify({
      question: q, conversation_id: conversationId, mode: $("#mode").value, top_k: +$("#topk").value }) });
    conversationId = res.conversation_id;
    renderAnswer(bubble, res);
  } catch (e) {
    bubble.classList.remove("typing"); bubble.textContent = `${t("error")}: ${e.message}`;
  } finally { $("#send").disabled = false; }
}

/* ---------------------------------------------------------------- auth */
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
    try { const me = await api("/api/auth/me"); userName = me.full_name || me.email || ""; store.set("userName", userName); } catch {}
    $("#authModal").classList.remove("open"); renderAuth();
  } catch (e) { $("#authErr").textContent = e.message; }
}

/* ---------------------------------------------------------------- events */
document.querySelectorAll(".langs button").forEach((b) => b.onclick = () => { lang = b.dataset.lang; store.set("lang", lang); applyLang(); });
$("#authBtn").onclick = () => {
  if (token) { token = role = null; userName = ""; ["token", "role", "userName"].forEach((k) => store.set(k, null)); conversationId = null; renderAuth(); }
  else { registerMode = false; updateAuthModal(); $("#authErr").textContent = ""; $("#authModal").classList.add("open"); }
};
$("#authSubmit").onclick = submitAuth;
$("#authSwitch").onclick = () => { registerMode = !registerMode; updateAuthModal(); };
$("#authClose").onclick = () => $("#authModal").classList.remove("open");
$("#visionBtn").onclick = () => {
  const on = document.documentElement.classList.toggle("vision"); store.set("vision", on ? "1" : null);
  $("#visionBtn span").textContent = on ? t("visionOff") : t("vision");
};
$("#homeBtn").onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
$("#ctaBtn").onclick = newChat;
$("#settingsBtn").onclick = () => $("#dsettings").classList.toggle("open");
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
if (store.get("vision")) document.documentElement.classList.add("vision");
applyLang();
