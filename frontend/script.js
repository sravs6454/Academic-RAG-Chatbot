/* ================================================================
   AcaRAG Pro — Frontend Logic
   Served from FastAPI at http://localhost:8000  (same origin)
   ================================================================ */

// Empty string = relative URL → works when served by FastAPI at :8000
// Fallback to explicit port when opened as file:// (dev convenience)
const BACKEND_URL = window.location.protocol === "file:"
  ? "http://localhost:8000"
  : "";

// ── State ────────────────────────────────────────────────────────
let sessionId        = null;   // assigned by backend on first message
let isWaiting        = false;
let studentProfile   = null;   // { name, year, semester, branch }
let sidebarOpen      = true;
let activeConvId     = null;   // ID of the currently active conversation

// ── DOM shortcuts ─────────────────────────────────────────────────
const chatBody   = document.getElementById("chatBody");
const userInput  = document.getElementById("userInput");
const emojiPanel = document.getElementById("emojiPanel");

// ================================================================
//  PROFILE MANAGEMENT
// ================================================================

const PROFILE_KEY    = "acarag_profile_v2";
const SESSION_ID_KEY = "acarag_session_id";
const DEVICE_ID_KEY  = "acarag_device_id";

// Each browser gets a stable random ID stored in localStorage.
// This is the primary namespace separator — even if two students
// enter identical profiles on different devices, their histories stay separate.
function getDeviceId() {
  let id = localStorage.getItem(DEVICE_ID_KEY);
  if (!id) {
    id = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    localStorage.setItem(DEVICE_ID_KEY, id);
  }
  return id;
}

// Returns a per-student-per-device storage key.
// Format: acarag_conv_{deviceId}_{name}_{year}_{branch}_{semester}
function getConvKey() {
  const dev = getDeviceId();
  if (!studentProfile) return `acarag_conv_${dev}_default`;
  const { name = "", year = "", branch = "", semester = "" } = studentProfile;
  const slug = [name, year, branch, semester]
    .map(s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, ""))
    .join("_");
  return `acarag_conv_${dev}_${slug}`;
}

// ================================================================
//  CONVERSATION HISTORY (localStorage-based, like ChatGPT)
// ================================================================

function loadConvList() {
  try {
    const raw = localStorage.getItem(getConvKey());
    if (raw) return JSON.parse(raw);
  } catch (_) {}
  return [];
}

function saveConvList(list) {
  localStorage.setItem(getConvKey(), JSON.stringify(list));
}

function getConv(id) {
  return loadConvList().find(c => c.id === id) || null;
}

function upsertConv(conv) {
  const list = loadConvList();
  const idx  = list.findIndex(c => c.id === conv.id);
  if (idx >= 0) list[idx] = conv;
  else          list.unshift(conv);  // newest first
  // Keep only last 30 conversations
  saveConvList(list.slice(0, 30));
}

function deleteConv(id) {
  const list = loadConvList().filter(c => c.id !== id);
  saveConvList(list);
}

/** Add a message to the active conversation in storage. */
function appendMsgToConv(convId, role, text) {
  const list = loadConvList();
  const conv = list.find(c => c.id === convId);
  if (!conv) return;
  conv.messages.push({ role, text, ts: Date.now() });
  saveConvList(list);
}

/** Render the conversations list in the sidebar. */
function renderConvList() {
  const el = document.getElementById("convList");
  if (!el) return;
  const list = loadConvList();
  if (!list.length) {
    el.innerHTML = `<div class="conv-empty">No history yet</div>`;
    return;
  }
  el.innerHTML = list.map(c => {
    const active = c.id === activeConvId ? " conv-active" : "";
    const title  = (c.title || "Untitled").substring(0, 34);
    const date   = new Date(c.ts).toLocaleDateString([], { month: "short", day: "numeric" });
    return `
      <div class="conv-item${active}" onclick="loadConv('${c.id}')">
        <span class="conv-title">${escHtml(title)}</span>
        <span class="conv-date">${date}</span>
        <button class="conv-delete-btn" onclick="removeConv(event,'${c.id}')" title="Delete">×</button>
      </div>`;
  }).join("");
}

/** Load a past conversation into the chat view. */
function loadConv(id) {
  const conv = getConv(id);
  if (!conv) return;

  activeConvId = id;
  sessionId    = conv.sessionId || null;

  chatBody.innerHTML = "";
  for (const msg of conv.messages) {
    if (msg.role === "user") {
      addUserMessage(msg.text, false);    // false = don't save again
    } else {
      // Bot messages were stored as plain text; re-render
      addBotMessage(formatText(msg.text), [], null, false);
    }
  }
  renderConvList();
  scrollDown();
}

/** Start a fresh conversation and create a storage entry. */
function startNewConv() {
  activeConvId = `conv_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
  sessionId    = null;
  localStorage.removeItem(SESSION_ID_KEY);

  const conv = {
    id:        activeConvId,
    title:     "New conversation",
    messages:  [],
    sessionId: null,
    ts:        Date.now(),
  };
  upsertConv(conv);
  return conv;
}

/** Update conversation title from first user message. */
function maybeSetConvTitle(convId, userText) {
  const list = loadConvList();
  const conv = list.find(c => c.id === convId);
  if (!conv) return;
  if (conv.title === "New conversation") {
    conv.title = userText.substring(0, 45);
    saveConvList(list);
  }
}

function removeConv(e, id) {
  e.stopPropagation();
  deleteConv(id);
  if (activeConvId === id) {
    // Start fresh if we deleted the active conversation
    newChat();
    return;
  }
  renderConvList();
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function loadSavedProfile() {
  try {
    const raw = localStorage.getItem(PROFILE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (_) {}
  return null;
}

function saveProfile(profile) {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

const YEAR_LABEL = { "1": "1st Year", "2": "2nd Year", "3": "3rd Year", "4": "4th Year" };

function applyProfile(profile) {
  studentProfile = profile;

  const displayName = profile.name || "Student";
  const yearLabel   = YEAR_LABEL[profile.year] || `Year ${profile.year}`;
  const details     = `${yearLabel} · Sem ${profile.semester} · ${profile.branch}`;

  // Update sidebar profile card
  const nameEl    = document.getElementById("spName");
  const detailEl  = document.getElementById("spDetails");
  const chipEl    = document.getElementById("topbarChip");
  if (nameEl)   nameEl.textContent   = displayName;
  if (detailEl) detailEl.textContent = details;
  if (chipEl)   chipEl.innerHTML     =
    `<span class="chip-name">${displayName}</span><span class="chip-detail">${yearLabel} · Sem ${profile.semester}</span>`;

  // Show app, hide profile overlay
  document.getElementById("profileOverlay").style.display = "none";
  document.getElementById("appLayout").style.display      = "flex";

  // Render conversation list in sidebar
  renderConvList();

  // Resume last active conversation or start a fresh one
  const convList = loadConvList();
  if (convList.length > 0) {
    loadConv(convList[0].id);   // most recent conversation
  } else {
    startNewConv();
    renderWelcome(profile);
  }
}

function renderWelcome(profile) {
  const yearLabel   = YEAR_LABEL[profile.year] || `Year ${profile.year}`;
  const displayName = profile.name ? ` ${profile.name}` : "";

  chatBody.innerHTML = `
    <div class="msg-wrapper bot-wrapper">
      <div class="bot-avatar">AR</div>
      <div class="bot-msg">
        <p>Hi${displayName}! 👋 I'm your Academic Assistant for <strong>SVECW, Bhimavaram</strong>.</p>
        <p>I see you are a <strong>${yearLabel}, Semester ${profile.semester}</strong> student in <strong>${profile.branch}</strong>.
           I'll automatically filter exam schedules, syllabus, and regulations for your year and semester.</p>
        <div class="quick-replies">
          <button onclick="askQuick('What are my exam dates?')">📅 My Exam Dates</button>
          <button onclick="askQuick('Show me the academic calendar for this semester.')">🗓 Academic Calendar</button>
          <button onclick="askQuick('What are the exam regulations and attendance rules?')">📋 Exam Regulations</button>
          <button onclick="askQuick('What subjects do I have this semester?')">📚 My Subjects</button>
        </div>
      </div>
    </div>`;
}

function editProfile() {
  localStorage.removeItem(PROFILE_KEY);
  localStorage.removeItem(SESSION_ID_KEY);
  sessionId    = null;
  activeConvId = null;
  document.getElementById("profileOverlay").style.display = "flex";
  document.getElementById("appLayout").style.display      = "none";
}

// Profile form submit
document.getElementById("profileForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const year     = document.getElementById("pYear").value;
  const semester = document.getElementById("pSemester").value;
  const branch   = document.getElementById("pBranch").value;
  const name     = document.getElementById("pName").value.trim();

  if (!year || !semester || !branch) {
    document.getElementById("profileError").style.display = "block";
    return;
  }
  document.getElementById("profileError").style.display = "none";

  const profile = { name, year, semester, branch };
  saveProfile(profile);
  applyProfile(profile);
});

// ================================================================
//  SEND MESSAGE
// ================================================================

// Enter = send · Shift+Enter = newline
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !isWaiting) {
    e.preventDefault();
    sendMessage();
  }
});

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function askQuick(text) {
  userInput.value = text;
  sendMessage();
}

function sendMessage() {
  const msg = userInput.value.trim();
  if (!msg || isWaiting) return;

  // Ensure an active conversation exists
  if (!activeConvId) startNewConv();

  addUserMessage(msg);
  maybeSetConvTitle(activeConvId, msg);
  userInput.value = "";
  userInput.style.height = "auto";

  // Local short-circuit for greetings
  const lower = msg.toLowerCase().replace(/[^\w\s]/g, "").trim();
  const greetings = ["hi", "hello", "hii", "hey", "hey there", "hello there",
    "hi there", "good morning", "good afternoon", "good evening", "greetings"];
  if (greetings.some(g => lower.startsWith(g))) {
    const n = studentProfile?.name ? `, ${studentProfile.name}` : "";
    const botText = `Hi${n}! 👋 How can I help you academically today?`;
    addBotMessage(`<p>${botText}</p>`, [], null);
    appendMsgToConv(activeConvId, "bot", botText);
    return;
  }
  if (lower === "how are you") {
    const botText = "I'm ready to assist! What academic question do you have?";
    addBotMessage(`<p>${botText}</p>`, [], null);
    appendMsgToConv(activeConvId, "bot", botText);
    return;
  }

  const typingEl = addTypingIndicator();
  isWaiting = true;
  setInputState(false);

  const payload = { question: msg, session_id: sessionId };
  if (studentProfile) {
    payload.year     = studentProfile.year;
    payload.semester = studentProfile.semester;
    payload.branch   = studentProfile.branch;
    payload.name     = studentProfile.name || "";
  }

  fetch(`${BACKEND_URL}/chat`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  })
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(data => {
      typingEl.remove();

      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem(SESSION_ID_KEY, sessionId);
        // Persist session ID to conversation
        const list = loadConvList();
        const conv = list.find(c => c.id === activeConvId);
        if (conv) { conv.sessionId = sessionId; saveConvList(list); }
      }

      const answer = data.answer ||
        "Information not available in provided academic documents.";

      // Show autocorrect notice if the query was changed
      let correctionHtml = "";
      if (data.corrected_query) {
        correctionHtml = `<div class="correction-notice">
          🔤 Auto-corrected: <em>${escHtml(data.corrected_query)}</em></div>`;
      }

      addBotMessage(correctionHtml + formatText(answer), data.sources || [], data.cache_hit || null);
      appendMsgToConv(activeConvId, "bot", answer);
      renderConvList();
    })
    .catch(err => {
      typingEl.remove();
      addBotMessage(
        `<p>⚠️ Could not connect to the academic server. Make sure the backend is running:<br>
        <code>uvicorn app:app --host 0.0.0.0 --port 8000</code></p>`,
        [], null
      );
      console.error("AcaRAG error:", err);
    })
    .finally(() => {
      isWaiting = false;
      setInputState(true);
      userInput.focus();
    });
}

// ================================================================
//  TEXT FORMATTER
// ================================================================

function formatText(text) {
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  const lines = text.split("\n");
  let html = "", inList = "";

  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith("- ") || t.startsWith("• ")) {
      if (inList !== "ul") { if (inList) html += `</${inList}>`; html += "<ul>"; inList = "ul"; }
      html += `<li>${t.slice(2)}</li>`;
    } else if (/^\d+\.\s/.test(t)) {
      if (inList !== "ol") { if (inList) html += `</${inList}>`; html += "<ol>"; inList = "ol"; }
      html += `<li>${t.replace(/^\d+\.\s/, "")}</li>`;
    } else {
      if (inList) { html += `</${inList}>`; inList = ""; }
      if (t) html += `<p>${t}</p>`;
    }
  }
  if (inList) html += `</${inList}>`;
  return html || `<p>${text}</p>`;
}

// ================================================================
//  MESSAGE RENDERERS
// ================================================================

function addBotMessage(htmlContent, sources = [], cacheHit = null, save = true) {
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const wrapper = document.createElement("div");
  wrapper.className = "msg-wrapper bot-wrapper";

  const avatar = document.createElement("div");
  avatar.className = "bot-avatar";
  avatar.textContent = "AR";

  const bubble = document.createElement("div");
  bubble.className = "bot-msg";
  bubble.innerHTML = htmlContent;

  // Source citation card
  if (sources && sources.length > 0) {
    const card = document.createElement("div");
    card.className = "source-card";
    card.innerHTML = `<div class="source-label"><span class="source-icon">📄</span> Sources</div>`;
    sources.forEach(src => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.innerHTML =
        `<span class="doc-name">${src.document || "Unknown"}</span>` +
        `<span class="page-badge">p.${src.page_number || src.page || "?"}</span>`;
      card.appendChild(item);
    });
    bubble.appendChild(card);
  }

  // Cache badge
  if (cacheHit) {
    const badge = document.createElement("div");
    badge.className = "cache-badge";
    badge.textContent = cacheHit === "exact"
      ? "⚡ Instant reply (cached)"
      : "🔁 Fast reply (semantic cache)";
    bubble.appendChild(badge);
  }

  // Timestamp
  const ts = document.createElement("div");
  ts.className = "msg-time";
  ts.textContent = timeStr;
  bubble.appendChild(ts);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatBody.appendChild(wrapper);
  scrollDown();
}

function addUserMessage(text, save = true) {
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const wrapper = document.createElement("div");
  wrapper.className = "msg-wrapper user-wrapper";
  const bubble = document.createElement("div");
  bubble.className = "user-msg";
  bubble.innerHTML = `<p>${escHtml(text)}</p><div class="msg-time user-time">${timeStr}</div>`;
  wrapper.appendChild(bubble);
  chatBody.appendChild(wrapper);
  scrollDown();
  if (save && activeConvId) appendMsgToConv(activeConvId, "user", text);
}

function addTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "msg-wrapper bot-wrapper";
  const avatar = document.createElement("div");
  avatar.className = "bot-avatar";
  avatar.textContent = "AR";
  const bubble = document.createElement("div");
  bubble.className = "bot-msg typing-bubble";
  bubble.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatBody.appendChild(wrapper);
  scrollDown();
  return wrapper;
}

function scrollDown() {
  chatBody.scrollTop = chatBody.scrollHeight;
}

function setInputState(enabled) {
  userInput.disabled = !enabled;
  document.querySelector(".send-btn").disabled = !enabled;
}

// ================================================================
//  UI CONTROLS
// ================================================================

function newChat() {
  startNewConv();
  chatBody.innerHTML = "";
  if (studentProfile) renderWelcome(studentProfile);
  renderConvList();
}

function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  document.getElementById("sidebar").classList.toggle("sidebar-hidden", !sidebarOpen);
  document.getElementById("appLayout").classList.toggle("sidebar-collapsed", !sidebarOpen);
}

function toggleTheme() {
  document.body.classList.toggle("dark");
  const btn = document.querySelector(".theme-toggle-btn");
  if (btn) btn.textContent = document.body.classList.contains("dark")
    ? "☀️ Light Mode" : "🌙 Dark Mode";
}

function toggleEmoji() {
  emojiPanel.style.display = emojiPanel.style.display === "flex" ? "none" : "flex";
}

function addEmoji(emoji) {
  userInput.value += emoji;
  emojiPanel.style.display = "none";
  userInput.focus();
}

// ================================================================
//  HEALTH CHECK
// ================================================================

function checkHealth() {
  const dot   = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  fetch(`${BACKEND_URL}/health`)
    .then(r => r.json())
    .then(() => {
      if (dot)   dot.className   = "status-dot online";
      if (label) label.textContent = "Backend online";
    })
    .catch(() => {
      if (dot)   dot.className   = "status-dot offline";
      if (label) label.textContent = "Backend offline";
    });
}

// ================================================================
//  INIT
// ================================================================

window.addEventListener("load", () => {
  const saved = loadSavedProfile();
  if (saved) {
    // Populate form fields with saved values in case they edit later
    document.getElementById("pYear").value     = saved.year     || "";
    document.getElementById("pSemester").value = saved.semester || "";
    document.getElementById("pBranch").value   = saved.branch   || "";
    document.getElementById("pName").value     = saved.name     || "";
    applyProfile(saved);
  }
  checkHealth();
  setInterval(checkHealth, 30000);  // re-check every 30 seconds
});

// ================================================================
//  EDIT PROFILE override — clear active conv tracker
// ================================================================
