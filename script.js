/**
 * RAG AI Chatbot — Frontend Script
 * Infinite memory, self-learning, anti-hallucination
 */

const API = 'http://localhost:5000/api';

// ── State ──────────────────────────────────────
let currentSessionId = null;

let authToken =
  localStorage.getItem('token') || null;

// ── DOM ────────────────────────────────────────
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const sessionList = document.getElementById('session-list');
const statusText = document.getElementById('status-text');
const memCount = document.getElementById('mem-count');
const sessionCount = document.getElementById('session-count');
const tagMemory = document.getElementById('tag-memory');
const tagKnowledge = document.getElementById('tag-knowledge');
const feedbackBtn = document.getElementById('feedback-btn');
const feedbackQ = document.getElementById('feedback-q');
const feedbackA = document.getElementById('feedback-a');
const uploadPdfBtn =
  document.getElementById(
    'upload-pdf-btn'
  );

const pdfUploadInput =
  document.getElementById(
    'pdf-upload'
  );

const authModal =
  document.getElementById('auth-modal');

const authUsername =
  document.getElementById('auth-username');

const authPassword =
  document.getElementById('auth-password');

const loginBtn =
  document.getElementById('login-btn');

const registerBtn =
  document.getElementById('register-btn');

const authStatus =
  document.getElementById('auth-status');

const logoutBtn =
  document.getElementById('logout-btn');

const generateNotesBtn =
  document.getElementById(
    'generate-notes-btn'
  );

const generatePlanBtn =
  document.getElementById(
    'generate-plan-btn'
  );

// ─────────────────────────────────────────────
// PDF Upload
// ─────────────────────────────────────────────

uploadPdfBtn.addEventListener(
  'click',
  () => {

    pdfUploadInput.click();
  }
);

pdfUploadInput.addEventListener(
  'change',
  async (e) => {

    const file =
      e.target.files[0];

    if (!file) return;

    const formData =
      new FormData();

    formData.append(
      'pdf',
      file
    );

    try {

      appendMessage(
        'assistant',
        `📄 Uploading ${file.name}...`
      );

      const res =
        await fetch(
          `${API}/pdf/upload`,
          {
            method: 'POST',

            headers: {
              'Authorization':
                `Bearer ${authToken}`
            },

            body: formData
          }
        );

      const data =
        await res.json();

      if (data.success) {

        appendMessage(
          'assistant',
          `✅ PDF uploaded successfully!\n\nDocument: ${data.filename}\nChunks created: ${data.chunks}`
        );

      } else {

        appendMessage(
          'assistant',
          `❌ Upload failed: ${data.error}`
        );
      }

    } catch (err) {

      console.error(err);

      appendMessage(
        'assistant',
        '❌ PDF upload failed'
      );
    }
  }
);

// ─────────────────────────────────────────────
// SMART NOTES GENERATION
// ─────────────────────────────────────────────

generateNotesBtn.addEventListener(
  'click',

  async () => {

    try {

      appendMessage(
        'assistant',
        '📝 Generating smart study notes from uploaded PDFs...'
      );

      const res =
        await fetch(
          `${API}/notes/generate`,
          {
            method: 'POST',

            headers: {
              'Authorization':
                `Bearer ${authToken}`,
              'Content-Type':
                'application/json'
            }
          }
        );

      const data =
        await res.json();

      if (data.success) {

        appendMessage(
          'assistant',
          data.notes
        );

      } else {

        appendMessage(
          'assistant',
          `❌ ${data.error}`
        );
      }

    } catch (err) {

      console.error(err);

      appendMessage(
        'assistant',
        '❌ Failed to generate notes'
      );
    }
  }
);

// ─────────────────────────────────────────────
// STUDY PLAN GENERATION
// ─────────────────────────────────────────────

generatePlanBtn.addEventListener(
  'click',

  async () => {

    try {

      appendMessage(
        'assistant',
        '📅 Generating AI study plan from uploaded PDFs...'
      );

      const res =
        await fetch(
          `${API}/planner/generate`,
          {
            method: 'POST',

            headers: {
              'Authorization':
                `Bearer ${authToken}`,
              'Content-Type':
                'application/json'
            }
          }
        );

      const data =
        await res.json();

      if (data.success) {

        appendMessage(
          'assistant',
          data.plan
        );

      } else {

        appendMessage(
          'assistant',
          `❌ ${data.error}`
        );
      }

    } catch (err) {

      console.error(err);

      appendMessage(
        'assistant',
        '❌ Failed to generate study plan'
      );
    }
  }
);

// ── Clock ──────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('clock-time').textContent = now.toLocaleTimeString();
  document.getElementById('clock-date').textContent = now.toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'short', day: 'numeric'
  }).toUpperCase();
}
setInterval(updateClock, 1000);
updateClock();

// ── Session Management ─────────────────────────
async function loadSessions() {
  try {
    const res = await fetch(
      `${API}/sessions`,
      {
        headers: {
          'Authorization':
            `Bearer ${authToken}`
        }
      }
    );
    const { sessions } = await res.json();
    sessionList.innerHTML = '';
    sessions.forEach(s => {
      const el = document.createElement('div');
      el.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
      el.innerHTML = `<div class="session-title">${escapeHtml(s.title)}</div>
        <div class="session-meta">${new Date(s.updatedAt).toLocaleDateString()}</div>`;
      el.onclick = () => loadSession(s.id);
      sessionList.appendChild(el);
    });
  } catch (e) { console.error('Load sessions error', e); }
}

async function createSession() {
  const res = await fetch(`${API}/session`, {
    method: 'POST', headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
    body: JSON.stringify({ title: 'New Mission' })
  });
  const { session } = await res.json();
  currentSessionId = session.id;
  chatBox.innerHTML = '';
  appendMessage('assistant', '🚀 New mission initiated. I have access to all previous memories. How can I assist you today?');
  await loadSessions();
}

async function loadSession(sessionId) {
  currentSessionId = sessionId;
  const res = await fetch(
    `${API}/session/${sessionId}`,
    {
      headers: {
        'Authorization':
          `Bearer ${authToken}`
      }
    }
  );
  const { session } = await res.json();
  chatBox.innerHTML = '';
  session.messages.forEach(m => appendMessage(m.role, m.content, false));
  chatBox.scrollTop = chatBox.scrollHeight;
  await loadSessions();
}

// ── Memory Stats ───────────────────────────────
async function refreshMemoryStats() {
  try {
    const res = await fetch(`${API}/memory/stats`);
    const stats = await res.json();
    memCount.textContent = stats.total;
    sessionCount.textContent = stats.sessions;
  } catch {}
}

// ── Chat ───────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || !currentSessionId) return;

  userInput.value = '';
  appendMessage('user', text);
  setStatus('TRANSMITTING...', true);

  // Hide rag tags
  tagMemory.style.display = 'none';
  tagKnowledge.style.display = 'none';

  const typingEl = appendTyping();

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
      },
      body: JSON.stringify({ sessionId: currentSessionId, message: text })
    });
    const data = await res.json();
    typingEl.remove();

    if (data.error) {
      appendMessage('assistant', `⚠️ Error: ${data.error}`);
      setStatus('ERROR', false);
    } else {
      let finalResponse =
        data.response;

      // ─────────────────────────────
      // ADD CITATIONS
      // ─────────────────────────────

      if (
        data.sources &&
        data.sources.length > 0
      ) {

        finalResponse +=
          `\n\n📚 Sources Used:\n`;

        data.sources.forEach(
          source => {

            finalResponse +=
              `• ${source}\n`;
          }
        );
      }

      appendMessage(
        'assistant',
        finalResponse,
        true,
        {
          memoryUsed:
            data.memoryUsed,

          knowledgeUsed:
            data.knowledgeUsed
        }
      );
      if (data.memoryUsed) tagMemory.style.display = 'inline-block';
      if (data.knowledgeUsed) tagKnowledge.style.display = 'inline-block';
      setStatus('SYSTEM READY', false);
      await refreshMemoryStats();
      await loadSessions();
    }
  } catch (err) {
    typingEl.remove();
    appendMessage('assistant', '⚠️ Connection failed. Is the server running on port 5000?');
    setStatus('DISCONNECTED', false);
  }
}

// ── DOM Helpers ────────────────────────────────
function appendMessage(role, content, animate = true, meta = {}) {
  const el = document.createElement('div');
  el.className = `message ${role}${animate ? ' appear' : ''}`;

  let badge = '';
  if (meta.memoryUsed) badge += '<span class="msg-badge mem-badge">🧠 memory</span>';
  if (meta.knowledgeUsed) badge += '<span class="msg-badge rag-badge">📚 rag</span>';

  el.innerHTML = `
    <div class="msg-content">${formatMessage(content)}</div>
    ${badge ? `<div class="msg-badges">${badge}</div>` : ''}
  `;
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
  return el;
}

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'message assistant typing-indicator';
  el.innerHTML = `<div class="msg-content"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
  return el;
}

function formatMessage(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function setStatus(msg, busy) {
  statusText.textContent = msg;
  document.querySelector('.pulse-container').style.opacity = busy ? '1' : '0.5';
}

// ── Feedback / Self-Learning ───────────────────
feedbackBtn.addEventListener('click', async () => {
  const q = feedbackQ.value.trim();
  const a = feedbackA.value.trim();
  if (!q || !a) { alert('Fill both fields'); return; }

  feedbackBtn.textContent = 'LEARNING...';
  await fetch(`${API}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: currentSessionId, question: q, correction: a })
  });

  feedbackQ.value = '';
  feedbackA.value = '';
  feedbackBtn.textContent = '✓ LEARNED!';
  setTimeout(() => { feedbackBtn.textContent = 'SUBMIT CORRECTION'; }, 2000);
  appendMessage('assistant', `✅ Got it! I've learned: "${a}" — I'll remember this for all future conversations.`);
});

// ── Event Listeners ────────────────────────────
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener(
  'keydown',
  e => {

    // ENTER → SEND
    if (
      e.key === 'Enter' &&
      !e.shiftKey
    ) {

      e.preventDefault();

      sendMessage();
    }

    // SHIFT + ENTER → NEW LINE
    if (
      e.key === 'Enter' &&
      e.shiftKey
    ) {

      const start =
        userInput.selectionStart;

      const end =
        userInput.selectionEnd;

      userInput.value =
        userInput.value.substring(
          0,
          start
        ) +
        '\n' +
        userInput.value.substring(
          end
        );

      userInput.selectionStart =
        userInput.selectionEnd =
        start + 1;

      e.preventDefault();
    }
  }
);


logoutBtn.addEventListener(
  'click',
  () => {

    localStorage.removeItem(
      'token'
    );

    authToken = null;

    location.reload();
  }
);

// ── AUTH ──────────────────────────────────────

registerBtn.addEventListener('click', async () => {

  const username =
    authUsername.value.trim();

  const password =
    authPassword.value.trim();

  if (!username || !password) {

    authStatus.textContent =
      'Enter username and password';

    return;
  }

  authStatus.textContent =
    'Creating account...';

  try {

    const res = await fetch(
      `${API}/auth/register`,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json'
        },

        body: JSON.stringify({
          username,
          password
        })
      }
    );

    const data = await res.json();

    if (data.error) {

      authStatus.textContent =
        data.error;

    } else {

      authStatus.textContent =
        '✅ Registration successful';

    }

  } catch (err) {

    authStatus.textContent =
      '⚠️ Registration failed';
  }
});



loginBtn.addEventListener('click', async () => {

  const username =
    authUsername.value.trim();

  const password =
    authPassword.value.trim();

  if (!username || !password) {

    authStatus.textContent =
      'Enter username and password';

    return;
  }

  authStatus.textContent =
    'Logging in...';

  try {

    const res = await fetch(
      `${API}/auth/login`,
      {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json'
        },

        body: JSON.stringify({
          username,
          password
        })
      }
    );

    const data = await res.json();

    if (data.error) {

      authStatus.textContent =
        data.error;

      return;
    }

    // SAVE TOKEN
    authToken = data.token;

    localStorage.setItem(
      'token',
      data.token
    );

    // HIDE LOGIN MODAL
    authModal.style.display =
      'none';

    authStatus.textContent = '';

    // LOAD CHATBOT
    await loadSessions();

    const res2 = await fetch(
      `${API}/sessions`,
      {
        headers: {
          'Authorization':
            `Bearer ${authToken}`
        }
      }
    );

    const sessionsData =
      await res2.json();

    if (
      sessionsData.sessions.length > 0
    ) {

      await loadSession(
        sessionsData.sessions[0].id
      );

    } else {

      await createSession();
    }

  } catch (err) {

    console.error(err);

    authStatus.textContent =
      '⚠️ Login failed';
  }
});

// ── Init ───────────────────────────────────────
(async () => {

  // USER ALREADY LOGGED IN
  if (authToken) {

    authModal.style.display =
      'none';

    await loadSessions();

    await refreshMemoryStats();

    const res = await fetch(
      `${API}/sessions`,
      {
        headers: {
          'Authorization':
            `Bearer ${authToken}`
        }
      }
    );

    const { sessions } =
      await res.json();

    if (sessions.length > 0) {

      await loadSession(
        sessions[0].id
      );

    } else {

      await createSession();
    }

  }

})();
