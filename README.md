# 🚀 RAG Agentic AI Chatbot — Infinite Memory Edition

> Built on top of [musix-amrita1611/ai-chatbot-assistant](https://github.com/musix-amrita1611/ai-chatbot-assistant)  
> Powered by Google Gemini AI · Space-themed UI · RAG + Infinite Memory + Self-Learning

---

## ✨ Features Added

| Feature | Description |
|---|---|
| 🧠 **Infinite Memory** | Remembers everything across ALL sessions, forever (stored in `data/memory.json`) |
| 📚 **RAG Engine** | TF-IDF vector retrieval — answers are grounded in stored context, not hallucinated |
| 🤖 **Agentic Reasoning** | Retrieves relevant memories before answering each question |
| 🎓 **Self-Learning** | Use the "Teach the AI" panel to correct wrong answers — it learns permanently |
| 🚫 **Anti-Hallucination** | Strict system prompt forces the AI to cite uncertainty and never invent facts |
| 💾 **Persistent Sessions** | All conversations saved to disk, resumable anytime |
| ⚡ **Fact Extraction** | Automatically extracts and stores facts from your messages ("My name is...", "I work at...") |

---

## 📁 File Structure

```
ai-chatbot-assistant/
├── server.js          ← Main server (RAG engine, memory, knowledge base)
├── index.html         ← Space-themed UI (original layout preserved)
├── style.css          ← All styles (original + RAG UI)
├── script.js          ← Frontend logic (session mgmt, chat, feedback)
├── package.json       ← Dependencies
├── .env.example       ← Environment variable template
└── data/              ← Auto-created on first run
    ├── memory.json    ← All conversation memories (infinite)
    ├── sessions.json  ← Session history
    ├── knowledge.json ← Self-learned knowledge base
    └── feedback.json  ← User corrections log
```

---

## 🛠️ Setup — Step by Step

### Step 1: Clone and enter the project

```bash
git clone https://github.com/musix-amrita1611/ai-chatbot-assistant.git
cd ai-chatbot-assistant
```

### Step 2: Replace all files

Copy over the new `server.js`, `script.js`, `style.css`, `index.html`, and `package.json` from this release into the project folder.

### Step 3: Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key

### Step 4: Set your API key

**Option A — Environment variable (recommended):**
```bash
# Linux / Mac
export GEMINI_API_KEY=your_key_here

# Windows CMD
set GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

**Option B — Edit server.js directly:**

Open `server.js` and replace line:
```js
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || 'YOUR_GEMINI_API_KEY_HERE';
```
with:
```js
const GEMINI_API_KEY = 'paste_your_key_here';
```

### Step 5: Install dependencies

```bash
npm install
```

### Step 6: Start the server

```bash
npm start
```

You'll see:
```
🚀 RAG Agentic AI Chatbot running at http://localhost:3000
📊 Memory stats: {"total":0,"facts":0,"sessions":0}
```

### Step 7: Open the chatbot

Open your browser and go to: **http://localhost:3000**

---

## 🎮 How to Use

### Basic Chat
- Type a message in the input and press **Enter** or click **TRANSMIT**
- The AI will retrieve relevant memories before answering

### Memory Indicators
- 🧠 **MEM** badge = answer used long-term memory from past sessions
- 📚 **RAG** badge = answer used the learned knowledge base

### Self-Learning (Teach the AI)
In the sidebar at the bottom:
1. Enter the wrong question/topic in the first field
2. Enter the correct answer in the second field
3. Click **SUBMIT CORRECTION**

The AI permanently stores this and uses it in future responses.

### Multiple Sessions
- Click **✚ NEW MISSION** to start a fresh conversation
- Previous sessions appear in the sidebar — click any to resume
- All sessions share the same memory pool

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/session` | Create new session |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/session/:id` | Get session with messages |
| POST | `/api/chat` | Send a message |
| POST | `/api/feedback` | Submit a correction (self-learning) |
| GET | `/api/memory/stats` | Memory statistics |
| POST | `/api/memory/search` | Search memories by query |
| POST | `/api/knowledge/upload` | Bulk-upload knowledge entries |

### Example: Inject custom knowledge
```bash
curl -X POST http://localhost:3000/api/knowledge/upload \
  -H "Content-Type: application/json" \
  -d '{
    "entries": [
      {"question": "What is our product?", "answer": "AcmeCorp sells AI-powered widgets."},
      {"question": "Support email?", "answer": "support@acmecorp.com"}
    ]
  }'
```

---

## 🔧 Development Mode (auto-restart)

```bash
npm run dev
```

Requires `nodemon` (installed automatically via `npm install`).

---

## 🧹 Reset Memory

To wipe all memory and start fresh:
```bash
rm -rf data/
```

The `data/` folder is recreated automatically on next start.

---

## 📦 Dependencies

- `@google/generative-ai` — Gemini AI SDK
- `express` — Web server
- `fs-extra` — File system helpers
- `uuid` — Unique IDs
- `natural` — NLP utilities
- `cors` + `body-parser` — API middleware
