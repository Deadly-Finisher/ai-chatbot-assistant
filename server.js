/**
 * RAG Agentic AI Chatbot Server
 * Features: RAG, Infinite Persistent Memory, Self-Learning, Anti-Hallucination
 * Based on: musix-amrita1611/ai-chatbot-assistant (Gemini AI + Space UI)
 */



require('dotenv').config();

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs-extra');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { tavily } = require("@tavily/core");

const {
  MemoryRepository,
  SessionRepository,
  KnowledgeRepository,
  FeedbackRepository,
} = require('./db/repositories');


const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(express.static('.'));

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const genAI = new GoogleGenerativeAI(
  GEMINI_API_KEY
);

const MODEL = 'gemini-2.5-flash';

const tavilyClient = tavily({
  apiKey: process.env.TAVILY_API_KEY,
});
// JSON persistence removed in favor of SQLite (db/*)


// ─────────────────────────────────────────────
// Persistent store implemented via SQLite (db/*)
// ─────────────────────────────────────────────


// ─────────────────────────────────────────────
// SIMPLE TF-IDF VECTOR ENGINE (no external deps)
// ─────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2);
}

function tfidf(text, corpus) {
  const terms = tokenize(text);
  const tf = {};
  terms.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
  Object.keys(tf).forEach(t => { tf[t] /= terms.length; });

  const N = corpus.length + 1;
  const idf = {};
  Object.keys(tf).forEach(term => {
    const df = corpus.filter(doc => tokenize(doc.text || doc.content || '').includes(term)).length + 1;
    idf[term] = Math.log(N / df);
  });

  const vec = {};
  Object.keys(tf).forEach(t => { vec[t] = tf[t] * (idf[t] || 1); });
  return vec;
}

function cosineSimilarity(vecA, vecB) {
  const keys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
  let dot = 0, normA = 0, normB = 0;
  keys.forEach(k => {
    const a = vecA[k] || 0, b = vecB[k] || 0;
    dot += a * b; normA += a * a; normB += b * b;
  });
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ─────────────────────────────────────────────
// MEMORY STORE
// ─────────────────────────────────────────────
class MemoryStore {
  constructor() {
    this.repo = new MemoryRepository();
  }

  add(entry) {
    // Keep same shape used elsewhere
    return this.repo.add({
      sessionId: entry.sessionId,
      role: entry.role,
      text: entry.text,
      metadata: entry.metadata || {},
    });
  }

  // Retrieve top-k relevant memories using TF-IDF cosine sim (same behavior, but data comes from SQLite)
  retrieve(query, topK = 8) {
    const memories = this.repo.all();
    if (memories.length === 0) return [];

    const corpus = memories;
    const qVec = tfidf(query, corpus);

    return corpus
      .filter(m => m.role === 'user' || m.role === 'fact')
      .map(m => ({
        ...m,
        metadata: m.metadata || {},
        score: cosineSimilarity(qVec, tfidf(m.text, corpus))
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  // Extract and store facts from a message
  extractFacts(text, sessionId) {
    const factPatterns = [
      /my name is (.+)/i,
      /i am (.+)/i,
      /i'm (.+)/i,
      /i live in (.+)/i,
      /i work (?:at|for|as) (.+)/i,
      /i have (.+)/i,
      /i like (.+)/i,
      /i prefer (.+)/i,
      /remember that (.+)/i,
      /note that (.+)/i,
      /important[: ]+(.+)/i,
    ];

    factPatterns.forEach(pattern => {
      const match = text.match(pattern);
      if (match) {
        this.add({
          sessionId,
          role: 'fact',
          text: text.trim(),
          metadata: { extracted: match[1] },
        });
      }
    });
  }

  getStats() {
    return this.repo.stats();
  }
}


// ─────────────────────────────────────────────
// KNOWLEDGE BASE (self-learning store)
// ─────────────────────────────────────────────
class KnowledgeBase {
  constructor() {
    this.repo = new KnowledgeRepository();
  }

  // Add a learned piece of knowledge (from corrections/feedback)
  // NOTE: to keep behavior close to original, we still do similarity checks in-memory.
  learn(question, answer, confidence = 1.0) {
    const entries = this.repo.all();

    const existing = entries.find(e =>
      cosineSimilarity(tfidf(e.question, entries), tfidf(question, entries)) > 0.85
    );

    if (existing) {
      this.repo.update(existing.id, {
        answer,
        confidence: Math.min(existing.confidence + 0.1, 1.0),
      });
    } else {
      this.repo.learn(question, answer, confidence);
    }
  }

  retrieve(query, topK = 3) {
    const entries = this.repo.all();
    if (entries.length === 0) return [];

    const corpus = entries.map(e => ({ text: e.question + ' ' + e.answer }));
    const qVec = tfidf(query, corpus);

    return entries
      .map(e => ({
        ...e,
        score: cosineSimilarity(qVec, tfidf(e.question + ' ' + e.answer, corpus)),
      }))
      .sort((a, b) => b.score - a.score)
      .filter(e => e.score > 0.1)
      .slice(0, topK);
  }

  incrementUsage(id) {
    this.repo.incrementUsage(id);
  }
}


// ─────────────────────────────────────────────
// SESSION STORE
// ─────────────────────────────────────────────
class SessionStore {
  constructor() {
    this.repo = new SessionRepository();
  }

  createSession(title = 'New Mission') {
    return this.repo.createSession(title);
  }

  addMessage(sessionId, role, content) {
    return this.repo.addMessage(sessionId, role, content);
  }

  getSession(id) {
    return this.repo.getSession(id);
  }

  getAllSessions() {
    return this.repo.getAllSessions();
  }

  getRecentMessages(sessionId, limit = 10) {
    return this.repo.getRecentMessages(sessionId, limit);
  }
}


// ─────────────────────────────────────────────
// RAG AGENT
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// WEB SEARCH DETECTION
// ─────────────────────────────────────────────

function needsWebSearch(query) {

  const realtimePatterns = [
    /latest/i,
    /today/i,
    /current/i,
    /news/i,
    /recent/i,
    /weather/i,
    /price/i,
    /stock/i,
    /live/i,
    /score/i,
    /2026/i,
    /internet/i,
    /search/i,
    /who won/i
  ];

  return realtimePatterns.some(pattern =>
    pattern.test(query)
  );
}


// ─────────────────────────────────────────────
// RAG AGENT
// ─────────────────────────────────────────────

class RAGAgent {

  constructor() {
    this.memory = new MemoryStore();
    this.knowledge = new KnowledgeBase();
    this.sessions = new SessionStore();
    this.model = genAI.getGenerativeModel({ model: MODEL });
  }

  buildSystemPrompt(query, sessionId) {

    // 1. Retrieve relevant long-term memories
    const relevantMemories = this.memory.retrieve(query, 8);

    // 2. Retrieve from knowledge base
    const knowledgeHits = this.knowledge.retrieve(query, 3);

    let contextBlock = '';

    if (relevantMemories.length > 0) {

      contextBlock += `\n### LONG-TERM MEMORY (from previous conversations):\n`;

      relevantMemories.forEach(m => {

        contextBlock += `- [${new Date(m.timestamp).toLocaleDateString()}] ${m.text}\n`;

      });
    }

    if (knowledgeHits.length > 0) {

      contextBlock += `\n### LEARNED KNOWLEDGE BASE:\n`;

      knowledgeHits.forEach(k => {

        contextBlock += `Q: ${k.question}\nA: ${k.answer}\n`;

        this.knowledge.incrementUsage(k.id);

      });
    }

    return `You are an intelligent AI assistant with RAG (Retrieval-Augmented Generation) capabilities.

STRICT RULES TO PREVENT HALLUCINATION:

1. ONLY answer based on the context provided below or well-established facts you are certain about.

2. If you don't know something or it's not in context, say "I don't have information about that" — never make up facts.

3. If the user corrects you, accept the correction, apologize briefly, and update your understanding.

4. Always cite when you're using memory: say "(from memory)" when referencing past conversations.

5. Be honest about uncertainty: use phrases like "I believe...", "Based on what I know...", "I'm not certain but..."

PERSONA:

- You are friendly, direct, and helpful
- You remember everything the user has told you across ALL sessions
- You learn from corrections and feedback
- You keep a space/cosmic theme in your personality

${contextBlock ? `\n---\nRETRIEVED CONTEXT:\n${contextBlock}\n---\n` : ''}

Current date/time: ${new Date().toLocaleString()}

Memory stats: ${JSON.stringify(this.memory.getStats())}`;
  }


  async chat(sessionId, userMessage) {

    // Store user message in memory
    this.memory.add({
      sessionId,
      role: 'user',
      text: userMessage
    });

    this.memory.extractFacts(userMessage, sessionId);

    this.sessions.addMessage(
      sessionId,
      'user',
      userMessage
    );

    // Build full chat history for Gemini
    const recentMsgs =
      this.sessions.getRecentMessages(sessionId, 12);

    const systemPrompt =
      this.buildSystemPrompt(userMessage, sessionId);

    // ─────────────────────────────────────────────
    // REALTIME INTERNET SEARCH
    // ─────────────────────────────────────────────

    let webContext = '';

    if (needsWebSearch(userMessage)) {

      try {

        console.log('🌐 Performing realtime web search...');


        const searchResults = await tavilyClient.search(
          userMessage,
          {
            maxResults: 5
          }
        );

        webContext = `
    

### REALTIME INTERNET SEARCH RESULTS:

${JSON.stringify(searchResults)}

Use these search results as factual grounding.

Do not hallucinate beyond these results.
`;

      } catch (err) {

        console.error('Web search failed:', err);

        webContext = `
Realtime web search currently unavailable.
`;

      }
    }

    // Convert to Gemini format
    const history = recentMsgs
      .slice(0, -1)
      .filter(m =>
        m.role === 'user' ||
        m.role === 'assistant'
      )
      .map(m => ({
        role:
          m.role === 'user'
            ? 'user'
            : 'model',

        parts: [{
          text: m.content
        }]
      }));


    // Gemini requires first role to be user
    if (
      history.length > 0 &&
      history[0].role !== 'user'
    ) {
      history.shift();
    }


    const chat = this.model.startChat({

      history,

      systemInstruction: {
        role: 'system',
        parts: [{
          text: systemPrompt + webContext
        }]
      },

      generationConfig: {
        temperature: 0.7,
        topP: 0.9,
        maxOutputTokens: 1500,
      }
    });


    // ─────────────────────────────────────────────
    // RETRY LOGIC
    // ─────────────────────────────────────────────

    let result;
    let retries = 3;

    while (retries > 0) {

      try {

        result =
          await chat.sendMessage(userMessage);

        break;

      } catch (err) {

        if (err.message.includes('503')) {

          console.log(
            '⚠️ Gemini overloaded. Retrying...'
          );

          await new Promise(resolve =>
            setTimeout(resolve, 3000)
          );

          retries--;

        } else {

          throw err;

        }
      }
    }

    if (!result) {

      throw new Error(
        'Gemini API unavailable after retries'
      );
    }

    const responseText =
      result.response.text();


    // Store assistant response
    this.memory.add({
      sessionId,
      role: 'assistant',
      text: responseText
    });

    this.sessions.addMessage(
      sessionId,
      'assistant',
      responseText
    );


    // Self-learning
    if (
      userMessage.includes('?') &&
      responseText.length > 20
    ) {

      this.knowledge.learn(
        userMessage,
        responseText,
        0.8
      );
    }

    return {

      response: responseText,

      memoryUsed:
        this.memory.retrieve(userMessage, 3).length > 0,

      knowledgeUsed:
        this.knowledge.retrieve(userMessage, 1).length > 0,

      memoryStats:
        this.memory.getStats()
    };
  }


  // Self-learning from user feedback/correction
  async learnFromFeedback(
    sessionId,
    question,
    correction
  ) {

    this.knowledge.learn(
      question,
      correction,
      1.0
    );

    this.memory.add({

      sessionId,

      role: 'fact',

      text: `CORRECTION: ${question} → ${correction}`,

      metadata: {
        type: 'correction'
      }
    });

    const feedbackRepo =
      new FeedbackRepository();

    feedbackRepo.add({
      sessionId,
      question,
      correction
    });

    return {
      learned: true
    };
  }
}

// ─────────────────────────────────────────────
// INITIALIZE AGENT
// ─────────────────────────────────────────────
const agent = new RAGAgent();

// ─────────────────────────────────────────────
// API ROUTES
// ─────────────────────────────────────────────

// Create new session
app.post('/api/session', (req, res) => {
  const session = agent.sessions.createSession(req.body.title);
  res.json({ session });
});

// Get all sessions
app.get('/api/sessions', (req, res) => {
  res.json({ sessions: agent.sessions.getAllSessions() });
});

// Get session messages
app.get('/api/session/:id', (req, res) => {
  const session = agent.sessions.getSession(req.params.id);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  res.json({ session });
});

// Main chat endpoint
app.post('/api/chat', async (req, res) => {
  const { sessionId, message } = req.body;
  if (!message || !sessionId) return res.status(400).json({ error: 'sessionId and message required' });

  try {
    const result = await agent.chat(sessionId, message);
    res.json(result);
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Feedback / correction endpoint (self-learning)
app.post('/api/feedback', async (req, res) => {
  const { sessionId, question, correction } = req.body;
  if (!question || !correction) return res.status(400).json({ error: 'question and correction required' });
  const result = await agent.learnFromFeedback(sessionId, question, correction);
  res.json(result);
});

// Memory stats
app.get('/api/memory/stats', (req, res) => {
  res.json(agent.memory.getStats());
});

// Search memory
app.post('/api/memory/search', (req, res) => {
  const { query, topK } = req.body;
  const results = agent.memory.retrieve(query, topK || 5);
  res.json({ results });
});

// Upload knowledge (manual RAG document injection)
app.post('/api/knowledge/upload', (req, res) => {
  const { entries } = req.body; // array of { question, answer }
  if (!Array.isArray(entries)) return res.status(400).json({ error: 'entries array required' });
  entries.forEach(e => agent.knowledge.learn(e.question, e.answer, 1.0));
  res.json({ learned: entries.length });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', memoryStats: agent.memory.getStats() });
});

// ─────────────────────────────────────────────
// START SERVER
// ─────────────────────────────────────────────
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`\n🚀 RAG Agentic AI Chatbot running at http://localhost:${PORT}`);
  console.log(`📊 Memory stats: ${JSON.stringify(agent.memory.getStats())}`);
  console.log(`\nMake sure GEMINI_API_KEY is set in your environment or in server.js\n`);
});
