/**
 * db/repositories.js
 * Implements MemoryRepository, SessionRepository, KnowledgeRepository, FeedbackRepository
 * using JSON file persistence (fs-extra). Drop-in replacement for SQLite layer.
 * No native compilation required.
 */

const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const DATA_DIR = path.join(__dirname, '..', 'data');
fs.ensureDirSync(DATA_DIR);

// ── Generic JSON store ──────────────────────────────────────────────────────
function storeFile(name) {
  return path.join(DATA_DIR, `${name}.json`);
}

function loadStore(name, def) {
  const file = storeFile(name);
  try {
    return fs.existsSync(file) ? fs.readJSONSync(file) : def;
  } catch {
    return def;
  }
}

function saveStore(name, data) {
  fs.writeJSONSync(storeFile(name), data, { spaces: 2 });
}

// ── MemoryRepository ────────────────────────────────────────────────────────
class MemoryRepository {
  constructor() {
    this._name = 'memory';
    const d = loadStore(this._name, { memories: [] });
    this._data = d.memories || [];
  }

  _save() {
    saveStore(this._name, { memories: this._data, lastUpdated: new Date().toISOString() });
  }

  add({ sessionId, role, text, metadata = {} }) {
    const mem = {
      id: uuidv4(),
      sessionId,
      role,
      text,
      metadata,
      timestamp: new Date().toISOString(),
    };
    this._data.push(mem);
    this._save();
    return mem;
  }

  all() {
    return this._data;
  }

  stats() {
    return {
      total: this._data.length,
      facts: this._data.filter(m => m.role === 'fact').length,
      sessions: [...new Set(this._data.map(m => m.sessionId))].length,
    };
  }
}

// ── SessionRepository ───────────────────────────────────────────────────────
class SessionRepository {
  constructor() {
    this._name = 'sessions';
    const d = loadStore(this._name, { sessions: {} });
    this._data = d.sessions || {};
  }

  _save() {
    saveStore(this._name, { sessions: this._data });
  }

  createSession(title = 'New Mission') {
    const id = uuidv4();
    this._data[id] = {
      id,
      title,
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    this._save();
    return this._data[id];
  }

  addMessage(sessionId, role, content) {
    if (!this._data[sessionId]) this.createSession();
    const msg = { role, content, timestamp: new Date().toISOString() };
    this._data[sessionId].messages.push(msg);
    this._data[sessionId].updatedAt = new Date().toISOString();

    // Auto-title from first user message
    const userMsgs = this._data[sessionId].messages.filter(m => m.role === 'user');
    if (role === 'user' && userMsgs.length === 1) {
      this._data[sessionId].title = content.substring(0, 40) + (content.length > 40 ? '...' : '');
    }

    this._save();
    return msg;
  }

  getSession(id) {
    return this._data[id] || null;
  }

  getAllSessions() {
    return Object.values(this._data)
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
  }

  getRecentMessages(sessionId, limit = 10) {
    const session = this._data[sessionId];
    if (!session) return [];
    return session.messages.slice(-limit);
  }
}

// ── KnowledgeRepository ─────────────────────────────────────────────────────
class KnowledgeRepository {
  constructor() {
    this._name = 'knowledge';
    const d = loadStore(this._name, { entries: [] });
    this._data = d.entries || [];
  }

  _save() {
    saveStore(this._name, { entries: this._data });
  }

  learn(question, answer, confidence = 1.0) {
    const entry = {
      id: uuidv4(),
      question,
      answer,
      confidence,
      usageCount: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    this._data.push(entry);
    this._save();
    return entry;
  }

  update(id, { answer, confidence }) {
    const e = this._data.find(e => e.id === id);
    if (e) {
      if (answer !== undefined) e.answer = answer;
      if (confidence !== undefined) e.confidence = confidence;
      e.updatedAt = new Date().toISOString();
      this._save();
    }
  }

  all() {
    return this._data;
  }

  incrementUsage(id) {
    const e = this._data.find(e => e.id === id);
    if (e) { e.usageCount = (e.usageCount || 0) + 1; this._save(); }
  }
}

// ── FeedbackRepository ──────────────────────────────────────────────────────
class FeedbackRepository {
  constructor() {
    this._name = 'feedback';
    const d = loadStore(this._name, { items: [] });
    this._data = d.items || [];
  }

  _save() {
    saveStore(this._name, { items: this._data });
  }

  add({ sessionId, question, correction }) {
    const item = {
      id: uuidv4(),
      sessionId,
      question,
      correction,
      timestamp: new Date().toISOString(),
    };
    this._data.push(item);
    this._save();
    return item;
  }

  all() {
    return this._data;
  }
}

module.exports = {
  MemoryRepository,
  SessionRepository,
  KnowledgeRepository,
  FeedbackRepository,
};
