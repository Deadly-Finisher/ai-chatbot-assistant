PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  sessionId TEXT NOT NULL,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  metadataJson TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_session_role ON memories(sessionId, role);
CREATE INDEX IF NOT EXISTS idx_memories_role_timestamp ON memories(role, timestamp);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  createdAt TEXT NOT NULL,
  updatedAt TEXT NOT NULL,
  messagesJson TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_updatedAt ON sessions(updatedAt);

CREATE TABLE IF NOT EXISTS knowledge (
  id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  confidence REAL NOT NULL,
  usageCount INTEGER NOT NULL,
  createdAt TEXT NOT NULL,
  updatedAt TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_usageCount ON knowledge(usageCount);

CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  sessionId TEXT NOT NULL,
  question TEXT NOT NULL,
  correction TEXT NOT NULL,
  timestamp TEXT NOT NULL
);

