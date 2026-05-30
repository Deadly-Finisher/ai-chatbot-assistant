/**
 * RAG Agentic AI Chatbot Server
 * Features: RAG, Infinite Persistent Memory, Self-Learning, Anti-Hallucination
 * Based on: musix-amrita1611/ai-chatbot-assistant (Gemini AI + Space UI)
 */



require('dotenv').config();

const connectDB = require('./db');
const express = require('express');
const cors = require('cors');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');
const path = require('path');
const Groq = require('groq-sdk');
const { tavily } = require("@tavily/core");
const fs = require('fs-extra');
const multer = require('multer');
const pdfParse = require('pdf-parse');
// Suppress noisy pdf.js font warnings
process.env.PDFJS_DISABLE_WARNINGS = 'true';
const PDFRepository = require('./db/PDFRepository');
const TaskRepository = require('./db/TaskRepository');




const {
  MemoryRepository,
  SessionRepository,
  KnowledgeRepository,
  FeedbackRepository,
} = require('./db/repositories');
const UserRepository =
  require('./db/MongoUserRepository');


const app = express();
const userRepo = new UserRepository();
const pdfRepo =
  new PDFRepository();
const taskRepo =
  new TaskRepository();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));
app.use(express.static('.'));

// ─────────────────────────────────────────────
// FILE UPLOAD CONFIG
// ─────────────────────────────────────────────

const storage =
  multer.diskStorage({

    destination:
      function (req, file, cb) {

        cb(null, 'uploads/');
      },

    filename:
      function (req, file, cb) {

        cb(
          null,
          Date.now() +
          '-' +
          file.originalname
        );
      }
  });

const upload =
  multer({ storage });

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────
const GROQ_API_KEY = process.env.GROQ_API_KEY;

const groq = new Groq({
  apiKey: GROQ_API_KEY,
});

const MODEL = 'llama-3.3-70b-versatile';

// ─────────────────────────────
// EMBEDDING MODEL
// ─────────────────────────────

// ─────────────────────────────
// EMBEDDING MODEL
// ─────────────────────────────

let embedder = null;

async function loadEmbeddingModel() {

  if (!embedder) {

    console.log(
      '🧠 Loading embedding model...'
    );

    const {
      pipeline
    } = await import(
      '@xenova/transformers'
    );

    embedder =
      await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2'
      );

    console.log(
      '✅ Embedding model loaded'
    );
  }

  return embedder;
}

// ─────────────────────────────
// CREATE EMBEDDING
// ─────────────────────────────

async function createEmbedding(text) {

  const model =
    await loadEmbeddingModel();

  const output =
    await model(
      text,
      {
        pooling: 'mean',
        normalize: true
      }
    );

  return Array.from(
    output.data
  );
}

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

// ─────────────────────────────
// EMBEDDING COSINE SIMILARITY
// ─────────────────────────────

function embeddingSimilarity(
  vecA,
  vecB
) {

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (
    let i = 0;
    i < vecA.length;
    i++
  ) {

    dot +=
      vecA[i] * vecB[i];

    normA +=
      vecA[i] * vecA[i];

    normB +=
      vecB[i] * vecB[i];
  }

  return (
    dot /
    (
      Math.sqrt(normA) *
      Math.sqrt(normB)
    )
  );
}

function keywordSimilarity(
  query,
  text
) {

  const queryWords =
    query
      .toLowerCase()
      .split(/\W+/)
      .filter(w => w.length > 2);

  const textWords =
    new Set(
      text
        .toLowerCase()
        .split(/\W+/)
    );

  let matches = 0;

  for (const word of queryWords) {

    if (
      textWords.has(word)
    ) {

      matches++;
    }
  }

  return (
    matches /
    Math.max(
      queryWords.length,
      1
    )
  );
}

// ─────────────────────────────────────────────
// MEMORY STORE
// ─────────────────────────────────────────────
class MemoryStore {
  constructor() {
    this.repo = new MemoryRepository();
  }

  add(entry) {

    return this.repo.add({

      userId: entry.userId || 'guest',

      sessionId: entry.sessionId,

      role: entry.role,

      text: entry.text,

      metadata: entry.metadata || {},
    });
  
  }

  // Retrieve top-k relevant memories using TF-IDF cosine sim (same behavior, but data comes from SQLite)
  retrieve(userId, query, topK = 8){
    const memories =
      this.repo
        .all()
        .filter(
          m => m.userId === userId
        );
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
  extractFacts(userId,text, sessionId) {
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
          userId,
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

  updateTitle(
    sessionId,
    title
  ) {

    return this.repo.updateTitle(
      sessionId,
      title
    );
  }

  constructor() {
    this.repo = new SessionRepository();
  }

  createSession(
    userId,
    title = 'New Mission'
  ) {

    return this.repo.createSession(
      userId,
      title
    );
  }

  addMessage(sessionId, role, content) {
    return this.repo.addMessage(sessionId, role, content);
  }

  getSession(id) {
    return this.repo.getSession(id);
  }

  getAllSessions(userId) {

    return this.repo.getAllSessions(
      userId
    );
  }

  getRecentMessages(sessionId, limit = 10) {
    return this.repo.getRecentMessages(sessionId, limit);
  }
}


// ─────────────────────────────────────────────
// RAG AGENT
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
// SMALL TALK DETECTION
// ─────────────────────────────────────────────

function isSmallTalk(query) {

  const smallTalkPatterns = [
    /^hi$/i,
    /^hello$/i,
    /^hey$/i,
    /^yo$/i,
    /^good morning$/i,
    /^good evening$/i,
    /^how are you$/i,
    /^thanks?$/i,
    /^ok$/i,
    /^okay$/i
  ];

  return smallTalkPatterns.some(
    pattern =>
      pattern.test(
        query.trim()
      )
  );
}

// ─────────────────────────────────────────────
// WEB SEARCH DETECTION
// ─────────────────────────────────────────────

function needsWebSearch(query) {

  const realtimePatterns = [
    /latest/i,
    /today/i,
    /news/i,
    /weather/i,
    /price/i,
    /stock/i,
    /live/i,
    /score/i
  ];

  return realtimePatterns.some(
    pattern =>
      pattern.test(query)
  );
}

// ─────────────────────────────
// QUERY INTENT CLASSIFIER
// ─────────────────────────────

function classifyQuery(
  userId,
  query
) {

  const q =
    query
      .trim()
      .toLowerCase();

  // Small talk
  if (isSmallTalk(q)) {

    return 'smalltalk';
  }

  // Realtime web info
  

  // PDF / study / document questions
  if (

    /\b(pdf|document|paper|summary|notes|study|chapter)\b/i
      .test(q)

  ) {

    return 'rag';
  }

  // Detect uploaded document names
  const documents =
    pdfRepo.getUserDocuments(
      userId
    );

  for (const doc of documents) {

    const filename =
      doc.filename
        .toLowerCase()
        .replace('.pdf', '');

    if (
      q.includes(filename)
    ) {

      return 'rag';
    }
  }
  if (

    /\b(today|latest|current|news|weather|price|stock|bitcoin|gold)\b/i
      .test(q)

  ) {

    return 'web';
  }

  // Task related
  if (

    /\b(task|deadline|plan|schedule|remind)\b/i
      .test(q)

  ) {

    return 'task';
  }

  // Memory related
  if (

    /\b(remember|memory|about me)\b/i
      .test(q)

  ) {

    return 'memory';
  }

  return 'general';
}


// ─────────────────────────────────────────────
// RAG AGENT
// ─────────────────────────────────────────────

class RAGAgent {

  constructor() {
    this.memory = new MemoryStore();
    this.knowledge = new KnowledgeBase();
    this.sessions = new SessionStore();
    this.model = groq;
  }

  // ─────────────────────────────
  // DETECT TARGET PDF
  // ─────────────────────────────

  detectTargetDocument(
    userId,
    query
  ) {

    const documents =
      pdfRepo.getUserDocuments(
        userId
      );

    const normalizedQuery =
      query.toLowerCase();

    // Handle "last uploaded pdf"
    if (

      normalizedQuery.includes(
        'last uploaded pdf'
      ) ||

      normalizedQuery.includes(
        'latest pdf'
      ) ||

      normalizedQuery.includes(
        'recent pdf'
      )

    ) {

      const sortedDocs =
        [...documents].sort(

          (a, b) =>

            new Date(
              b.uploadedAt
            ) -

            new Date(
              a.uploadedAt
            )
        );

      console.log(
        '🎯 Target PDF:',
        sortedDocs[0]?.filename
        );

      return sortedDocs[0];
      }

    for (const doc of documents) {

      const filename =
        doc.filename
          .toLowerCase()
          .replace('.pdf', '');

      if (
        normalizedQuery.includes(
          filename
        )
      ) {

        console.log(
          '🎯 Matched PDF:',
          doc.filename
        );

        return doc;
      }
    }

    return null;
  }

  getDocumentSummary(
    targetDocument
  ) {

    if (
      targetDocument &&
      targetDocument.summary
    ) {

      return targetDocument.summary;
    }

    return null;
  }
  async retrieveRelevantPDFChunks(
    userId,
    query,
    topK = 5
  ) {

    const documents =
      pdfRepo.getUserDocuments(
        userId
      );
    // Detect target document
    const targetDocument =
      this.detectTargetDocument(
        userId,
        query
      );

    if (!documents.length)
      return [];

    const allChunks = [];

    const docsToSearch =

      targetDocument
        ? [targetDocument]
        : documents;

    docsToSearch.forEach(doc => {

      doc.chunks.forEach(chunk => {

        allChunks.push({

          filename:
            doc.filename,

          text:
            chunk.text,

          embedding:
            chunk.embedding,

          chunkIndex:
            chunk.chunkIndex,

          relativePosition:
            chunk.relativePosition
        });
      });
        });


    // Otherwise use TF-IDF retrieval

    // ─────────────────────────────
    // EMBEDDING-BASED RETRIEVAL
    // ─────────────────────────────

    const queryEmbedding =
      await createEmbedding(
        query
      );

    const scoredChunks = [];

    for (const chunk of allChunks) {

      if (!chunk.embedding)
        continue;

      const embeddingScore =
        embeddingSimilarity(
          queryEmbedding,
          chunk.embedding
        );

      const keywordScore =
        keywordSimilarity(
          query,
          chunk.text
        );

      const score =
        (
          0.7 * embeddingScore
        ) +
        (
          0.3 * keywordScore
        );

      scoredChunks.push({

        ...chunk,

        score
      });
    }

    const topChunks =
      scoredChunks
        .sort(
          (a, b) =>
            b.score - a.score
        )
        .slice(0, topK);

    console.log(
      '\n🔍 TOP RETRIEVED CHUNKS'
    );

    topChunks.forEach(
      (chunk, index) => {

        console.log(
          `\n#${index + 1}`
        );

        console.log(
          'Score:',
          chunk.score
        );

        console.log(
          'Source:',
          chunk.filename
        );

        console.log(
          chunk.text
            .substring(0, 200)
            .replace(/\n/g, ' ')
        );
      }
    );

    return topChunks;
  }

  async buildSystemPrompt(
    userId,
    query,
    sessionId,
    pdfChunks = []
  ){

    // Detect target PDF
    const targetDocument =
      this.detectTargetDocument(
        userId,
        query
      );

    const documentSummary =
      this.getDocumentSummary(
        targetDocument
      );

    // 1. Retrieve relevant long-term memories
    const relevantMemories =
      this.memory.retrieve(
        userId,
        query,
        8
      );

    // 2. Retrieve from knowledge base
    const knowledgeHits = this.knowledge.retrieve(query, 3);

    let contextBlock = '';

    // ─────────────────────────────
    // PDF RAG Retrieval
    // ─────────────────────────────

    

    if (

      !targetDocument &&

      relevantMemories.length > 0

      ) {

      contextBlock += `\n### LONG-TERM MEMORY (from previous conversations):\n`;

      relevantMemories.forEach(m => {

        contextBlock += `- [${new Date(m.timestamp).toLocaleDateString()}] ${m.text}\n`;

      });
    }

    if (

      !targetDocument &&

      knowledgeHits.length > 0

    ) {

      contextBlock += `\n### LEARNED KNOWLEDGE BASE:\n`;

      knowledgeHits.forEach(k => {

        contextBlock += `Q: ${k.question}\nA: ${k.answer}\n`;

        this.knowledge.incrementUsage(k.id);

      });
    }

    const summaryQuery =
      /summary|summarize|overview|contribution|main contribution|what is this paper about/i;

    if (
      summaryQuery.test(query) &&
      documentSummary
    ) {

      contextBlock += `
### DOCUMENT SUMMARY

${documentSummary}

IMPORTANT:
For summary, overview, contribution,
and paper-description questions,
use the DOCUMENT SUMMARY as the primary source.
`;
    }
    if (pdfChunks.length > 0) {

      contextBlock += `
### RELEVANT PDF CONTEXT

IMPORTANT:
The answer MUST be derived from the PDF context below.
Do NOT answer from memory if PDF context exists.

`;

      pdfChunks.forEach(
        (chunk, index) => {

          contextBlock +=
            `\n[PDF ${index + 1}] (${chunk.filename})\n`;

          contextBlock +=
            `${chunk.text}\n`;
        }
      );
    }

    return `You are an intelligent AI assistant with RAG (Retrieval-Augmented Generation) capabilities.

STRICT RULES TO PREVENT HALLUCINATION:

1. If DOCUMENT SUMMARY is provided,
use it as the primary source for:

- summaries
- overviews
- contributions
- paper descriptions

2. If PDF chunks are provided,
use them for detailed factual questions.

3. Answer ONLY from the provided PDF information.

4. Give priority in this order:
   PDF Context > Web Search > Knowledge Base > Memory

5. Never use memory when answering questions about an uploaded PDF.

6. If the answer is not present in the PDF context, say:
   "I couldn't find that information in the uploaded PDF."

7. Do not guess, infer, or hallucinate missing details.

8. If the user corrects you, accept the correction, apologize briefly, and update your understanding.

9. Always cite when you're using memory: say "(from memory)" when referencing past conversations.

10. Be honest about uncertainty: use phrases like "I believe...", "Based on what I know...", "I'm not certain but..."

PERSONA:

- You are friendly, direct, and helpful
- You remember everything the user has told you across ALL sessions
- You learn from corrections and feedback
- You keep a space/cosmic theme in your personality

${contextBlock ? `\n---\nRETRIEVED CONTEXT:\n${contextBlock}\n---\n` : ''}

Current date/time: ${new Date().toLocaleString()}

Memory stats: ${JSON.stringify(this.memory.getStats())}`;
  }


  async chat(userId, sessionId, userMessage) {

    const intent =
      classifyQuery(
        userId,
        userMessage
      );

    console.log(
      '🧠 Intent:',
      intent
    );

    // ─────────────────────────────
    // FAST SMALL-TALK RESPONSES
    // ─────────────────────────────

    if (isSmallTalk(userMessage)) {

      const quickReplies = {

        hi: "Hello 👋",

        hello: "Hello 👋",

        hey: "Hey there 🚀",

        yo: "Yo 🚀",

        "good morning":
          "Good morning ☀️",

        "good evening":
          "Good evening 🌙",

        "how are you":
          "I'm functioning perfectly 🚀",

        thanks:
          "You're welcome 🚀",

        ok:
          "👍",

        okay:
          "👍"
      };

      const normalized =
        userMessage
          .trim()
          .toLowerCase();

      const response =
        quickReplies[normalized]
        || "Hello 👋";

      // Store lightweight conversation
      this.sessions.addMessage(
        sessionId,
        'user',
        userMessage
      );

      this.sessions.addMessage(
        sessionId,
        'assistant',
        response
      );

      return {

        response,

        memoryUsed: false,

        knowledgeUsed: false,

        sources: [],

        memoryStats:
          this.memory.getStats()
      };
    }

    
    // Store user message in memory
    this.memory.add({
      userId,
      sessionId,
      role: 'user',
      text: userMessage
    });

    this.memory.extractFacts(userId,userMessage, sessionId);

    this.sessions.addMessage(
      sessionId,
      'user',
      userMessage
    );



    // ─────────────────────────────
    // FAST MODE
    // ─────────────────────────────

    const fastMode =
      intent === 'web' ||
      intent === 'smalltalk';

    let recentMsgs = [];

    if (!fastMode) {

      recentMsgs =
        this.sessions.getRecentMessages(
          sessionId,
          12
        );

    } else {

      recentMsgs = [
        {
          role: 'user',
          content: userMessage
        }
      ];
      }


    // ─────────────────────────────
    // FAST PDF METADATA QUERIES
    // ─────────────────────────────

    const normalizedQuery =
      userMessage
        .trim()
        .toLowerCase();

    if (

      normalizedQuery ===
      'how many pdfs' ||

      normalizedQuery ===
      'how many pdf' ||

      normalizedQuery ===
      'how many documents' ||

      normalizedQuery ===
      'uploaded pdf'

        ){

      const documents =
        pdfRepo.getUserDocuments(
          userId
        );

      const response =
        `📚 You have uploaded ${documents.length} PDF documents.`;

      this.sessions.addMessage(
        sessionId,
        'user',
        userMessage
      );

      this.sessions.addMessage(
        sessionId,
        'assistant',
        response
      );

      return {

        response,

        memoryUsed: false,

        knowledgeUsed: false,

        sources: [],

        memoryStats:
          this.memory.getStats()
      };
      }

    

    if (
      normalizedQuery.includes('last pdf') ||
      normalizedQuery.includes('latest pdf') ||
      normalizedQuery.includes('last uploaded pdf') ||
      normalizedQuery.includes('name of the last pdf')
    ) {

      const documents =
        pdfRepo.getUserDocuments(userId);

      if (!documents.length) {

        return {
          response: 'No PDFs uploaded.',
          sources: []
        };
      }

      const latest =
        [...documents].sort(
          (a, b) =>
            new Date(b.uploadedAt) -
            new Date(a.uploadedAt)
        )[0];

      return {
        response:
          `📄 Latest PDF: ${latest.filename}`,
        sources: [latest.filename]
      };
    }

    // ─────────────────────────────
    // TRACK PDF SOURCES
    // ─────────────────────────────

    let pdfSources = [];
let pdfChunks = [];

const summaryQuery =
  /summary|summarize|overview|contribution|main contribution|what is this paper about/i;

if (
  intent === 'rag' &&
  !summaryQuery.test(userMessage)
) {

  pdfChunks =
    await this.retrieveRelevantPDFChunks(
      userId,
      userMessage,
      5
    );

  pdfSources =
    [
      ...new Set(
        pdfChunks.map(
          c => c.filename
        )
      )
    ];
}


console.log(
  '📄 Summary Query:',
  summaryQuery.test(userMessage)
);
    const systemPrompt =
      await this.buildSystemPrompt(
        userId,
        userMessage,
        sessionId,
        pdfChunks
      );

    // ─────────────────────────────────────────────
    // REALTIME INTERNET SEARCH
    // ─────────────────────────────────────────────

    let webContext = '';

    if (needsWebSearch(userMessage)) {

        try {

          console.log(
            '🌐 Performing realtime web search...'
          );

          let searchQuery = userMessage
            .replace(/\n/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          if (searchQuery.length > 350) {

            searchQuery =
              searchQuery.substring(0, 350);

            console.log(
              '⚠️ Search query truncated for Tavily'
            );
          }

          // ─────────────────────────────
          // SMART WEB SEARCH
          // ─────────────────────────────

          const needsRealtimeSearch =

            /\b(today|latest|current|now|live)\b/i
              .test(userMessage);

          if (needsRealtimeSearch) {

            const searchResults =
              await tavilyClient.search(
                searchQuery,
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

          } else {

            webContext = `
Use your own built-in knowledge.
`;
          }

        } catch (err) {

          console.error(
            'Web search failed:',
            err
          );

          webContext = `
        Realtime web search currently unavailable.
        `;
        }
    }

    // Convert to Gemini format
    // Convert conversation history to Groq/OpenAI format
    const history = recentMsgs
      .slice(0, -1)
      .filter(m =>
        m.role === 'user' ||
        m.role === 'assistant'
      )
      .map(m => ({
        role: m.role,
        content: m.content
      }));

    // Add system prompt
    const messages = [
      {
        role: 'system',
        content: systemPrompt + webContext
      },
      ...history,
      {
        role: 'user',
        content: userMessage
      }
    ];

    // ─────────────────────────────────────────────
    // RETRY LOGIC
    // ─────────────────────────────────────────────

    let result;
    let retries = 3;

    while (retries > 0) {

      try {

        result = await this.model.chat.completions.create({
          model: MODEL,
          messages,
          temperature: 0.7,
          max_tokens: 700,
          top_p: 0.9,
        });

        break;

      } catch (err) {

        console.log('⚠️ Groq API Error:', err.message);

        if (retries > 1) {

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
        'Groq API unavailable after retries'
      );
    }

    const responseText =
      result.choices[0].message.content;

    // ─────────────────────────────
    // TOKEN USAGE TRACKING
    // ─────────────────────────────

    const usage =
      result.usage || {};

    const promptTokens =
      usage.prompt_tokens || 0;

    const completionTokens =
      usage.completion_tokens || 0;

    const totalTokens =
      usage.total_tokens || 0;

    // Your Groq daily limit
    const DAILY_LIMIT = 100000;

    // Running total
    global.tokenUsage =
      (global.tokenUsage || 0)
      + totalTokens;

    const remaining =
      DAILY_LIMIT -
      global.tokenUsage;

    console.log('\n📊 TOKEN USAGE');
    console.log(
      '────────────────────────'
    );

    console.log(
      `🧠 Prompt Tokens: ${promptTokens}`
    );

    console.log(
      `💬 Completion Tokens: ${completionTokens}`
    );

    console.log(
      `📦 Total Tokens: ${totalTokens}`
    );

    console.log(
      `⚡ Tokens Used Today: ${global.tokenUsage}`
    );

    console.log(
      `🟢 Tokens Remaining: ${remaining}`
    );

    console.log(
      '────────────────────────\n'
    );


    // Store assistant response
    this.memory.add({
      userId,
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
        this.memory.retrieve(userId,userMessage, 3).length > 0,

      knowledgeUsed:
        this.knowledge.retrieve(userMessage, 1).length > 0,

      sources:
        pdfSources,

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

// Load embedding model on startup
loadEmbeddingModel();

// ─────────────────────────────────────────────
// AUTH MIDDLEWARE
// ─────────────────────────────────────────────

function authMiddleware(req, res, next) {

  const authHeader =
    req.headers.authorization;

  if (!authHeader) {

    return res.status(401).json({
      error: 'No token provided'
    });
  }

  const token =
    authHeader.split(' ')[1];

  try {

    const decoded =
      jwt.verify(
        token,
        process.env.JWT_SECRET
      );

    req.user = decoded;

    next();

  } catch (err) {

    return res.status(401).json({
      error: 'Invalid token'
    });
  }
}

// ─────────────────────────────────────────────
// API ROUTES
// ─────────────────────────────────────────────

// Create new session
app.post(
  '/api/session',
  authMiddleware,
  (req, res) => {

    const session =
      agent.sessions.createSession(
        req.user.userId,
        req.body.title
      );

    res.json({ session });
  }
);

// Get all sessions
app.get(
  '/api/sessions',
  authMiddleware,
  (req, res) => {

    const sessions =
      agent.sessions.getAllSessions(
        req.user.userId
      );

    res.json({ sessions });
  }
);

// Get session messages
app.get('/api/session/:id', authMiddleware, (req, res) => {
  const session = agent.sessions.getSession(req.params.id);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  res.json({ session });
});

// Main chat endpoint
app.post('/api/chat', authMiddleware, async (req, res) => {
  const { sessionId, message } = req.body;
  if (!message || !sessionId) return res.status(400).json({ error: 'sessionId and message required' });

  try {
    const result = await agent.chat(
      req.user.userId,
      sessionId,
      message
    );
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
  const results = agent.memory.retrieve(req.user.userId, query, topK || 5);
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
// AUTH ROUTES
// ─────────────────────────────────────────────

app.post('/api/auth/register', async (req, res) => {

  try {

    const { username, password } = req.body;

    if (!username || !password) {

      return res.status(400).json({
        error: 'Username and password required'
      });
    }

    const existing =
      await userRepo.findByUsername(username);

    if (existing) {

      return res.status(400).json({
        error: 'User already exists'
      });
    }

    const hashedPassword =
      await bcrypt.hash(password, 10);

    const user =
      await userRepo.createUser(
        username,
        hashedPassword
      );

    res.json({
      success: true,
      user
    });

  } catch (err) {

    console.error(err);

    res.status(500).json({
      error: 'Registration failed'
    });
  }
});

app.post('/api/auth/login', async (req, res) => {

  try {

    const { username, password } = req.body;

    const user =
      await userRepo.findByUsername(username);

    if (!user) {

      return res.status(401).json({
        error: 'Invalid username or password'
      });
    }

    const valid =
      await bcrypt.compare(
        password,
        user.password
      );

    if (!valid) {

      return res.status(401).json({
        error: 'Invalid username or password'
      });
    }

    const token = jwt.sign(
      {
        userId: user.id,
        username: user.username
      },
      process.env.JWT_SECRET,
      {
        expiresIn: '7d'
      }
    );

    res.json({
      success: true,
      token,
      user: {
        id: user.id,
        username: user.username
      }
    });

  } catch (err) {

    console.error(err);

    res.status(500).json({
      error: 'Login failed'
    });
  }
});

// ─────────────────────────────────────────────
// PDF UPLOAD + RAG INGESTION
// ─────────────────────────────────────────────

app.post(
  '/api/pdf/upload',

  authMiddleware,

  upload.single('pdf'),

  async (req, res) => {

    try {

      if (!req.file) {

        return res.status(400).json({
          error: 'No PDF uploaded'
        });
      }

      // Read uploaded PDF
      const dataBuffer =
        fs.readFileSync(req.file.path);

      // Extract text
      const pdfData =
        await pdfParse(
          dataBuffer,
          {
            max: 0
          }
        );

      const fullText =
        pdfData.text;

      // ─────────────────────────────
      // CHUNKING
      // ─────────────────────────────

      const chunkSize = 800;

      const chunks = [];

      for (
        let i = 0;
        i < fullText.length;
        i += chunkSize
      ) {

        const text =
          fullText.substring(
            i,
            i + chunkSize
          );

        chunks.push({

          text,

          chunkIndex:
            chunks.length,

          relativePosition:
            i / fullText.length
          });
      }

      // ─────────────────────────────
      // GENERATE EMBEDDINGS
      // ─────────────────────────────

      console.log(
        '🧠 Generating chunk embeddings...'
      );

      for (const chunk of chunks) {

        chunk.embedding =
          await createEmbedding(
            chunk.text
          );
      }

      console.log(
        '✅ Embeddings generated'
      );

      // ─────────────────────────────
      // GENERATE PDF SUMMARY
      // ─────────────────────────────

      let pdfSummary =
        'Summary unavailable';

      try {

        console.log(
          '📝 Generating PDF summary...'
        );

        const summaryPrompt = `
Analyze this PDF and identify its document type.

Possible types:
- Research Paper
- Certificate
- Resume
- Notes
- Report
- Other

Then provide a concise summary appropriate for that document type.

Content:

${fullText.substring(0, 4000)}
`;

        const summaryResult =
          await groq.chat.completions.create({

            model: MODEL,

            messages: [
              {
                role: 'system',
                content:
                  'You are an expert research paper analyst.'
              },
              {
                role: 'user',
                content:
                  summaryPrompt
              }
            ],

            temperature: 0.2,

            max_tokens: 500
          });

        pdfSummary =
          summaryResult
            .choices[0]
            .message.content;

        console.log(
          '✅ PDF summary generated'
        );

      }
      catch (err) {

        console.log(
          '⚠️ Summary generation skipped'
        );

        console.log(
          err.message
        );
      }

      // Save document
      const document = {

        id:
          Date.now().toString(),

        userId:
          req.user.userId,

        filename:
          req.file.originalname,

        summary:
          pdfSummary,

        path:
          req.file.path,

        uploadedAt:
          new Date().toISOString(),

        chunks
      };

      pdfRepo.addDocument(document);

      res.json({

        success: true,

        filename:
          req.file.originalname,

        chunks:
          chunks.length
      });

    } catch (err) {

      console.error(
        'PDF Upload Error:',
        err
      );

      res.status(500).json({
        error:
          'PDF processing failed'
      });
    }
  }
);

// ─────────────────────────────────────────────
// SMART NOTES GENERATION
// ─────────────────────────────────────────────

app.post(
  '/api/notes/generate',

  authMiddleware,

  async (req, res) => {

    try {

      const userId =
        req.user.userId;

      // Get all user PDFs
      const documents =
        pdfRepo.getUserDocuments(
          userId
        );

      if (!documents.length) {

        return res.status(400).json({
          error:
            'No PDFs uploaded'
        });
      }

      // Merge chunks
      let combinedText = '';

      documents.forEach(doc => {

        combinedText +=
          `\n\nDOCUMENT: ${doc.filename}\n`;

        combinedText +=
          doc.chunks
            .slice(0, 8)
            .map(c => c.text)
            .join('\n');
      });

      // Limit size
      combinedText =
        combinedText.substring(
          0,
          12000
        );

      // AI Prompt
      const prompt = `
Generate high-quality structured study notes from the following academic content.

FORMAT:

# Title

## Key Concepts
- bullet points

## Important Explanations

## Key Takeaways

## Possible Exam Questions

CONTENT:
${combinedText}
`;

      const result =
        await groq.chat.completions.create({

          model: MODEL,

          messages: [
            {
              role: 'system',
              content:
                'You are an expert study notes generator.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],

          temperature: 0.4,

          max_tokens: 1800
        });

      const notes =
        result.choices[0]
          .message.content;

      res.json({

        success: true,

        notes
      });

    } catch (err) {

      console.error(
        'Notes generation error:',
        err
      );

      res.status(500).json({
        error:
          'Notes generation failed'
      });
    }
  }
);

// ─────────────────────────────────────────────
// STUDY PLANNER GENERATION
// ─────────────────────────────────────────────

app.post(
  '/api/planner/generate',

  authMiddleware,

  async (req, res) => {

    try {

      const userId =
        req.user.userId;

      // Get uploaded PDFs
      const documents =
        pdfRepo.getUserDocuments(
          userId
        );

      if (!documents.length) {

        return res.status(400).json({
          error:
            'No PDFs uploaded'
        });
      }

      // Merge chunks
      let combinedText = '';

      documents.forEach(doc => {

        combinedText +=
          `\n\nDOCUMENT: ${doc.filename}\n`;

        combinedText +=
          doc.chunks
            .slice(0, 8)
            .map(c => c.text)
            .join('\n');
      });

      // Limit size
      combinedText =
        combinedText.substring(
          0,
          12000
        );

      // Planner prompt
      const prompt = `
Generate a structured 7-day study plan from the following academic material.

FORMAT:

# 7-Day Study Plan

## Day 1
- Topics
- Estimated study time
- Revision tasks

## Day 2
...

Include:
- priorities
- revision strategy
- difficulty progression
- exam preparation tips

CONTENT:
${combinedText}
`;

      const result =
        await groq.chat.completions.create({

          model: MODEL,

          messages: [
            {
              role: 'system',
              content:
                'You are an expert AI study planner.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],

          temperature: 0.5,

          max_tokens: 1800
        });

      const plan =
        result.choices[0]
          .message.content;
      
      // ─────────────────────────────
      // AUTO CREATE TASKS FROM PLAN
      // ─────────────────────────────

      const lines =
        plan.split('\n');

      const createdTasks = [];

      for (const line of lines) {

        const clean =
          line.trim();

        if (

          clean.startsWith('-') ||

          clean.match(/^day\s+\d+/i)

        ) {

          const task = {

            id:
              Date.now().toString() +
              Math.random(),

            userId:
              req.user.userId,

            title:
              clean.replace(/^-\s*/, ''),

            deadline:
              null,

            priority:
              'medium',

            completed:
              false,

            createdAt:
              new Date().toISOString()
          };

          taskRepo.addTask(task);

          createdTasks.push(task);
        }
          }

      res.json({

        success: true,

        plan,

        tasks:
          createdTasks
          });

    } catch (err) {

      console.error(
        'Study planner error:',
        err
      );

      res.status(500).json({
        error:
          'Study planner generation failed'
      });
    }
  }
);

// ─────────────────────────────────────────────
// TASK MANAGEMENT APIs
// ─────────────────────────────────────────────

// Create task
app.post(
  '/api/tasks',

  authMiddleware,

  (req, res) => {

    try {

      const {
        title,
        deadline,
        priority
      } = req.body;

      const task = {

        id:
          Date.now().toString(),

        userId:
          req.user.userId,

        title,

        deadline,

        priority:
          priority || 'medium',

        completed:
          false,

        createdAt:
          new Date().toISOString()
      };

      taskRepo.addTask(task);

      res.json({
        success: true,
        task
      });

    } catch (err) {

      console.error(err);

      res.status(500).json({
        error:
          'Failed to create task'
      });
    }
  }
);

// Get user tasks
app.get(
  '/api/tasks',

  authMiddleware,

  (req, res) => {

    try {

      const tasks =
        taskRepo.getUserTasks(
          req.user.userId
        );

      res.json({
        success: true,
        tasks
      });

    } catch (err) {

      console.error(err);

      res.status(500).json({
        error:
          'Failed to fetch tasks'
      });
    }
  }
);

// Mark task complete
app.put(
  '/api/tasks/:id',

  authMiddleware,

  (req, res) => {

    try {

      const updated =
        taskRepo.updateTask(
          req.params.id,
          req.body
        );

      if (!updated) {

        return res.status(404).json({
          error:
            'Task not found'
        });
      }

      res.json({
        success: true,
        task: updated
      });

    } catch (err) {

      console.error(err);

      res.status(500).json({
        error:
          'Failed to update task'
      });
    }
  }
);

// ─────────────────────────────────────────────
// START SERVER
// ─────────────────────────────────────────────
const PORT =
  process.env.PORT || 5000;

async function startServer() {

  await connectDB();

  app.listen(
    PORT,
    () => {

      console.log(
        `🚀 RAG Agentic AI Chatbot running at http://localhost:${PORT}`
      );

    }
  );
}

startServer();