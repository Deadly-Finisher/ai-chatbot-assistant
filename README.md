# RAG Agentic AI Chatbot

A full-stack Agentic RAG (Retrieval-Augmented Generation) chatbot built with Node.js, Express, MongoDB, Qdrant, Groq LLMs, and Tavily Web Search.

The chatbot supports PDF ingestion, semantic search, study note generation, study planning, task management, memory, and real-time web search.

---

# Features

## Authentication

* User Registration
* User Login
* JWT Authentication
* Protected API Routes

---

## PDF Knowledge Base

* Upload PDF documents
* Automatic PDF text extraction
* Chunking of large documents
* Embedding generation
* Vector storage in Qdrant
* Semantic document retrieval
* Multi-document support

Supported Queries:

* Latest uploaded PDF
* List all uploaded PDFs
* Count uploaded PDFs
* Summarize PDF
* Ask questions about PDFs
* Follow-up document conversations

Examples:

```text
What is the latest PDF?

Summarize it

What are its contributions?

How does it compare with CLIP?
```

---

## RAG Pipeline

### Flow

```text
User Query
      ↓
Intent Detection
      ↓
Vector Search (Qdrant)
      ↓
Relevant Chunks Retrieval
      ↓
Groq LLM
      ↓
Final Answer
```

### Components

* PDF Processing
* Chunking
* Embeddings
* Qdrant Vector Database
* Context Retrieval
* Response Generation

---

## Agentic AI Features

### Intent Detection

Automatically detects:

* PDF Queries
* Research Queries
* Task Queries
* Study Planning Queries
* Notes Generation Queries
* Web Search Queries
* General Conversation

---

## Long-Term Memory

Stores:

* Previous conversations
* User preferences
* Learning context

Used for:

* Follow-up conversations
* Context retention
* Personalized responses

---

## Document Context Tracking

The chatbot remembers the current document.

Example:

```text
What is the latest PDF?

→ BLIP-2.pdf

Summarize it

→ BLIP-2 Summary

What are its contributions?

→ BLIP-2 Contributions
```

No need to repeatedly specify document names.

---

# Study Notes Generator

Generate structured study notes from uploaded PDFs.

Example:

```text
Generate notes
```

Output includes:

* Topic summaries
* Key concepts
* Important formulas
* Research contributions
* Exam revision points

---

# Study Planner Generator

Automatically creates study plans from uploaded PDFs.

Example:

```text
Generate study plan
```

Features:

* Multi-day learning schedule
* Topic ordering
* Learning objectives
* Revision planning
* Viva preparation

---

# Task Management

Automatically creates tasks from generated study plans.

Features:

* Task creation
* Deadline generation
* Task listing
* Completion tracking
* Priority tracking

Example:

```text
Show my tasks
```

Output:

```text
Day 1 - Foundations
Deadline: 31/05/2026

Day 2 - Core Concepts
Deadline: 01/06/2026
```

---

# Web Search

Real-time web search using Tavily.

Example:

```text
Can you fetch land prices in the USA?

Latest papers on diabetic retinopathy VLMs

Recent AI conference deadlines
```

Features:

* Internet search
* Research search
* Current information retrieval
* Source citation support

---

# Tech Stack

## Backend

* Node.js
* Express.js

## Database

* MongoDB

Collections:

* Users
* Documents
* SessionContexts

## Vector Database

* Qdrant

## AI Models

### LLM

* Groq API
* Llama 3.3 70B Versatile

### Embeddings

* Xenova Transformers
* all-MiniLM-L6-v2

## Web Search

* Tavily API

---

# Project Structure

```text
Chatbot/
│
├── server.js
│
├── db/
│   ├── MongoUserRepository.js
│   ├── MongoDocumentRepository.js
│   └── TaskRepository.js
│
├── models/
│   ├── User.js
│   ├── Document.js
│   └── SessionContext.js
│
├── services/
│   ├── DocumentService.js
│   ├── DocumentContextService.js
│   ├── EmbeddingService.js
│   └── MemoryService.js
│
├── uploads/
│
├── data/
│   ├── tasks.json
│   ├── memory.json
│   ├── knowledge.json
│   └── sessions.json
│
└── public/
```

---

# Environment Variables

Create a `.env` file:

```env
PORT=5000

JWT_SECRET=your_secret

MONGODB_URI=your_mongodb_connection_string

GROQ_API_KEY=your_groq_api_key

TAVILY_API_KEY=your_tavily_api_key

QDRANT_URL=your_qdrant_url

QDRANT_API_KEY=your_qdrant_api_key
```

---

# Installation

Install dependencies:

```bash
npm install
```

Run:

```bash
node server.js
```

Expected startup:

```text
Loading AI models...
AI models loaded

MongoDB Connected
Database indexes ready
Qdrant collection exists

RAG Agentic AI Chatbot running at http://localhost:5000
```

---

# API Features

## Authentication

```text
POST /api/register
POST /api/login
```

## PDF Upload

```text
POST /api/upload-pdf
```

## Chat

```text
POST /api/chat
```

## Notes

```text
POST /api/notes/generate
```

## Planner

```text
POST /api/planner/generate
```

## Tasks

```text
GET /api/tasks
POST /api/tasks
PUT /api/tasks/:id
DELETE /api/tasks/:id
```

---

# Current Status

## Completed

* Authentication
* MongoDB Integration
* PDF Upload
* PDF Summarization
* Qdrant Search
* RAG Retrieval
* Memory
* Context Tracking
* Web Search
* Notes Generation
* Study Planner
* Task Creation
* Deadline Generation

## Future Improvements

* Migrate TaskRepository from JSON to MongoDB
* Multi-user task analytics
* Calendar integration
* Research paper recommendation engine
* Agent workflow orchestration
* Citation-aware answer generation

---

# Version

Current Version: v1.0 Stable

Status: Production Ready
