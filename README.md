# AI Research Assistant Chatbot

An Agentic AI Research Assistant that supports PDF-based Retrieval-Augmented Generation (RAG), document summarization, study note generation, research planning, memory, and semantic search using MongoDB and Qdrant.

---

# Features

## User Management

* User Registration
* User Login
* Secure Password Hashing
* MongoDB User Storage

---

## PDF Management

* Upload PDF Documents
* Automatic Text Extraction
* Chunk Generation
* Embedding Generation
* Semantic Search
* Latest PDF Detection
* PDF Collection Management

---

## Research Assistance

* Structured PDF Summaries
* Key Contributions Extraction
* Methodology Extraction
* Applications Extraction
* Limitations Extraction
* Research-Oriented Notes Generation
* 7-Day Research Study Planner
* Viva Question Generation
* Exam Question Generation

---

## RAG (Retrieval-Augmented Generation)

* Semantic PDF Retrieval
* Context-Aware Question Answering
* Multi-Document Search
* Qdrant Vector Database Integration
* User-Specific Document Filtering

---

## Conversation Memory

* Current Document Tracking
* Current Topic Tracking
* Current Entity Tracking
* Intent Tracking
* Follow-Up Question Resolution

Examples:

* What is the latest PDF?
* Summarize it
* What is its contribution?
* How many PDFs do I have?
* Name them

---

# Technology Stack

## Backend

* Node.js
* Express.js

## Database

* MongoDB Atlas

Collections:

* users
* documents
* sessioncontexts

## Vector Database

* Qdrant

Stores:

* PDF Chunk Embeddings
* Semantic Search Vectors

## AI Models

### Embeddings

* Xenova Transformers

### LLM

* Groq API

Supported Models:

* Llama
* Qwen
* DeepSeek
* Gemma

---

# Project Structure

```text
CHATBOT/
│
├── db/
│   ├── MongoDocumentRepository.js
│   ├── MongoUserRepository.js
│   ├── TaskRepository.js
│
├── models/
│   ├── Document.js
│   ├── SessionContext.js
│   ├── user.js
│
├── services/
│   ├── DocumentService.js
│   ├── DocumentContextService.js
│   ├── QdrantService.js
│
├── uploads/
│
├── server.js
├── qdrant.js
├── qdrant-init.js
├── package.json
│
└── README.md
```

---

# Storage Architecture

## MongoDB

### users

Stores:

* User Information
* Login Credentials

### documents

Stores:

* User ID
* Filename
* Summary
* File Path
* Upload Date

### sessioncontexts

Stores:

* User ID
* Current Document
* Current Topic
* Current Entity
* Last Intent
* Updated Time

---

## Qdrant

Collection:

```text
pdf_chunks
```

Payload:

```json
{
  "userId": "...",
  "filename": "...",
  "text": "..."
}
```

Stores semantic embeddings for PDF chunks.

---

# Supported Commands

## PDF Commands

```text
What is the latest PDF?
Summarize it
What is its contribution?
How many PDFs do I have?
Name them
```

## Research Commands

```text
Generate notes
Generate study plan
Create viva questions
Create exam questions
```

## RAG Commands

```text
Explain self-attention
What is Flamingo?
Compare CLIP and MedCLIP
Explain Vision Transformer
```

---

# Setup

## Install Dependencies

```bash
npm install
```

## Configure Environment Variables

Create:

```text
.env
```

Example:

```env
MONGO_URI=your_mongodb_connection_string

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

GROQ_API_KEY=your_groq_api_key
```

---

# Start Application

```bash
node server.js
```

Server:

```text
http://localhost:5000
```

---

# Completed Migrations

## MongoDB Migration

Completed

* Users moved from JSON to MongoDB
* Documents moved from JSON to MongoDB
* Session Context moved to MongoDB

## Qdrant Migration

Completed

* Embeddings moved to Qdrant
* Semantic Search moved to Qdrant

## Service Layer Refactor

Completed

* DocumentService
* DocumentContextService
* QdrantService

---

# Future Enhancements

* Multi-PDF Comparative Analysis
* Citation Extraction
* Research Gap Detection
* Thesis Topic Recommendation
* Paper Recommendation Engine
* Agentic Research Workflow
* Medical VLM Research Assistant

---

# Author

Sougata Roy

M.Tech (Computer Science)

AI Research | Computer Vision | Vision Language Models | Deep Learning
