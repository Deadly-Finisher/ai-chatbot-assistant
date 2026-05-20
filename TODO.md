# TODO

## Phase 1 — SQLite persistence (infinite memory, no JSON loss)
- [x] Replace `data/*.json` stores with SQLite (`memories`, `sessions`, `knowledge`, `feedback`).
- [x] Preserve existing API routes and response formats for frontend compatibility.
- [x] Add DB helper layer and parameterized queries.



## Phase 2 — Better grounding + anti-hallucination enforcement
- [ ] Add source-tracked context assembly (include retrieved IDs/snippets).
- [ ] Add “insufficient info” response rule to prevent fabrication.

## Phase 3 — Local file upload + ingestion
- [ ] Add upload UI to `index.html`.
- [ ] Add frontend handler in `script.js` (multipart upload).
- [ ] Implement `POST /api/files/upload` in `server.js`.
- [ ] Implement chunking + storing uploaded content into SQLite.

## Phase 4 — Web fallback (Tavily)
- [ ] Add Tavily integration (env var + request to search).
- [ ] Implement grounded web retrieval when local knowledge is insufficient.
- [ ] If web grounding still insufficient, return “I don't have information about that”.

## Phase 5 — Reliability, validation, and quality
- [ ] Add input validation for all endpoints.
- [ ] Add centralized error handling.
- [ ] Add rate limiting (lightweight) to prevent spam learning.

## Phase 6 — Hackathon polish
- [ ] Add/adjust README with env vars for Gemini + SQLite + Tavily.
- [ ] Smoke test the full flow: sessions, chat, feedback learning, persistence restart, file upload, web fallback.

