const path = require('path');
const fs = require('fs');

let betterSqlite3;
try {
  betterSqlite3 = require('better-sqlite3');
} catch (e) {
  betterSqlite3 = null;
}

const SCHEMA_PATH = path.join(__dirname, 'schema.sql');


const DB_PATH = process.env.SQLITE_PATH
  ? path.resolve(process.env.SQLITE_PATH)
  : path.join(__dirname, '..', 'data', 'chatbot.sqlite');

function ensureDbDir() {
  const dir = path.dirname(DB_PATH);
  fs.mkdirSync(dir, { recursive: true });
}

function initDb() {
  ensureDbDir();
  const schemaSql = fs.readFileSync(SCHEMA_PATH, 'utf8');

  if (betterSqlite3) {
    const db = new betterSqlite3(DB_PATH);
    db.pragma('journal_mode = WAL');
    db.exec(schemaSql);
    return db;
  }

  // If better-sqlite3 isn't available, we cannot proceed (avoid pulling sqlite3 native deps).
  throw new Error('No SQLite driver available. Install better-sqlite3.');


  // Apply schema synchronously-ish (callback-based)
  return new Promise((resolve, reject) => {
    db.exec(schemaSql, (err) => {
      if (err) return reject(err);
      resolve(db);
    });
  });
}

// Synchronous wrapper for better-sqlite3
const getDb = () => {
  if (!initDb._db) {
    initDb._db = initDb();
  }
  return initDb._db;
};

module.exports = {
  DB_PATH,
  initDb,
  getDb,
};

