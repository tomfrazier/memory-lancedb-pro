import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import Module from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";

import { Command } from "commander";
import jitiFactory from "jiti";

process.env.NODE_PATH = [
  process.env.NODE_PATH,
  "/opt/homebrew/lib/node_modules/openclaw/node_modules",
  "/opt/homebrew/lib/node_modules",
].filter(Boolean).join(":");
Module._initPaths();

const jiti = jitiFactory(import.meta.url, { interopDefault: true });

async function captureStdout(run) {
  const chunks = [];
  const originalLog = console.log;
  console.log = (...args) => {
    chunks.push(args.join(" "));
  };
  try {
    await run();
  } finally {
    console.log = originalLog;
  }
  return chunks.join("\n");
}

async function createSourceDb(sourceDbPath) {
  // Create a minimal LanceDB database with a `memories` table and 1 row.
  const { loadLanceDB } = jiti("../src/store.ts");
  const lancedb = await loadLanceDB();
  const db = await lancedb.connect(sourceDbPath);

  // Create table if missing.
  // LanceDB JS API supports createTable(name, data).
  const row = {
    id: "test_smoke_1",
    text: "hello from smoke test",
    category: "other",
    scope: "global",
    importance: 0.7,
    timestamp: Date.now(),
    metadata: "{}",
    vector: [0, 0, 0, 0],
  };

  try {
    await db.createTable("memories", [row]);
  } catch {
    // If already exists, ignore.
    const table = await db.openTable("memories");
    await table.add([row]);
  }
}

async function runCliSmoke() {
  const workDir = mkdtempSync(path.join(tmpdir(), "memory-lancedb-pro-smoke-"));
  const sourceDbPath = path.join(workDir, "source-db");

  await createSourceDb(sourceDbPath);

  const { createMemoryCLI } = jiti("../cli.ts");
  const { MemoryStore } = jiti("../src/store.ts");
  const { registerMemoryRecallTool } = jiti("../src/tools.ts");

  const program = new Command();
  program.exitOverride();

  const context = {
    // Minimal store interface for reembed dry-run.
    store: { dbPath: path.join(workDir, "target-db") },
    retriever: {},
    scopeManager: {},
    migrator: {},
    // Presence required, but dry-run exits before embeddings.
    embedder: {},
  };

  // Register commands under `memory-pro`
  createMemoryCLI(context)({ program });

  // 1) version command should not throw
  await program.parseAsync(["node", "openclaw", "memory-pro", "version"]);

  // 2) reembed dry-run should not crash (regression test for clampInt)
  await program.parseAsync([
    "node",
    "openclaw",
    "memory-pro",
    "reembed",
    "--source-db",
    sourceDbPath,
    "--limit",
    "1",
    "--batch-size",
    "999",
    "--dry-run",
  ]);

  // 3) search should rebuild a working retriever when the wrapper-provided one is broken
  const entry = {
    id: "search_regression_1",
    text: "Jige profile memory",
    vector: [1, 0],
    category: "fact",
    scope: "global",
    importance: 0.9,
    timestamp: Date.now(),
    metadata: "{}",
  };

  const brokenContext = {
    store: {
      dbPath: path.join(workDir, "target-db"),
      hasFtsSupport: true,
      async vectorSearch() {
        return [{ entry, score: 0.72 }];
      },
      async bm25Search() {
        return [{ entry, score: 0.88 }];
      },
      async hasId(id) {
        return id === entry.id;
      },
    },
    retriever: {
      async retrieve() {
        return [];
      },
      getConfig() {
        return {
          mode: "hybrid",
          vectorWeight: 0.7,
          bm25Weight: 0.3,
          minScore: 0.3,
          rerank: "none",
          candidatePoolSize: 20,
          recencyHalfLifeDays: 0,
          recencyWeight: 0,
          filterNoise: false,
          lengthNormAnchor: 0,
          hardMinScore: 0,
          timeDecayHalfLifeDays: 0,
        };
      },
    },
    scopeManager: {},
    migrator: {},
    embedder: {
      async embedQuery() {
        return [1, 0];
      },
    },
  };

  const searchProgram = new Command();
  searchProgram.exitOverride();
  createMemoryCLI(brokenContext)({ program: searchProgram });

  const searchOutput = await captureStdout(async () => {
    await searchProgram.parseAsync([
      "node",
      "openclaw",
      "memory-pro",
      "search",
      "Jige",
      "--scope",
      "global",
      "--json",
    ]);
  });

  assert.match(searchOutput, /search_regression_1/);

  // 4) bm25Search should fall back to lexical substring matching for short CJK queries
  const lexicalStore = new MemoryStore({
    dbPath: path.join(workDir, "lexical-db"),
    vectorDim: 4,
  });
  await lexicalStore.importEntry({
    id: "bm25_zh_1",
    text: "用户测试饮料偏好是乌龙茶，不喜欢美式咖啡。",
    vector: [0, 0, 0, 0],
    category: "preference",
    scope: "global",
    importance: 0.95,
    timestamp: Date.now(),
    metadata: "{}",
  });
  const lexicalHits = await lexicalStore.bm25Search("乌龙茶", 5, ["global"]);
  assert.equal(lexicalHits[0]?.entry.id, "bm25_zh_1");

  // 5) memory_recall tool should retry once when the first retrieval is empty
  let recallCalls = 0;
  const recallApi = {
    registerTool(factory, meta) {
      const tool = factory({ agentId: "main", sessionKey: "agent:main:test" });
      recallApi.tool = tool;
      recallApi.name = meta.name;
    },
  };
  registerMemoryRecallTool(recallApi, {
    retriever: {
      async retrieve() {
        recallCalls += 1;
        if (recallCalls === 1) return [];
        return [
          {
            entry,
            score: 0.88,
            sources: {},
          },
        ];
      },
      getConfig() {
        return { mode: "hybrid" };
      },
    },
    store: {
      async patchMetadata() {},
    },
    scopeManager: {
      getAccessibleScopes() {
        return ["agent:main"];
      },
      isAccessible() {
        return true;
      },
    },
    embedder: {},
    agentId: "main",
  });
  const recallResult = await recallApi.tool.execute("call", { query: "Jige", limit: 5 });
  assert.equal(recallResult.details.count, 1);
  assert.equal(recallCalls, 2);

  rmSync(workDir, { recursive: true, force: true });
}

runCliSmoke()
  .then(() => {
    console.log("OK: CLI smoke test passed");
  })
  .catch((err) => {
    console.error("FAIL: CLI smoke test failed");
    console.error(err);
    process.exit(1);
  });
