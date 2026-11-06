"use strict";
// Repo-owned admission policy, loaded before direct tsc/tsx/vitest/Node entrypoints.
// This is coordination, not a security boundary: do not inspect arbitrary repo
// content or change provider CLI behavior when no check policy is present.
const fs = require("node:fs");
const path = require("node:path");

if (process.env.BAWT_AGENT_BRIDGE === "1" || process.env.BRIDGE_BROKER_TOKEN_FILE) {
  const seen = new Set();
  for (const origin of [process.cwd(), ...process.argv.slice(1).filter(arg => path.isAbsolute(arg)).map(arg => path.dirname(arg))]) {
    let directory = origin;
    while (!seen.has(directory)) {
      seen.add(directory);
      const guard = path.join(directory, "scripts", "heavy-check-guard.cjs");
      if (fs.existsSync(guard)) {
        try { require(guard).check(); }
        catch (error) {
          if (error.code !== "BAWT_HEAVY_CHECK_REFUSED") throw error;
          process.stderr.write(error.message + "\n");
          process.exit(2);
        }
        break;
      }
      const parent = path.dirname(directory);
      if (parent === directory) break;
      directory = parent;
    }
  }
}
