import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import coachHandler from "./api/coach.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
loadLocalEnv();

const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 3000);

const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".wasm": "application/wasm"
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host}`);

  if (url.pathname === "/api/coach") {
    await handleApiRequest(req, res);
    return;
  }

  await serveStaticFile(url.pathname, res);
});

server.listen(port, host, () => {
  console.log(`SpikeCheck running at http://${host}:${port}`);
});

function loadLocalEnv() {
  const envPath = path.join(__dirname, ".env");

  if (!existsSync(envPath)) {
    return;
  }

  try {
    const contents = readFileSync(envPath, "utf8");

    for (const line of contents.split(/\r?\n/)) {
      const match = line.match(/^\s*([\w.-]+)\s*=\s*(.*)\s*$/);

      if (!match || match[1].startsWith("#") || process.env[match[1]]) {
        continue;
      }

      process.env[match[1]] = stripEnvQuotes(match[2]);
    }
  } catch (error) {
    console.warn(`Could not load .env: ${error.message}`);
  }
}

function stripEnvQuotes(value) {
  const trimmed = value.trim();

  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

async function handleApiRequest(req, res) {
  try {
    req.body = await readJsonBody(req);

    const apiRes = createApiResponse(res);
    await coachHandler(req, apiRes);

    if (!res.writableEnded) {
      res.end();
    }
  } catch (error) {
    if (!res.writableEnded) {
      res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
      res.end(
        JSON.stringify({
          error: error.message || "Local API server failed."
        })
      );
    }
  }
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let raw = "";

    req.on("data", (chunk) => {
      raw += chunk;

      if (raw.length > 1_000_000) {
        reject(new Error("Request body is too large."));
        req.destroy();
      }
    });

    req.on("end", () => {
      if (!raw) {
        resolve({});
        return;
      }

      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("Request body must be valid JSON."));
      }
    });

    req.on("error", reject);
  });
}

function createApiResponse(res) {
  return {
    status(code) {
      res.statusCode = code;
      return this;
    },
    json(payload) {
      if (!res.headersSent) {
        res.setHeader("Content-Type", "application/json; charset=utf-8");
      }

      res.end(JSON.stringify(payload));
      return this;
    }
  };
}

async function serveStaticFile(pathname, res) {
  const requestedPath = pathname === "/" ? "/index.html" : pathname;
  const decodedPath = decodeURIComponent(requestedPath);
  const filePath = path.normalize(path.join(__dirname, decodedPath));

  const relativePath = path.relative(__dirname, filePath);

  if (relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  try {
    const file = await readFile(filePath);
    const extension = path.extname(filePath);

    res.writeHead(200, {
      "Content-Type": contentTypes[extension] || "application/octet-stream"
    });
    res.end(file);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
}
