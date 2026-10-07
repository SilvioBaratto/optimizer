#!/usr/bin/env node
// Normalize LaTeX math in generated Markdown so GitHub's renderer accepts it.
//
// GitHub runs CommonMark/GFM inline parsing (emphasis, escapes, flanking) BEFORE
// its math extension, so a bare `$...$` whose body contains a markdown-active pair
// (`*...*` / `_..._`), an escaped `\$`, or a `$` glued to adjacent punctuation never
// has its delimiters matched and silently falls through to plain markdown. This
// script rewrites inline math as the code-span-protected `` $`...`$ `` form (GitHub's
// documented inline syntax) and emits every `$$...$$` block as a blank-line-separated
// display block, which the block parser renders reliably. The transform is idempotent.
//
// Usage: node scripts/fix-github-math.mjs [dir ...]   (default dir: openwiki)

import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

/**
 * Split OKF/YAML frontmatter from the body.
 *
 * @param {string} text Full file contents (LF-normalized).
 * @returns {[string, string]} The frontmatter block (empty when absent) and the body.
 */
function splitFrontmatter(text) {
  if (text.startsWith("---\n")) {
    const end = text.indexOf("\n---\n", 4);
    if (end !== -1) {
      const cut = end + 5;
      return [text.slice(0, cut), text.slice(cut)];
    }
  }
  return ["", text];
}

/**
 * Rewrite the body's math into GitHub-safe form.
 *
 * Code spans/fences, already-wrapped inline math, and escaped `\$` are preserved;
 * inline `$...$` becomes `` $`...`$ `` and each `$$...$$` becomes a blank-separated
 * display block (single-line content stays glued to the delimiters so no interior
 * line can be read as a list/heading marker).
 *
 * @param {string} s The page body.
 * @returns {string} The transformed body.
 */
function transformBody(s) {
  const n = s.length;
  const out = [];
  let i = 0;
  while (i < n) {
    const c = s[i];
    if (c === "\\") {
      out.push(s.slice(i, i + 2));
      i += 2;
      continue;
    }
    if (c === "`") {
      let k = 0;
      while (i + k < n && s[i + k] === "`") k += 1;
      const fence = "`".repeat(k);
      const j = s.indexOf(fence, i + k);
      if (j === -1) {
        out.push(fence);
        i += k;
      } else {
        out.push(s.slice(i, j + k));
        i = j + k;
      }
      continue;
    }
    if (c === "$" && i + 1 < n && s[i + 1] === "$") {
      const j = s.indexOf("$$", i + 2);
      if (j === -1) {
        out.push("$$");
        i += 2;
        continue;
      }
      const inner = s.slice(i + 2, j).trim();
      const lineStart = s.lastIndexOf("\n", i - 1) + 1;
      const inBlockquote = s.slice(lineStart, i).replace(/^\s+/, "").startsWith(">");
      if (inBlockquote) {
        // A display block can't be blank-line-separated without leaving the blockquote
        // (which fragments it), and `>`-prefixed multi-line $$ does not render on
        // GitHub — keep the math in the quote as inline.
        out.push("$`" + inner.split(/\s+/).join(" ") + "`$");
      } else {
        out.push(inner.includes("\n") ? `\n\n$$\n${inner}\n$$\n\n` : `\n\n$$${inner}$$\n\n`);
      }
      i = j + 2;
      continue;
    }
    if (c === "$") {
      // A `$` right before digits plus a money multiplier (`$20M`, `$1 billion`) is
      // literal currency the generator forgot to escape, not math; escaping it stops
      // the stray `$` from pairing with a real formula. `$150$`/`$20\%$` stay math.
      if (/^\d[\d.,/]*(?:\s?(?:million|billion|trillion|bn|tn)\b|[MBK]\b)/.test(s.slice(i + 1, i + 30))) {
        out.push("\\$");
        i += 1;
        continue;
      }
      let k = i + 1;
      let close = -1;
      while (k < n) {
        if (s[k] === "\\" && k + 1 < n) {
          k += 2;
          continue;
        }
        if (s[k] === "$") {
          close = k;
          break;
        }
        if (s[k] === "\n" && k + 1 < n && s[k + 1] === "\n") break;
        k += 1;
      }
      if (close === -1) {
        out.push("$");
        i += 1;
        continue;
      }
      const content = s.slice(i + 1, close);
      if (content.startsWith("`") && content.endsWith("`")) {
        out.push(s.slice(i, close + 1));
      } else {
        out.push("$`" + content + "`$");
      }
      i = close + 1;
      continue;
    }
    out.push(c);
    i += 1;
  }
  return out.join("").replace(/\n{3,}/g, "\n\n");
}

/**
 * Recursively collect Markdown files under a directory, skipping dot-entries.
 *
 * @param {string} dir Directory to walk.
 * @returns {string[]} Absolute-or-relative paths of `.md` files found.
 */
function walkMarkdown(dir) {
  const found = [];
  for (const entry of readdirSync(dir)) {
    if (entry.startsWith(".")) continue;
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) found.push(...walkMarkdown(p));
    else if (entry.endsWith(".md")) found.push(p);
  }
  return found;
}

const roots = process.argv.slice(2);
if (roots.length === 0) roots.push("openwiki");

let changed = 0;
let scanned = 0;
for (const root of roots) {
  for (const file of walkMarkdown(root)) {
    scanned += 1;
    const raw = readFileSync(file, "utf8").replace(/\r\n/g, "\n");
    const [fm, body] = splitFrontmatter(raw);
    const next = fm + transformBody(body);
    if (next !== raw) {
      writeFileSync(file, next, "utf8");
      changed += 1;
      console.log(`fixed  ${file}`);
    }
  }
}
console.log(`\nscanned ${scanned} markdown files, rewrote ${changed}`);
