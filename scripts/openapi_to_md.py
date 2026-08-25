"""TEMP: render an OpenAPI 3.x JSON spec to readable Markdown docs.

Zero deps (stdlib only). Groups operations by tag, renders parameters,
request/response bodies (resolving $ref to schema links), and a full
components/schemas reference with enums and property tables.

    python scripts/openapi_to_md.py docs/api.json docs/api.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

METHOD_ORDER = ["get", "post", "put", "patch", "delete", "head", "options"]


def _anchor(name: str) -> str:
    """GitHub-style anchor slug for a heading."""
    slug = name.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]+", "-", slug)


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _schema_link(name: str) -> str:
    return f"[`{name}`](#schema-{_anchor(name)})"


def type_str(schema: dict[str, Any] | None) -> str:
    """One-line human type for a schema node (resolves $ref, arrays, enums)."""
    if not schema:
        return "any"
    if "$ref" in schema:
        return _schema_link(_ref_name(schema["$ref"]))
    t = schema.get("type")
    if t == "array":
        return f"array<{type_str(schema.get('items'))}>"
    if schema.get("enum"):
        return "enum(" + ", ".join(f"`{v}`" for v in schema["enum"]) + ")"
    fmt = schema.get("format")
    if t and fmt:
        return f"{t} ({fmt})"
    if t == "object" and "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        inner = type_str(ap) if isinstance(ap, dict) else "any"
        return f"map<string, {inner}>"
    return t or "object"


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _rate_limit(text: str | None) -> str:
    if not text:
        return ""
    m = re.search(r"Rate limit:\**\s*([^\n*]+)", text)
    return m.group(1).strip() if m else ""


def render_operation(path: str, method: str, op: dict[str, Any]) -> list[str]:
    out: list[str] = []
    summary = op.get("summary") or op.get("operationId") or f"{method.upper()} {path}"
    out.append(f"### {method.upper()} `{path}`")
    out.append("")
    out.append(f"**{summary}**")
    out.append("")
    desc = (op.get("description") or "").strip()
    rl = _rate_limit(desc)
    # Strip the trailing "Rate limit:" line from the body; show it as its own field.
    body = re.sub(r"\**Rate limit:\**.*$", "", desc, flags=re.S).strip()
    if body:
        out.append(body)
        out.append("")
    meta: list[str] = []
    if op.get("operationId"):
        meta.append(f"- **operationId:** `{op['operationId']}`")
    if rl:
        meta.append(f"- **Rate limit:** {rl}")
    if op.get("tags"):
        meta.append(f"- **Tags:** {', '.join(op['tags'])}")
    if meta:
        out += meta + [""]

    params = op.get("parameters") or []
    if params:
        out.append("**Parameters**")
        out.append("")
        out.append("| Name | In | Required | Type | Description |")
        out.append("|------|----|----------|------|-------------|")
        for p in params:
            req = "yes" if p.get("required") else "no"
            desc_p = _first_line(p.get("description")).replace("|", "\\|")
            out.append(
                f"| `{p.get('name', '')}` | {p.get('in', '')} | {req} | "
                f"{type_str(p.get('schema'))} | {desc_p} |"
            )
        out.append("")

    rb = op.get("requestBody")
    if rb:
        schema = (
            rb.get("content", {}).get("application/json", {}).get("schema", {})
        )
        req = "required" if rb.get("required") else "optional"
        out.append(f"**Request body** ({req}): {type_str(schema)}")
        out.append("")

    responses = op.get("responses") or {}
    if responses:
        out.append("**Responses**")
        out.append("")
        out.append("| Code | Description | Body |")
        out.append("|------|-------------|------|")
        for code in sorted(responses):
            r = responses[code]
            schema = (
                r.get("content", {}).get("application/json", {}).get("schema", {})
            )
            body_t = type_str(schema) if schema else "—"
            rdesc = _first_line(r.get("description")).replace("|", "\\|")
            out.append(f"| {code} | {rdesc} | {body_t} |")
        out.append("")
    return out


def render_schema(name: str, schema: dict[str, Any]) -> list[str]:
    out = [f"### <a id=\"schema-{_anchor(name)}\"></a>`{name}`", ""]
    if schema.get("description"):
        out += [schema["description"].strip(), ""]

    if schema.get("enum") and schema.get("type") == "string":
        out.append("**Enum:** " + ", ".join(f"`{v}`" for v in schema["enum"]))
        out.append("")
        return out

    props = schema.get("properties")
    if props:
        required = set(schema.get("required", []))
        out.append("| Property | Type | Required | Description |")
        out.append("|----------|------|----------|-------------|")
        for pname, pschema in props.items():
            req = "yes" if pname in required else ""
            d = _first_line(pschema.get("description")).replace("|", "\\|")
            out.append(
                f"| `{pname}` | {type_str(pschema)} | {req} | {d} |"
            )
        out.append("")
    elif schema.get("type"):
        out += [f"Type: {type_str(schema)}", ""]
    return out


def convert(spec: dict[str, Any]) -> str:
    info = spec.get("info", {})
    lines: list[str] = [f"# {info.get('title', 'API')} — `{info.get('version', '')}`", ""]
    lines.append(
        f"> Generated from OpenAPI `{spec.get('openapi', '?')}` by "
        "`scripts/openapi_to_md.py`. Do not edit by hand."
    )
    lines.append("")
    if info.get("description"):
        lines += ["## Overview", "", info["description"].strip(), ""]

    # Security schemes
    sec = spec.get("components", {}).get("securitySchemes")
    if sec:
        lines += ["## Authentication", ""]
        for sname, s in sec.items():
            bits = [f"**{sname}** — type `{s.get('type')}`"]
            if s.get("scheme"):
                bits.append(f"scheme `{s['scheme']}`")
            if s.get("in"):
                bits.append(f"in `{s['in']}` (`{s.get('name', '')}`)")
            line = ", ".join(bits)
            if s.get("description"):
                line += f" — {_first_line(s['description'])}"
            lines += [f"- {line}"]
        lines.append("")

    # Group operations by tag
    paths = spec.get("paths", {})
    by_tag: dict[str, list[tuple[str, str, dict]]] = {}
    for path, methods in paths.items():
        for method, op in methods.items():
            if method not in METHOD_ORDER:
                continue
            tag = (op.get("tags") or ["Endpoints"])[0]
            by_tag.setdefault(tag, []).append((path, method, op))

    # Table of contents
    lines += ["## Endpoints", ""]
    for tag in sorted(by_tag):
        lines.append(f"- [{tag}](#{_anchor(tag)})")
    lines.append("")

    for tag in sorted(by_tag):
        lines += [f"## {tag}", ""]
        ops = sorted(
            by_tag[tag],
            key=lambda t: (t[0], METHOD_ORDER.index(t[1])),
        )
        for path, method, op in ops:
            lines += render_operation(path, method, op)

    # Schemas
    schemas = spec.get("components", {}).get("schemas", {})
    if schemas:
        lines += ["## Schemas", ""]
        for name in sorted(schemas):
            lines += render_schema(name, schemas[name])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/openapi_to_md.py <spec.json> <out.md>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads(src.read_text(encoding="utf-8"))
    md = convert(spec)
    dst.write_text(md, encoding="utf-8")
    n_paths = len(spec.get("paths", {}))
    n_schemas = len(spec.get("components", {}).get("schemas", {}))
    print(f"wrote {dst} ({len(md.splitlines())} lines, {n_paths} paths, {n_schemas} schemas)")


if __name__ == "__main__":
    main()
