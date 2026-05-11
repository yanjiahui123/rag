from __future__ import annotations

import json
from typing import List, Optional

from rag_service.agent_retrieval.models import SkillPackageResponse


def generate_opencode_skill_package(
    base_url: Optional[str] = None,
    kb_sn_list: Optional[List[str]] = None,
) -> SkillPackageResponse:
    base_url = base_url or "https://your-rag-service.example.com"
    kb_sn_list = list(kb_sn_list or [])
    config = {
        "base_url": base_url,
        "uid": "",
        "kb_sn_list": kb_sn_list,
        "timeout_seconds": 30,
    }
    return SkillPackageResponse(
        files={
            "SKILL.md": _skill_md(base_url, kb_sn_list),
            "config.json": json.dumps(config, ensure_ascii=False, indent=2),
            "references/api_schema.md": _api_schema_md(),
            "scripts/kb_retrieval.py": _client_script(),
        }
    )


def _skill_md(base_url: str, kb_sn_list: List[str]) -> str:
    kb_text = ",".join(kb_sn_list) if kb_sn_list else "kb-sn-1,kb-sn-2"
    return SKILL_TEMPLATE.format(base_url=base_url, kb_text=kb_text)


def _api_schema_md() -> str:
    return API_SCHEMA


def _client_script() -> str:
    return CLIENT_SCRIPT


SKILL_TEMPLATE = """---
name: kb-retrieval
description: Search and explore configured enterprise knowledge bases through a RAG service. Use when answering questions that require evidence from configured KB serial numbers, inspecting sections, tables, or nearby original text.
---

# KB Retrieval

Use this skill to search configured knowledge bases and inspect source evidence. The service returns retrieval evidence only; you decide whether more exploration is needed and write the final answer.

## Configuration

Edit `config.json` in this skill folder:

```json
{{
  "base_url": "{base_url}",
  "uid": "your-employee-id",
  "kb_sn_list": ["{kb_text}"],
  "timeout_seconds": 30
}}
```

Environment variables can override the file when needed:

```bash
export KB_RETRIEVAL_BASE_URL="{base_url}"
export KB_RETRIEVAL_UID="your-employee-id"
export KB_SN_LIST="{kb_text}"
```

## Workflow

1. Start every knowledge-base question with search:

```bash
python scripts/kb_retrieval.py search --query "..." --top-k 20
```

2. Treat returned slices as candidate evidence. If a slice needs surrounding context, fetch the section:

```bash
python scripts/kb_retrieval.py section --section-handle "..."
```

3. If the slice is from a table, fetch the table in `llm_text` or `summary` mode:

```bash
python scripts/kb_retrieval.py table --table-handle "..." --mode llm_text
```

4. For broad or structural questions, inspect the document outline:

```bash
python scripts/kb_retrieval.py outline --document-handle "..."
```

5. If the slice is too narrow, fetch neighboring original text:

```bash
python scripts/kb_retrieval.py original-text --document-handle "..." --center-block-id "block_001" --before 2 --after 2
```

6. If results are weak, rewrite the query locally and search again.

Only answer with facts grounded in returned slices, sections, tables, or original text. Cite document titles and section/table names when available.
"""


API_SCHEMA = """# Agent Retrieval API

Base path: `/agent/retrieval`

## `POST /search_slices`

Body:

```json
{{"uid": "employee-id", "query": "question", "kb_sn_list": ["kb-1"], "top_k": 20}}
```

Returns ranked slices with document metadata, location metadata, action flags, and opaque handles.

## `POST /get_document_outline`

Body:

```json
{{"uid": "employee-id", "document_handle": "..."}}
```

Returns sections and tables discovered from the document manifest.

## `POST /get_section`

Body:

```json
{{"uid": "employee-id", "section_handle": "...", "max_chars": 12000}}
```

Returns section markdown text.

## `POST /get_table`

Body:

```json
{{"uid": "employee-id", "table_handle": "...", "mode": "llm_text"}}
```

Modes: `llm_text`, `json`, `html`, `summary`.

## `POST /get_original_text`

Body:

```json
{{"uid": "employee-id", "document_handle": "...", "center_block_id": "block_001", "before": 2, "after": 2}}
```

Returns bounded neighboring block text when available.
"""


CLIENT_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search and explore configured RAG knowledge bases.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON responses.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=20)

    outline = subparsers.add_parser("outline")
    outline.add_argument("--document-handle", required=True)

    section = subparsers.add_parser("section")
    section.add_argument("--section-handle", required=True)
    section.add_argument("--max-chars", type=int, default=12000)

    table = subparsers.add_parser("table")
    table.add_argument("--table-handle", required=True)
    table.add_argument("--mode", choices=["llm_text", "json", "html", "summary"], default="llm_text")
    table.add_argument("--max-chars", type=int, default=40000)

    original = subparsers.add_parser("original-text")
    original.add_argument("--document-handle", required=True)
    original.add_argument("--center-block-id")
    original.add_argument("--before", type=int, default=2)
    original.add_argument("--after", type=int, default=2)
    original.add_argument("--max-chars", type=int, default=12000)

    args = parser.parse_args(argv)
    config = load_config(args.config)
    payload = build_payload(args, config)
    result = post_json(config["base_url"], endpoint(args.command), payload, config.get("timeout_seconds", 30))
    print_json(result, pretty=args.pretty)


def load_config(path):
    config = {}
    config_path = Path(path) if path else default_config_path()
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    if os.getenv("KB_RETRIEVAL_BASE_URL"):
        config["base_url"] = os.environ["KB_RETRIEVAL_BASE_URL"]
    if os.getenv("KB_RETRIEVAL_UID"):
        config["uid"] = os.environ["KB_RETRIEVAL_UID"]
    if os.getenv("KB_SN_LIST"):
        config["kb_sn_list"] = [item.strip() for item in os.environ["KB_SN_LIST"].split(",") if item.strip()]
    if not config.get("base_url"):
        raise SystemExit("Missing KB_RETRIEVAL_BASE_URL or config base_url.")
    config.setdefault("kb_sn_list", [])
    return config


def default_config_path():
    return Path(__file__).resolve().parents[1] / "config.json"


def build_payload(args, config):
    if args.command == "search":
        return {"uid": config.get("uid"), "query": args.query, "kb_sn_list": config.get("kb_sn_list", []), "top_k": args.top_k}
    if args.command == "outline":
        return {"uid": config.get("uid"), "document_handle": args.document_handle}
    if args.command == "section":
        return {"uid": config.get("uid"), "section_handle": args.section_handle, "max_chars": args.max_chars}
    if args.command == "table":
        return {"uid": config.get("uid"), "table_handle": args.table_handle, "mode": args.mode, "max_chars": args.max_chars}
    if args.command == "original-text":
        return {
            "uid": config.get("uid"),
            "document_handle": args.document_handle,
            "center_block_id": args.center_block_id,
            "before": args.before,
            "after": args.after,
            "max_chars": args.max_chars,
        }
    raise SystemExit(f"Unsupported command: {args.command}")


def endpoint(command):
    return {
        "search": "/agent/retrieval/search_slices",
        "outline": "/agent/retrieval/get_document_outline",
        "section": "/agent/retrieval/get_section",
        "table": "/agent/retrieval/get_table",
        "original-text": "/agent/retrieval/get_original_text",
    }[command]


def post_json(base_url, path, payload, timeout):
    url = base_url.rstrip("/") + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc


def print_json(value, pretty=False):
    if pretty:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
'''
