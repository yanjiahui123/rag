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
    request = {
        "command": "search",
        "pretty": True,
        "utf8_output": True,
        "query": "",
        "top_k": 20,
        "retrieval_backend": "libing",
        "enable_rerank": False,
    }
    return SkillPackageResponse(
        files={
            "SKILL.md": _skill_md(base_url, kb_sn_list),
            "config.json": json.dumps(config, ensure_ascii=False, indent=2),
            "request.json": json.dumps(request, ensure_ascii=False, indent=2),
            "references/api_schema.md": _api_schema_md(),
            "scripts/kb_retrieval.py": _client_script(),
            "scripts/kb_retrieval.sh": _shell_wrapper(),
            "scripts/kb_retrieval_request.py": _request_runner_script(),
            "scripts/kb_retrieval_request.sh": _request_shell_wrapper(),
        }
    )


def _skill_md(base_url: str, kb_sn_list: List[str]) -> str:
    kb_text = ",".join(kb_sn_list) if kb_sn_list else "kb-sn-1,kb-sn-2"
    kb_json = json.dumps(kb_sn_list or ["kb-sn-1", "kb-sn-2"], ensure_ascii=False)
    return SKILL_TEMPLATE.format(base_url=base_url, kb_text=kb_text, kb_json=kb_json)


def _api_schema_md() -> str:
    return API_SCHEMA


def _client_script() -> str:
    return CLIENT_SCRIPT


def _shell_wrapper() -> str:
    return SHELL_WRAPPER


def _request_runner_script() -> str:
    return REQUEST_RUNNER_SCRIPT


def _request_shell_wrapper() -> str:
    return REQUEST_SHELL_WRAPPER


SKILL_TEMPLATE = """---
name: kb-retrieval
description: Use when answering questions that require evidence from configured enterprise knowledge bases, internal documents, troubleshooting records, SOPs, manuals, historical cases, document sections, tables, outlines, or nearby original text.
---

# KB Retrieval

Use this skill to search configured knowledge bases and inspect source evidence. The service returns retrieval evidence only; you decide whether more exploration is needed and write the final answer.

## Execution Rule

Prefer the request-file runner. It avoids passing Chinese text, quotes, or dynamic arguments through shell argv.

First edit `.opencode/skills/kb-retrieval/request.json`, then run this fixed no-argument command from the workspace root:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

If already running from inside the `kb-retrieval` skill directory, use:

```bash
python scripts/kb_retrieval_request.py
```

Do not pass the search query on the shell command line for normal use. Put dynamic values in `request.json`.

Do not prepend a Windows drive-path `cd` before the command when the shell is bash. Use the tool's working-directory option when available. If an absolute Windows path is necessary, use forward slashes in the Python script path, for example `python D:/project/knowledge_base_answer/.opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py`.

By default, the request runner prints raw UTF-8 so Chinese text remains readable. If the terminal or agent log path displays mojibake, set `"utf8_output": false` in `request.json` to switch back to ASCII-safe JSON.

## Configuration

Edit `config.json` in this skill folder:

```json
{{
  "base_url": "{base_url}",
  "uid": "your-employee-id",
  "kb_sn_list": {kb_json},
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

1. Start every knowledge-base question by editing `request.json`:

```json
{{
  "command": "search",
  "pretty": true,
  "utf8_output": true,
  "query": "USER_QUERY_HERE",
  "top_k": 20,
  "retrieval_backend": "libing",
  "enable_rerank": false
}}
```

Then run:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

Read the returned titles, snippets, scores, short OBS-key document handles, section handles, table handles, and block identifiers.

2. Treat returned slices as candidate evidence. If a slice needs surrounding context, edit `request.json` with the returned handle:

```json
{{
  "command": "section",
  "pretty": true,
  "section_handle": "...",
  "max_chars": 12000
}}
```

Then run:

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

3. If the slice is from a table, fetch the table in `llm_text` or `summary` mode:

```json
{{
  "command": "table",
  "pretty": true,
  "table_handle": "...",
  "mode": "llm_text",
  "max_chars": 40000
}}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

4. For broad or structural questions, inspect the document outline:

```json
{{
  "command": "outline",
  "pretty": true,
  "document_handle": "..."
}}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

5. If the slice is too narrow, fetch neighboring original text:

```json
{{
  "command": "original-text",
  "pretty": true,
  "document_handle": "...",
  "center_block_id": "block_001",
  "before": 2,
  "after": 2,
  "max_chars": 12000
}}
```

```bash
python .opencode/skills/kb-retrieval/scripts/kb_retrieval_request.py
```

6. Use `"retrieval_backend": "ipd"` only when the selected knowledge bases are mapped to IPD RAG. Set `"enable_rerank": true` when higher-quality ordering is worth expanding retrieval to 100 candidates before returning the requested `top_k`.

7. If results are weak, rewrite the query locally and search again.

Only answer with facts grounded in returned slices, sections, tables, or original text. Cite document titles and section/table names when available.

## Failure Handling

If `python` is unavailable, retry the same command with `python3`.

If the request runner path does not exist, run `pwd` and list `.opencode/skills/kb-retrieval/scripts/` to confirm where the skill was installed. Retry with the actual `kb_retrieval_request.py` path.

If bash reports that `cd` cannot find a path like `D:project...`, the command used a Windows backslash path under bash. Retry without `cd`, set the tool working directory instead, or use a forward-slash absolute Python script path.

If output is garbled, set `"utf8_output": false` in `request.json` and retry. ASCII-safe JSON uses `\\uXXXX` escapes and is intended for unreliable encoding paths.

For manual terminal debugging only, `scripts/kb_retrieval.sh` accepts normal CLI arguments.
"""


API_SCHEMA = """# Agent Retrieval API

Base path: `/agent/retrieval`

## `POST /search_slices`

Body:

```json
{"uid": "employee-id", "query": "question", "kb_sn_list": ["kb-1"], "top_k": 20, "retrieval_backend": "libing", "enable_rerank": false}
```

Returns ranked slices with document metadata, location metadata, action flags, and short OBS-key handles.

`retrieval_backend` is either `libing` or `ipd` and selects exactly one retrieval system for a request. When `enable_rerank` is true, retrieval expands to 100 candidates before returning the requested result limit.

## `POST /get_document_outline`

Body:

```json
{"uid": "employee-id", "document_handle": "..."}
```

Returns sections and tables discovered from the document manifest. `document_handle` is a `.../manifest.json` OBS key.

## `POST /get_section`

Body:

```json
{"uid": "employee-id", "section_handle": "...", "max_chars": 12000}
```

Returns section markdown text. `section_handle` is a `.../sections/<section>.md` OBS key.

## `POST /get_table`

Body:

```json
{"uid": "employee-id", "table_handle": "...", "mode": "llm_text"}
```

Modes: `llm_text`, `json`, `html`, `summary`. `table_handle` is a `.../tables/<table>.llm.md` OBS key; sibling `.json` and `.html` artifacts are derived from it when those modes are requested.

## `POST /get_original_text`

Body:

```json
{"uid": "employee-id", "document_handle": "...", "center_block_id": "block_001", "before": 2, "after": 2}
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
    configure_stdio()
    parser = argparse.ArgumentParser(description="Search and explore configured RAG knowledge bases.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON responses.")
    parser.add_argument("--utf8-output", action="store_true", help="Print raw UTF-8 instead of ASCII-escaped JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--retrieval-backend", choices=["libing", "ipd"], default="libing")
    search.add_argument("--enable-rerank", action="store_true")

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
    print_json(result, pretty=args.pretty, utf8_output=args.utf8_output)


def configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


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
        return {
            "uid": config.get("uid"),
            "query": args.query,
            "kb_sn_list": config.get("kb_sn_list", []),
            "top_k": args.top_k,
            "retrieval_backend": args.retrieval_backend,
            "enable_rerank": args.enable_rerank,
        }
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


def print_json(value, pretty=False, utf8_output=False):
    ensure_ascii = not utf8_output
    if pretty:
        print(json.dumps(value, ensure_ascii=ensure_ascii, indent=2))
    else:
        print(json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":")))


if __name__ == "__main__":
    main()
'''


SHELL_WRAPPER = r'''#!/usr/bin/env bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/kb_retrieval.py" "$@"
'''


REQUEST_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import kb_retrieval


def main():
    request = load_request()
    kb_retrieval.main(build_argv(request))


def load_request():
    request_path = Path(os.getenv("KB_RETRIEVAL_REQUEST_FILE") or SKILL_DIR / "request.json")
    if not request_path.exists():
        raise SystemExit(f"Missing request file: {request_path}")
    with request_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_argv(request):
    command = require(request, "command")
    argv = []
    if request.get("config"):
        argv.extend(["--config", resolve_path(request["config"])])
    if request.get("pretty", True):
        argv.append("--pretty")
    if request.get("utf8_output", True):
        argv.append("--utf8-output")
    argv.append(command)

    if command == "search":
        argv.extend(["--query", require(request, "query")])
        argv.extend(["--top-k", str(request.get("top_k", request.get("top-k", 20)))])
        argv.extend(["--retrieval-backend", request.get("retrieval_backend", "libing")])
        if request.get("enable_rerank", False):
            argv.append("--enable-rerank")
    elif command == "outline":
        argv.extend(["--document-handle", require(request, "document_handle")])
    elif command == "section":
        argv.extend(["--section-handle", require(request, "section_handle")])
        argv.extend(["--max-chars", str(request.get("max_chars", request.get("max-chars", 12000)))])
    elif command == "table":
        argv.extend(["--table-handle", require(request, "table_handle")])
        argv.extend(["--mode", request.get("mode", "llm_text")])
        argv.extend(["--max-chars", str(request.get("max_chars", request.get("max-chars", 40000)))])
    elif command == "original-text":
        argv.extend(["--document-handle", require(request, "document_handle")])
        if request.get("center_block_id"):
            argv.extend(["--center-block-id", request["center_block_id"]])
        argv.extend(["--before", str(request.get("before", 2))])
        argv.extend(["--after", str(request.get("after", 2))])
        argv.extend(["--max-chars", str(request.get("max_chars", request.get("max-chars", 12000)))])
    else:
        raise SystemExit(f"Unsupported request command: {command}")
    return argv


def require(request, key):
    value = request.get(key)
    if value in (None, ""):
        raise SystemExit(f"request.json missing required field: {key}")
    return str(value)


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = SKILL_DIR / path
    return str(path)


if __name__ == "__main__":
    main()
'''


REQUEST_SHELL_WRAPPER = r'''#!/usr/bin/env bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/kb_retrieval_request.py"
'''
