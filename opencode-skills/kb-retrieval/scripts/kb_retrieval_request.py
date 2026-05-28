#!/usr/bin/env python3
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
        if request.get("enable_rerank", True):
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
