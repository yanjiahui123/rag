#!/usr/bin/env python3
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
    result = post_json(
        config["base_url"],
        endpoint(args.command),
        payload,
        config.get("timeout_seconds", 30),
        config.get("headers", {}),
    )
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
    config.setdefault("headers", {})
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


def post_json(base_url, path, payload, timeout, headers=None):
    url = base_url.rstrip("/") + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    headers = headers or {}
    for header_name in ("X-HW-ID", "X-HW-APPKEY"):
        header_value = headers.get(header_name)
        if header_value:
            request.add_header(header_name, str(header_value))
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
