from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Callable, Dict, Optional


DEFAULT_HANDLE_TTL_SECONDS = 3600


class AgentRetrievalHandleCodec:
    def __init__(
        self,
        secret: Optional[str] = None,
        ttl_seconds: int = DEFAULT_HANDLE_TTL_SECONDS,
        time_func: Callable[[], float] = time.time,
    ):
        self.secret = (secret or os.getenv("AGENT_RETRIEVAL_HANDLE_SECRET") or "agent-retrieval-local-secret").encode(
            "utf-8"
        )
        self.ttl_seconds = max(int(ttl_seconds or DEFAULT_HANDLE_TTL_SECONDS), 1)
        self.time_func = time_func

    def encode(self, payload: Dict[str, Any], uid: str) -> str:
        body = {
            "uid": uid,
            "exp": int(self.time_func()) + self.ttl_seconds,
            "payload": dict(payload),
        }
        body_token = _b64encode(_json_bytes(body))
        signature = _sign(body_token, self.secret)
        return f"{body_token}.{signature}"

    def decode(self, handle: str, expected_uid: Optional[str] = None) -> Dict[str, Any]:
        handle = handle.strip()
        try:
            body_token, signature = handle.split(".", 1)
        except ValueError as exc:
            raise PermissionError("invalid retrieval handle") from exc
        if not hmac.compare_digest(_sign(body_token, self.secret), signature):
            raise PermissionError("invalid retrieval handle")
        try:
            body = json.loads(_b64decode(body_token).decode("utf-8"))
        except Exception as exc:
            raise PermissionError("invalid retrieval handle") from exc
        if int(body.get("exp") or 0) < int(self.time_func()):
            raise PermissionError("expired retrieval handle")
        uid = body.get("uid")
        if expected_uid is not None and uid != expected_uid:
            raise PermissionError("retrieval handle belongs to a different user")
        payload = body.get("payload")
        return dict(payload) if isinstance(payload, dict) else {}


def _json_bytes(value: Dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sign(body_token: str, secret: bytes) -> str:
    digest = hmac.new(secret, body_token.encode("ascii"), hashlib.sha256).digest()
    return _b64encode(digest)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
