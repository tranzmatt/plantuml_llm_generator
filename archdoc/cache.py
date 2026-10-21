"""
Model response cache (DESIGN.md 2.6): a rerun with no code or model change
makes no model calls.

Each attempt is cached on its own exact request: model, model digest, system
prompt, user prompt (instructions + payload, plus the repair text for a repair
round), response schema and answer budget. The payload already carries
everything the answer may depend on (the code facts, the detail levels, the
prompt version through the instructions), so the key is the request itself and
no separate invalidation is needed: change a function and only the requests
whose payload shows it change. That is also the content-hash cache of the
hierarchical summaries (12.5): a package summary's payload is made of its
modules' summaries, so it is asked again only when one of them changed.

Entries live in `<out>/.archdoc/cache/`, one JSON file per request. Only
answers the server returned are stored; backend errors and truncated prompts
are not.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, Optional


def request_key(model: Optional[str], digest: Optional[str], system: str, user: str,
                schema: Dict[str, Any], max_tokens: int) -> str:
    blob = json.dumps([model, digest, system, user, schema, max_tokens], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _file(self, key: str) -> str:
        return os.path.join(self.path, key[:2], f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with open(self._file(key), encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            self.misses += 1
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, artifact: str, text: str, prompt_tokens: Optional[int],
            completion_tokens: Optional[int]) -> None:
        if not self.enabled:
            return
        path = self._file(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {"artifact": artifact, "text": text, "prompt_tokens": prompt_tokens,
                 "completion_tokens": completion_tokens,
                 "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False)
        os.replace(tmp, path)


__all__ = ["ResponseCache", "request_key"]
