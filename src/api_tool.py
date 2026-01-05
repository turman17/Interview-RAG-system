from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import get_settings


def fetch_albums() -> list[dict[str, Any]]:
    settings = get_settings()
    request = urllib.request.Request(
        settings.albums_api_url,
        headers={"User-Agent": "InterviewApp/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.load(response)
            last_error = None
            break
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 * (2**attempt))

    if last_error is not None:
        fallback = _load_local_albums()
        if fallback:
            return fallback
        raise last_error

    if not isinstance(data, list):
        raise ValueError("Albums API did not return a list.")

    return data


def _load_local_albums() -> list[dict[str, Any]]:
    path = Path("data/albums.json")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return data
