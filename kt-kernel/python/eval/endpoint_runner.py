from __future__ import annotations

import requests


class EndpointClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_completions(self, payload: dict) -> dict:
        response = requests.post(
            self.base_url + "/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def post_generate(self, payload: dict) -> dict:
        base_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        response = requests.post(
            base_url + "/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
