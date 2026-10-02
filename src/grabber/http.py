"""HTTP-клиент grabber'а: троттлинг, повторы с backoff, уважение Retry-After."""
from __future__ import annotations

import logging
import time

import requests
from tenacity import (Retrying, retry_if_exception, stop_after_attempt,
                      wait_exponential_jitter)

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    def __init__(self, status: int, retry_after: float | None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableHTTPError, requests.ConnectionError, requests.Timeout))


class HttpClient:
    def __init__(self, user_agent: str, delay_sec: float = 0.3, timeout_sec: float = 30, max_retries: int = 5):
        self.session = requests.Session()
        # requests сам отправляет Accept-Encoding: gzip и распаковывает ответ
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay_sec
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self._last = 0.0
        self.requests_made = 0

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get_once(self, url: str, params: dict | None) -> requests.Response:
        self._throttle()
        self.requests_made += 1
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code in RETRY_STATUS:
            ra = r.headers.get("Retry-After")
            retry_after = float(ra) if ra and ra.isdigit() else None
            if retry_after:
                log.warning("HTTP %s, сервер просит подождать %ss", r.status_code, retry_after)
                time.sleep(min(retry_after, 120))
            raise RetryableHTTPError(r.status_code, retry_after)
        r.raise_for_status()                      # 4xx (кроме 429) — не повторяем
        return r

    def get(self, url: str, params: dict | None = None) -> requests.Response:
        for attempt in Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
            before_sleep=lambda s: log.warning("повтор %s для %s: %s", s.attempt_number, url, s.outcome.exception()),
        ):
            with attempt:
                return self._get_once(url, params)
        raise RuntimeError("unreachable")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return self.get(url, params).json()
