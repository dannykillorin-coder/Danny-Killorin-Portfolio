import time
import urllib.robotparser
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

import requests


USER_AGENT = "BESSAcousticIntelligenceResearchBot/0.1 (+public-document research)"


class PoliteWebClient:
    def __init__(
        self,
        *,
        timeout: float,
        rate_limit_seconds: float,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.timeout = timeout
        self.rate_limit_seconds = rate_limit_seconds
        self.session = session or requests.Session()
        self.sleeper = sleeper
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_denied: set[str] = set()
        self._robots_delays: dict[str, float] = {}

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
        current_url = url
        for _ in range(6):
            if not self._allowed_by_robots(current_url):
                raise PermissionError(f"Access disallowed by robots.txt: {current_url}")
            self._wait_for_host(current_url)
            response = self.session.get(
                current_url,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise requests.HTTPError(f"Redirect from {current_url} has no Location header")
                current_url = urljoin(current_url, location)
                continue
            return response
        raise requests.TooManyRedirects(f"Too many redirects for {url}")

    def _allowed_by_robots(self, url: str) -> bool:
        parts = urlsplit(url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            return False
        try:
            origin = _origin(url)
        except ValueError:
            return False
        if origin in self._robots_denied:
            return False
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                self._wait_for_host(robots_url)
                response = self.session.get(
                    robots_url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.RequestException:
                self._robots_denied.add(origin)
                return False
            self._last_request[origin] = time.monotonic()
            try:
                if response.status_code in {404, 410}:
                    parser.parse([])
                elif response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    self._robots_denied.add(origin)
                    return False
            finally:
                response.close()
            self._robots[origin] = parser
            delay = parser.crawl_delay(USER_AGENT) or 0
            request_rate = parser.request_rate(USER_AGENT)
            if request_rate and request_rate.requests > 0:
                delay = max(delay, request_rate.seconds / request_rate.requests)
            self._robots_delays[origin] = delay
        parser = self._robots[origin]
        return parser is not None and parser.can_fetch(USER_AGENT, url)

    def _wait_for_host(self, url: str) -> None:
        host = _origin(url)
        required_delay = max(
            self.rate_limit_seconds,
            self._robots_delays.get(host, 0.0),
        )
        previous = self._last_request.get(host)
        if previous is not None:
            remaining = required_delay - (time.monotonic() - previous)
            if remaining > 0:
                self.sleeper(remaining)
        self._last_request[host] = time.monotonic()


def _origin(url: str) -> str:
    parts = urlsplit(url)
    if not parts.hostname:
        raise ValueError(f"Invalid URL host: {url}")
    scheme = parts.scheme.lower()
    hostname = parts.hostname.lower()
    port = parts.port
    if ":" in hostname:
        hostname = f"[{hostname}]"
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        hostname = f"{hostname}:{port}"
    return f"{scheme}://{hostname}"
