from collections import deque
from html.parser import HTMLParser
import logging
from urllib.parse import parse_qs, urljoin, urlsplit

import requests

from bai.database import canonicalize_url
from bai.models import Candidate
from bai.web import PoliteWebClient


logger = logging.getLogger(__name__)


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in {"a", "iframe", "embed", "object", "source"}:
            return
        attr_names = {"a": ("href",), "object": ("data",)}.get(
            tag, ("src", "href")
        )
        values = dict(attrs)
        href = next((values.get(key) for key in attr_names if values.get(key)), None)
        if not href:
            return
        if tag == "a":
            self._href = href
            self._text = []
        else:
            self.links.append((href, values.get("title") or values.get("alt") or ""))

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(" ".join(self._text).split())))
            self._href = None
            self._text = []


def is_pdf_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.path.lower().endswith(".pdf"):
        return True
    query = parse_qs(parts.query.lower())
    return any(
        value == "pdf" or value.endswith(".pdf")
        for key in ("format", "type", "download", "file")
        for value in query.get(key, [])
    )


class LandingPageCrawler:
    def __init__(
        self,
        web: PoliteWebClient,
        *,
        max_depth: int,
        max_pages: int,
    ):
        self.web = web
        self.max_depth = max_depth
        self.max_pages = max_pages

    def discover(self, start: Candidate) -> list[Candidate]:
        if is_pdf_url(start.url):
            return [start]
        if self.max_depth < 0 or self.max_pages <= 0:
            return []

        candidates: list[Candidate] = []
        queue = deque([(start.url, 0, start.title, start.referring_url)])
        visited: set[str] = set()
        start_host = urlsplit(start.url).hostname

        while queue and len(visited) < self.max_pages:
            page_url, depth, page_title, referring_url = queue.popleft()
            try:
                canonical_page = canonicalize_url(page_url)
            except ValueError as exc:
                logger.warning("Skipping invalid landing page URL %s: %s", page_url, exc)
                continue
            if canonical_page in visited:
                continue
            visited.add(canonical_page)
            try:
                response = self.web.get(page_url)
                response.raise_for_status()
            except (requests.RequestException, PermissionError) as exc:
                logger.info("Skipping inaccessible landing page %s: %s", page_url, exc)
                continue
            with response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "pdf" in content_type:
                    candidates.append(
                        Candidate(
                            url=page_url,
                            title=page_title,
                            query=start.query,
                            provider=start.provider,
                            referring_url=referring_url,
                        )
                    )
                    continue
                if "html" not in content_type and "xhtml" not in content_type:
                    continue
                parser = _LinkParser()
                parser.feed(response.text)
                links = parser.links

            for href, link_title in links:
                absolute_url = urljoin(page_url, href)
                try:
                    canonical_url = canonicalize_url(absolute_url)
                except ValueError:
                    continue
                if is_pdf_url(absolute_url):
                    candidates.append(
                        Candidate(
                            url=absolute_url,
                            title=link_title or page_title,
                            query=start.query,
                            provider=start.provider,
                            referring_url=page_url,
                        )
                    )
                elif depth < self.max_depth and urlsplit(absolute_url).hostname == start_host:
                    queue.append(
                        (absolute_url, depth + 1, link_title or page_title, page_url)
                    )

        return candidates
