import os

import requests

from bai.models import SearchResult
from bai.providers.base import (
    FatalSearchProviderError,
    SearchProviderError,
    TransientSearchError,
)


class TavilyProvider:
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        client=None,
        timeout_seconds: float = 20.0,
    ):
        if not 0 < timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be greater than zero and at most 120")
        self.timeout_seconds = timeout_seconds
        if client is not None:
            self.client = client
            return
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to the project .env file or environment."
            )
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "Tavily support is unavailable; install the project requirements."
            ) from exc
        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self, query: str, *, search_depth: str, max_results: int
    ) -> list[SearchResult]:
        if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            raise ValueError(
                "search_depth must be 'basic', 'advanced', 'fast', or 'ultra-fast'"
            )
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        try:
            from tavily.errors import (
                BadRequestError,
                ForbiddenError,
                InvalidAPIKeyError,
                KeylessUnsupportedEndpointError,
                TavilyKeylessLimitError,
                UsageLimitExceededError,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Tavily support is unavailable; install the project requirements."
            ) from exc
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                timeout=self.timeout_seconds,
            )
        except (requests.RequestException, TimeoutError) as exc:
            raise TransientSearchError(f"Tavily request failed: {exc}") from exc
        except (
            ForbiddenError,
            InvalidAPIKeyError,
            KeylessUnsupportedEndpointError,
            TavilyKeylessLimitError,
            UsageLimitExceededError,
        ) as exc:
            raise FatalSearchProviderError(f"Tavily search stopped: {exc}") from exc
        except BadRequestError as exc:
            raise SearchProviderError(f"Tavily search failed: {exc}") from exc
        results = response.get("results", [])
        return [
            SearchResult(
                title=str(result.get("title") or ""),
                url=str(result.get("url") or ""),
                content=str(result.get("content") or ""),
            )
            for result in results
            if result.get("url")
        ]
