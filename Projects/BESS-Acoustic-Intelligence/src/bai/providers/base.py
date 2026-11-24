from typing import Protocol

from bai.models import SearchResult


class SearchProviderError(RuntimeError):
    """A search provider request failed and should not be retried automatically."""


class FatalSearchProviderError(SearchProviderError):
    """A provider error that should stop further API calls in this run."""


class TransientSearchError(SearchProviderError):
    """A temporary provider/network failure that may be retried."""


class SearchProvider(Protocol):
    name: str

    def search(
        self, query: str, *, search_depth: str, max_results: int
    ) -> list[SearchResult]:
        """Return search results for a query."""
