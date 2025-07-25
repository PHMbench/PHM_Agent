from __future__ import annotations

"""Minimal text-based web browser and related tools."""

from typing import Any
from urllib.parse import urljoin, unquote
import os
import re
import time

import requests

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover - optional import
    BeautifulSoup = None


class SimpleTextBrowser:
    """Simplified text browser for agent use."""

    def __init__(
        self,
        start_page: str | None = None,
        viewport_size: int = 1024 * 8,
        downloads_folder: str | None = None,
        serpapi_key: str | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.start_page = start_page or "about:blank"
        self.viewport_size = viewport_size
        self.downloads_folder = downloads_folder or "downloads"
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs or {"timeout": 60}
        self.history: list[tuple[str, float]] = []
        self.page_title: str | None = None
        self._page_content = ""
        self._viewports: list[tuple[int, int]] = []
        self._viewport = 0
        self._last_query: str | None = None
        self._last_idx: int | None = None
        os.makedirs(self.downloads_folder, exist_ok=True)
        self.set_address(self.start_page)

    # ------------------------------------------------------------------
    # Basic state helpers
    # ------------------------------------------------------------------
    @property
    def address(self) -> str:
        return self.history[-1][0]

    @property
    def viewport(self) -> str:
        start, end = self._viewports[self._viewport]
        return self._page_content[start:end]

    def _split_pages(self) -> None:
        self._viewports = []
        start = 0
        while start < len(self._page_content):
            end = min(start + self.viewport_size, len(self._page_content))
            self._viewports.append((start, end))
            start = end
        if not self._viewports:
            self._viewports = [(0, 0)]
        self._viewport = 0

    def _set_page_content(self, text: str) -> None:
        self._page_content = text
        self._split_pages()

    def _state(self) -> tuple[str, str]:
        header = f"Address: {self.address}\nViewport position: page {self._viewport + 1} of {len(self._viewports)}."
        return header, self.viewport

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def page_down(self) -> None:
        self._viewport = min(self._viewport + 1, len(self._viewports) - 1)

    def page_up(self) -> None:
        self._viewport = max(self._viewport - 1, 0)

    def find_on_page(self, query: str) -> int | None:
        self._last_query = query
        self._last_idx = None
        for i, (s, e) in enumerate(self._viewports):
            if query.lower() in self._page_content[s:e].lower():
                self._viewport = i
                self._last_idx = i
                return i
        return None

    def find_next(self) -> int | None:
        if self._last_query is None:
            return None
        start = (self._last_idx or 0) + 1
        for i in range(start, len(self._viewports)):
            s, e = self._viewports[i]
            if self._last_query.lower() in self._page_content[s:e].lower():
                self._viewport = i
                self._last_idx = i
                return i
        return None

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------
    def visit_page(self, url: str, filter_year: int | None = None) -> str:
        self.set_address(url, filter_year)
        return self.viewport

    def set_address(self, uri: str, filter_year: int | None = None) -> None:
        self.history.append((uri, time.time()))
        if uri == "about:blank":
            self._set_page_content("")
            return
        if uri.startswith("google:"):
            self._serpapi_search(uri[len("google:"):].strip(), filter_year)
        else:
            if not uri.startswith("http") and self.history:
                uri = urljoin(self.history[-2][0], uri)
                self.history[-1] = (uri, self.history[-1][1])
            self._fetch_page(uri)
        self._viewport = 0
        self._last_query = None

    def _fetch_page(self, url: str) -> None:
        if url.startswith("file://"):
            path = unquote(url[7:])
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            self.page_title = os.path.basename(path)
            self._set_page_content(text)
            return
        response = requests.get(url, **self.request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        text = response.text
        if ("html" in content_type.lower()) and BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")
            self.page_title = soup.title.string if soup.title else url
        self._set_page_content(text)

    def _serpapi_search(self, query: str, filter_year: int | None = None) -> None:
        try:
            from serpapi import GoogleSearch
        except Exception as exc:  # pragma: no cover - optional import
            raise RuntimeError("serpapi library is required for web search") from exc
        params = {"engine": "google", "q": query, "api_key": self.serpapi_key}
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
        results = GoogleSearch(params).get_dict()
        snippets = []
        for idx, entry in enumerate(results.get("organic_results", []), 1):
            snippet = f"{idx}. [{entry.get('title','')}]({entry.get('link','')})"
            if entry.get("date"):
                snippet += f"\nDate published: {entry['date']}"
            if entry.get("snippet"):
                snippet += f"\n{entry['snippet']}"
            snippets.append(snippet)
        content = f"A Google search for '{query}' found {len(snippets)} results:\n\n" + "\n\n".join(snippets)
        self.page_title = f"{query} - Search"
        self._set_page_content(content)


# ---------------------------------------------------------------------------
# Tools using the browser
# ---------------------------------------------------------------------------
from smolagents import Tool


class VisitTool(Tool):
    """Visit a webpage and return its text content."""

    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text content."
    inputs = {"url": {"type": "string", "description": "The webpage URL."}}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class PageUpTool(Tool):
    """Scroll up one page in the current webpage."""

    name = "page_up"
    description = "Scroll the viewport up one page and return the new content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class PageDownTool(Tool):
    """Scroll down one page in the current webpage."""

    name = "page_down"
    description = "Scroll the viewport down one page and return the new content."
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class FinderTool(Tool):
    """Find a string on the current page."""

    name = "find_on_page_ctrl_f"
    description = "Scroll to the first occurrence of the string on the page."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "String to search for. Supports simple substring matching.",
        }
    }
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()
        if result is None:
            return header.strip() + f"\n=======================\nThe string '{search_string}' was not found."
        return header.strip() + "\n=======================\n" + content


class FindNextTool(Tool):
    """Find the next occurrence of the last search string."""

    name = "find_next"
    description = "Jump to the next occurrence of the previous search string."
    inputs = {}
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        result = self.browser.find_next()
        header, content = self.browser._state()
        if result is None:
            return header.strip() + "\n=======================\nThe search string was not found."
        return header.strip() + "\n=======================\n" + content


class ArchiveSearchTool(Tool):
    """Retrieve an archived version of a URL from the Wayback Machine."""

    name = "find_archived_url"
    description = "Find the archived version of a URL closest to a given date."
    inputs = {
        "url": {"type": "string", "description": "URL to look up."},
        "date": {
            "type": "string",
            "description": "Date in YYYYMMDD format for the desired snapshot.",
        },
    }
    output_type = "string"

    def __init__(self, browser: SimpleTextBrowser) -> None:
        super().__init__()
        self.browser = browser

    def forward(self, url: str, date: str) -> str:
        api = f"https://archive.org/wayback/available?url={url}&timestamp={date}"
        alt_api = f"https://archive.org/wayback/available?url={url}"
        response = requests.get(api).json()
        if not response.get("archived_snapshots", {}).get("closest"):
            response = requests.get(alt_api).json()
        closest = response.get("archived_snapshots", {}).get("closest")
        if not closest:
            raise Exception(f"No archived snapshot found for {url}")
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )


__all__ = [
    "SimpleTextBrowser",
    "VisitTool",
    "PageUpTool",
    "PageDownTool",
    "FinderTool",
    "FindNextTool",
    "ArchiveSearchTool",
]


if __name__ == "__main__":
    browser = SimpleTextBrowser(start_page="about:blank")
    print("Initial address:", browser.address)
