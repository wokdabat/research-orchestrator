"""
fetchers.py — Web search and content extraction

No API keys required. Uses:
  - DuckDuckGo HTML search for query → URL resolution
  - httpx for async HTTP fetching
  - BeautifulSoup + heuristic extraction for clean content

Public interface (used by server.py):
  search_and_fetch(query, max_results) -> list[FetchResult]
  fetch_url(url)                        -> FetchResult

Both are async. Drop-in replaceable with Tavily/Brave by
implementing the same FetchResult dataclass and function signatures.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup, Comment


# ── Config ────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT          = httpx.Timeout(12.0, connect=6.0)
MAX_CONTENT_CHARS = 8_000     # chars fed to SourceAgent (~2k tokens)
MIN_CONTENT_CHARS = 120       # below this = page likely blocked/empty
MAX_CONCURRENT   = 5          # parallel fetch limit

# Tags that are never useful — strip entirely including children
STRIP_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "header", "footer", "nav", "aside", "form", "figure",
    "advertisement", "banner",
}

# Tags whose text content we keep but the tag itself is unwrapped
UNWRAP_TAGS = {"span", "div", "section", "article", "main"}

# CSS selectors for known boilerplate regions — removed before extraction
BOILERPLATE_SELECTORS = [
    "[class*='cookie']", "[class*='banner']", "[class*='popup']",
    "[class*='subscribe']", "[class*='newsletter']", "[class*='ad-']",
    "[class*='sidebar']", "[class*='related']", "[class*='recommended']",
    "[id*='cookie']", "[id*='banner']", "[id*='popup']", "[id*='sidebar']",
    "[role='navigation']", "[role='banner']", "[role='complementary']",
    "[aria-label*='advertisement']",
]


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    url: str
    title: str
    content: str                    # cleaned plain text, ≤ MAX_CONTENT_CHARS
    success: bool
    error: str | None               = None
    fetched_at: str                 = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_usable(self) -> bool:
        return self.success and len(self.content) >= MIN_CONTENT_CHARS


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

async def _ddg_search(query: str, max_results: int = 5) -> list[str]:
    """
    Scrape DuckDuckGo HTML results for a query.
    Returns a list of result URLs (no JS, no API key).
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            return []

    soup = BeautifulSoup(resp.text, "lxml")
    urls: list[str] = []

    for a in soup.select("a.result__url, a.result__a"):
        href = a.get("href", "")
        # DDG wraps URLs in a redirect — extract the real URL
        if "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            params = urllib.parse.parse_qs(parsed.query)
            real = params.get("uddg", [""])[0]
            if real:
                href = urllib.parse.unquote(real)
        # Skip DDG internal, ads, and non-http links
        if href.startswith("http") and "duckduckgo.com" not in href:
            if href not in urls:
                urls.append(href)
        if len(urls) >= max_results:
            break

    return urls


# ── Content extraction ────────────────────────────────────────────────────────

def _extract_title(soup: BeautifulSoup) -> str:
    """Best-effort title extraction."""
    for sel in ["h1", "title", 'meta[property="og:title"]']:
        tag = soup.select_one(sel)
        if tag:
            return (tag.get("content") or tag.get_text()).strip()[:200]
    return "Untitled"


def _remove_boilerplate(soup: BeautifulSoup) -> None:
    """Remove known boilerplate regions in-place."""
    # Strip by tag name
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Strip HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Strip by CSS selector patterns
    for sel in BOILERPLATE_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()


def _score_block(text: str) -> float:
    """
    Heuristic content score for a text block.
    Higher = more likely to be real content vs. nav/footer boilerplate.
    """
    if not text:
        return 0.0
    words       = text.split()
    word_count  = len(words)
    if word_count < 8:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / word_count
    # Penalize very short average word length (nav links, menu items)
    length_score = min(word_count / 40, 1.0)
    density_score = min(avg_word_len / 5.5, 1.0)
    # Reward sentence-like text (ends with punctuation)
    punct_ratio = sum(1 for w in words if w[-1] in ".!?,:;") / max(word_count, 1)
    return length_score * 0.5 + density_score * 0.3 + punct_ratio * 0.2


def _extract_content(html: str) -> tuple[str, str]:
    """
    Extract (title, clean_text) from raw HTML.

    Strategy:
    1. Remove boilerplate tags/regions
    2. Find the highest-scoring content block (article, main, or largest <div>)
    3. Walk its text nodes, score each paragraph, keep top content
    4. Normalise whitespace, cap at MAX_CONTENT_CHARS
    """
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)
    _remove_boilerplate(soup)

    # Priority content containers
    content_node = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|article|body|post", re.I))
        or soup.find(class_=re.compile(r"content|article|post|entry|story", re.I))
        or soup.body
        or soup
    )

    # Collect and score paragraphs
    paragraphs: list[tuple[float, str]] = []
    for tag in content_node.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote"]):
        text = tag.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 30:
            continue
        score = _score_block(text)
        if score > 0.1:
            paragraphs.append((score, text))

    if not paragraphs:
        # Fallback: just get all text
        raw = content_node.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", raw).strip()
        return title, text[:MAX_CONTENT_CHARS]

    # Sort by score descending, then reassemble in original order
    # (preserve reading order by keeping top-half scorers)
    threshold = sorted(s for s, _ in paragraphs)[len(paragraphs) // 3]
    kept = [t for s, t in paragraphs if s >= threshold]

    content = "\n\n".join(kept)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return title, content[:MAX_CONTENT_CHARS]


# ── Single URL fetch ──────────────────────────────────────────────────────────

async def fetch_url(url: str) -> FetchResult:
    """
    Fetch a single URL and return a cleaned FetchResult.
    Never raises — errors are captured in FetchResult.error.
    """
    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=TIMEOUT,
        follow_redirects=True,
        max_redirects=4,
    ) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                return FetchResult(
                    url=url, title="", content="", success=False,
                    error=f"Non-HTML content-type: {ct}"
                )
            title, content = _extract_content(resp.text)
            if not content:
                return FetchResult(
                    url=url, title=title, content="", success=False,
                    error="No extractable content"
                )
            return FetchResult(url=url, title=title, content=content, success=True)

        except httpx.TimeoutException:
            return FetchResult(url=url, title="", content="", success=False, error="Timeout")
        except httpx.HTTPStatusError as e:
            return FetchResult(url=url, title="", content="", success=False, error=f"HTTP {e.response.status_code}")
        except Exception as e:
            return FetchResult(url=url, title="", content="", success=False, error=str(e))


# ── Search + fetch pipeline ───────────────────────────────────────────────────

async def search_and_fetch(query: str, max_results: int = 5) -> list[FetchResult]:
    """
    Search DuckDuckGo for a query, fetch top results concurrently,
    return only usable FetchResults (success + enough content).

    This is the main entry point called by server.py's pipeline.
    """
    urls = await _ddg_search(query, max_results=max_results + 2)  # fetch extras in case some fail
    if not urls:
        return []

    # Semaphore caps concurrent fetches
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _guarded_fetch(url: str) -> FetchResult:
        async with sem:
            return await fetch_url(url)

    results = await asyncio.gather(*[_guarded_fetch(u) for u in urls[:max_results + 2]])
    usable = [r for r in results if r.is_usable]
    return usable[:max_results]


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    async def _test() -> None:
        query = " ".join(sys.argv[1:]) or "MCP model context protocol AI agents"
        print(f"\nSearching: '{query}'\n{'─'*60}")
        results = await search_and_fetch(query, max_results=3)
        if not results:
            print("No usable results returned.")
            return
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r.title}")
            print(f"    URL: {r.url}")
            print(f"    Content ({len(r.content)} chars):")
            print("    " + r.content[:300].replace("\n", "\n    ") + "...")

    asyncio.run(_test())
