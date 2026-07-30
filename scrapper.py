"""
Wikipedia full-text scraper.

Fetches the full plain-text content of a list of Wikipedia articles using
Wikipedia's official API (action=parse, prop=extracts) and saves each
article to a separate .txt file.

Usage:
    python wiki_scraper.py
"""

import os
import re
import time
import requests

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

LINKS = [
    "https://en.wikipedia.org/wiki/Tesla,_Inc.",
    "https://en.wikipedia.org/wiki/Microsoft",
    "https://en.wikipedia.org/wiki/Nvidia",
    "https://en.wikipedia.org/wiki/Advanced_Micro_Devices",
    "https://en.wikipedia.org/wiki/Google",
]

OUTPUT_DIR = "docs"  # directory to save the .txt files

# Wikipedia requires a descriptive User-Agent identifying the script/contact.
# Replace the email with your own before running this for real.
HEADERS = {
    "User-Agent": "SimpleWikiScraper/1.0 (contact: your-email@example.com)"
}

API_URL = "https://en.wikipedia.org/w/api.php"

REQUEST_DELAY_SECONDS = 1  # be polite between requests


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def url_to_title(url: str) -> str:
    """Extract the page title from a Wikipedia URL, e.g.
    'https://en.wikipedia.org/wiki/Tesla,_Inc.' -> 'Tesla,_Inc.'
    """
    return url.rstrip("/").split("/wiki/")[-1]


def safe_filename(title: str) -> str:
    """Turn a page title into a filesystem-safe filename."""
    name = title.replace("_", " ")
    name = re.sub(r'[\\/*?:"<>|]', "", name)  # strip illegal filename chars
    name = name.strip().rstrip(".")           # trailing dots are risky on some OSes
    return name


def fetch_full_text(title: str) -> str:
    """Fetch the full plain-text extract of a Wikipedia article via the API."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,   # plain text instead of HTML
        "titles": title,
        "redirects": 1,     # follow redirects (e.g. Google -> Google LLC)
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            raise ValueError(f"Page not found: {title}")
        return page.get("extract", "")

    raise ValueError(f"Unexpected API response for: {title}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for url in LINKS:
        title = url_to_title(url)
        print(f"Fetching: {title} ...")

        try:
            text = fetch_full_text(title)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if not text:
            print(f"  WARNING: empty content for {title}")
            continue

        filename = safe_filename(title) + ".txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  Saved -> {filepath} ({len(text)} characters)")

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nDone.")


if __name__ == "__main__":
    main()