"""Look up a reference by DOI: bibliographic details from Crossref, abstract from OpenAlex.

Usage:  python lookup_reference.py <doi> [<doi> ...]
        python lookup_reference.py --title "words from the title"

Used to check every reference before it is cited (see ../reference_verification.md).
Only public metadata services are queried; nothing is sent except the DOI or title.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "thesis-reference-check/1.0"}


def get(url: str):
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return {"error": f"HTTP {error.code}"}
        except Exception as error:  # noqa: BLE001
            return {"error": str(error)}
    return {"error": "no response"}


def crossref(doi: str) -> None:
    data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    if "error" in data:
        print("  Crossref :", data["error"])
        return
    m = data["message"]
    authors = "; ".join(f"{a.get('given', '')} {a.get('family', a.get('name', ''))}".strip() for a in m.get("author", []))
    issued = m.get("issued", {}).get("date-parts", [[None]])[0]
    print("  title    :", " ".join(m.get("title", [""])), "|", " ".join(m.get("subtitle", [])))
    print("  authors  :", authors)
    print("  venue    :", " | ".join(m.get("container-title", [])))
    print(f"  year {issued} | vol {m.get('volume')} | issue {m.get('issue')} | pages {m.get('page')} | "
          f"article {m.get('article-number')} | {m.get('type')} | {m.get('publisher')}")


def openalex_abstract(doi: str) -> None:
    data = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi, safe="/:"))
    if "error" in data:
        print("  OpenAlex :", data["error"])
        return
    index = data.get("abstract_inverted_index") or {}
    words = sorted((position, word) for word, positions in index.items() for position in positions)
    abstract = " ".join(word for _, word in words)
    location = (data.get("best_oa_location") or {}).get("landing_page_url")
    print("  open copy:", location)
    if abstract:
        print("  abstract (OpenAlex):", abstract)
        return
    # Fall back to Semantic Scholar, which holds abstracts for many publishers.
    time.sleep(1.2)
    s2 = get("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi, safe="/:")
             + "?fields=abstract,openAccessPdf")
    if "error" in s2:
        print("  abstract : (none in OpenAlex; Semantic Scholar:", s2["error"] + ")")
        return
    pdf = (s2.get("openAccessPdf") or {}).get("url")
    if pdf:
        print("  open copy:", pdf)
    print("  abstract (Semantic Scholar):", s2.get("abstract") or "(none)")


def search(title: str) -> None:
    data = get("https://api.crossref.org/works?rows=4&query.bibliographic=" + urllib.parse.quote(title))
    for m in data.get("message", {}).get("items", []):
        authors = "; ".join(a.get("family", a.get("name", "")) for a in m.get("author", []))
        issued = m.get("issued", {}).get("date-parts", [[None]])[0]
        print("  -", " ".join(m.get("title", [""])), "|", authors, "|", " | ".join(m.get("container-title", [])),
              "|", issued, "| DOI", m.get("DOI"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # abstracts contain non-ASCII characters
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--title":
        for title in arguments[1:]:
            print(f"\nSEARCH: {title}")
            search(title)
    else:
        for doi in arguments:
            print(f"\n=== {doi}")
            crossref(doi)
            openalex_abstract(doi)
            time.sleep(0.4)
