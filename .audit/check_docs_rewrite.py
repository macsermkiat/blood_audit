from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
import re
import json

root = Path(__file__).resolve().parents[1]
dist = root / "docs/dist"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if "href" in a and (
            tag == "a" or tag == "link" and a.get("rel") == "stylesheet"
        ):
            self.links.append(a["href"])


pages = {p: Page(p.read_text()) for p in dist.rglob("*.html")}
if not pages:
    raise SystemExit(
        "Build the documentation with npm run build before checking links."
    )
broken = []
refs = 0
for path, page in pages.items():
    rel = path.relative_to(dist).as_posix()
    url = "https://macsermkiat.github.io/blood_audit/" + rel.removesuffix("index.html")
    for link in page.links:
        u = urlsplit(urljoin(url, link))
        if u.netloc != "macsermkiat.github.io" or not u.path.startswith(
            "/blood_audit/"
        ):
            continue
        refs += 1
        part = unquote(u.path.removeprefix("/blood_audit/"))
        dest = dist / part
        if dest.is_dir():
            dest = dest / "index.html"
        elif not dest.exists():
            dest = dest / "index.html"
        if not dest.exists():
            broken.append((rel, link, "missing target"))
            continue
        if (
            u.fragment
            and dest.suffix == ".html"
            and unquote(u.fragment) not in pages[dest].ids
        ):
            broken.append((rel, link, "missing anchor"))
current = [
    root / p
    for p in [
        "README.md",
        "PRODUCT.md",
        "DESIGN.md",
        "docs/CONTEXT.md",
        "docs/production-cli.md",
        "docs/hosxp-export-reference.md",
        "docs/report-schema.md",
    ]
] + list((root / "docs/src/content/docs").rglob("*.mdx"))
md_broken = []
dashes = []
for p in current:
    s = p.read_text()
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    for link in re.findall(r"\]\(([^\s()]+)\)", s):
        if link.startswith(("http:", "https:", "mailto:", "#", "/blood_audit/")):
            continue
        target = unquote(link.split("#", 1)[0])
        if target and not (p.parent / target).exists():
            md_broken.append((str(p.relative_to(root)), link))
    if any(c in s for c in "\u2013\u2014"):
        dashes.append(str(p.relative_to(root)))
result = {
    "built_html_pages": len(pages),
    "local_href_checks": refs,
    "broken_html_links": broken,
    "current_markdown_files": len(current),
    "broken_repository_links": md_broken,
    "long_dashes_outside_code": dashes,
}
(root / ".audit/docs-rewrite-links.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(bool(broken or md_broken or dashes))
