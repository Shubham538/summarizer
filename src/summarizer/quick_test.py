from tools.render_markdown import render_markdown_html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
file_path = BASE_DIR / "research_findings.txt"


html = render_markdown_html(file_path)
Path("result.html").write_text(html, encoding="utf-8")