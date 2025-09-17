from __future__ import annotations
from typing import Union
from markdown import markdown as md
from pygments.formatters import HtmlFormatter  
import argparse
import sys
from pathlib import Path

BASE_CSS = """
<style>
  body{max-width:860px;margin:2rem auto;font:16px/1.6 system-ui,Segoe UI,Roboto,Arial,sans-serif;color:#111;padding:0 1rem}
  pre{overflow:auto;padding:1rem;border-radius:8px;background:#f6f8fa}
  code{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
  table{border-collapse:collapse;width:100%;margin:1rem 0}
  th,td{border:1px solid #ddd;padding:.5rem;text-align:left}
  blockquote{margin:1rem 0;padding:.5rem 1rem;border-left:4px solid #ddd;color:#555;background:#fafafa}
  h1,h2,h3{line-height:1.25;margin-top:1.5rem}
  img{max-width:100%;height:auto}
</style>
"""

def render_markdown_html(markdown_source: Union[str, Path]) -> str:
    """Convert Markdown text to a standalone HTML string with light styling + Pygments CSS."""

    if isinstance(markdown_source, Path):
        markdown_text = markdown_source.read_text(encoding="utf-8")
    else:
        markdown_text = markdown_source
    
    pyg_css = HtmlFormatter().get_style_defs('.codehilite')
    css = BASE_CSS.replace("</style>", f"\n{pyg_css}\n</style>")
    html_body = md(markdown_text or "", extensions=["tables", "fenced_code", "codehilite"])
    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{html_body}</body></html>"

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render Markdown to standalone HTML.")
    p.add_argument("input", nargs="?", help="Input Markdown file (default: stdin)")
    p.add_argument("-o", "--output", help="Output HTML file (default: stdout)")
    args = p.parse_args(argv)

    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    html = render_markdown_html(text)

    if args.output:
        Path(args.output).write_text(html, encoding="utf-8")
    else:
        sys.stdout.write(html)
    return 0

if __name__ == "__main__":
    raise SystemExit(_main())
