"""Build the standalone Streamlit usage guide from its Markdown source.

The project deliberately keeps this builder dependency-free so the guide can be
regenerated on a presentation laptop with the standard Python installation.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def inline(text: str) -> str:
    """Render the small inline-Markdown subset used by the guide."""
    placeholders: list[str] = []

    def keep(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(
        r"`([^`]+)`",
        lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"),
        text,
    )
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def slugify(value: str, used: set[str]) -> str:
    plain = re.sub(r"[*`/]", " ", value).strip().lower()
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", plain, flags=re.UNICODE).strip("-")
    slug = slug or "section"
    base = slug
    suffix = 2
    while slug in used:
        slug = f"{base}-{suffix}"
        suffix += 1
    used.add(slug)
    return slug


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_html(markdown: str) -> tuple[str, list[dict[str, str | int]]]:
    lines = markdown.splitlines()
    out: list[str] = []
    toc: list[dict[str, str | int]] = []
    used_ids: set[str] = set()
    paragraph: list[str] = []
    list_type: str | None = None
    in_code = False
    code_lines: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph)
            out.append(f"<p>{inline(text)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip()

        if in_code:
            if line.strip().startswith("```"):
                out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                in_code = False
                code_lines = []
            else:
                code_lines.append(raw)
            index += 1
            continue

        if line.strip().startswith("```"):
            flush_paragraph()
            close_list()
            in_code = True
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            section_id = slugify(title, used_ids)
            out.append(f'<h{level} id="{section_id}">{inline(title)}</h{level}>')
            if level in (2, 3):
                toc.append({"level": level, "title": re.sub(r"[*`]", "", title), "id": section_id})
            index += 1
            continue

        if re.match(r"^\s*---+\s*$", line):
            flush_paragraph()
            close_list()
            out.append("<hr>")
            index += 1
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            close_list()
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quote_lines.append(lines[index].lstrip()[1:].strip())
                index += 1
            out.append(f'<blockquote>{"<br>".join(inline(item) for item in quote_lines)}</blockquote>')
            continue

        if (
            line.strip().startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1])
        ):
            flush_paragraph()
            close_list()
            headers = split_table_row(line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index]))
                index += 1
            out.append('<div class="table-wrap"><table><thead><tr>')
            out.extend(f"<th>{inline(cell)}</th>" for cell in headers)
            out.append("</tr></thead><tbody>")
            for row in rows:
                row += [""] * (len(headers) - len(row))
                out.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row[: len(headers)]) + "</tr>")
            out.append("</tbody></table></div>")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        number = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet or number:
            flush_paragraph()
            wanted = "ul" if bullet else "ol"
            if list_type != wanted:
                close_list()
                list_type = wanted
                out.append(f"<{wanted}>")
            value = (bullet or number).group(1)
            out.append(f"<li>{inline(value)}</li>")
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            close_list()
            index += 1
            continue

        paragraph.append(line[:-2] + "<br>" if line.endswith("  ") else line)
        index += 1

    flush_paragraph()
    close_list()
    if in_code:
        out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(out), toc


def build_document(body: str, toc: list[dict[str, str | int]]) -> str:
    nav = []
    for item in toc:
        cls = "nav-major" if item["level"] == 2 else "nav-minor"
        nav.append(
            f'<a class="{cls}" href="#{item["id"]}">{html.escape(str(item["title"]))}</a>'
        )
    return fr"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Bilingual operational and presentation guide for the Telco Customer Churn Streamlit prototype.">
  <title>Telco Churn · Streamlit Guide</title>
  <style>
    :root {{ --navy:#14263f; --blue:#2f78bb; --green:#2d8d61; --amber:#e8a317;
      --red:#c94b48; --ink:#253246; --muted:#6d7786; --paper:#f4f7fb;
      --surface:#fff; --line:#dce4ee; --soft:#edf4fb; --sidebar:310px; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; scroll-padding-top:82px; }}
    body {{ margin:0; background:var(--paper); color:var(--ink);
      font:15.5px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    .progress {{ position:fixed; z-index:60; inset:0 0 auto; height:4px; }}
    .progress span {{ display:block; width:0; height:100%; background:linear-gradient(90deg,var(--blue),var(--green)); }}
    header {{ position:fixed; z-index:50; inset:4px 0 auto; height:68px; display:flex;
      align-items:center; gap:18px; padding:10px 22px; color:#fff; background:rgba(20,38,63,.97);
      box-shadow:0 8px 28px rgba(20,38,63,.18); }}
    .brand {{ min-width:275px; }} .brand strong {{ display:block; font-size:15px; }}
    .brand span {{ opacity:.7; font-size:12px; }}
    .search {{ flex:1; max-width:640px; }}
    .search input {{ width:100%; border:1px solid rgba(255,255,255,.2); border-radius:10px;
      padding:10px 13px; color:white; background:rgba(255,255,255,.1); outline:none; }}
    .search input::placeholder {{ color:rgba(255,255,255,.62); }}
    .actions button {{ border:1px solid rgba(255,255,255,.24); border-radius:9px; padding:9px 12px;
      color:white; background:transparent; cursor:pointer; }}
    aside {{ position:fixed; inset:72px auto 0 0; width:var(--sidebar); overflow:auto; padding:18px 15px 30px;
      border-right:1px solid var(--line); background:#fff; }}
    aside .label {{ margin:2px 10px 12px; color:var(--muted); font-size:11px; font-weight:800;
      letter-spacing:.12em; text-transform:uppercase; }}
    aside a {{ display:block; border-radius:9px; color:#465269; text-decoration:none; }}
    aside a:hover, aside a.active {{ color:#174f83; background:#e8f2fc; }}
    .nav-major {{ margin-top:9px; padding:9px 10px; font-weight:800; }}
    .nav-minor {{ padding:6px 10px 6px 21px; font-size:12.5px; }}
    main {{ margin-left:var(--sidebar); padding:98px 34px 70px; }}
    article {{ max-width:1180px; margin:auto; }}
    h1 {{ margin:0 0 8px; color:var(--navy); font-size:clamp(30px,4vw,48px); line-height:1.15; }}
    h1 + blockquote {{ margin-top:20px; }}
    h2 {{ margin:54px 0 22px; padding:22px 25px; border-radius:18px; color:#fff;
      background:linear-gradient(120deg,var(--navy),#285b85); box-shadow:0 14px 38px rgba(20,38,63,.13); }}
    h3 {{ margin:30px 0 12px; padding:17px 20px; border:1px solid var(--line); border-left:5px solid var(--blue);
      border-radius:13px; color:var(--navy); background:var(--surface); box-shadow:0 6px 20px rgba(20,38,63,.05); }}
    h4 {{ margin:23px 0 8px; color:#234e76; font-size:16px; }}
    p, ul, ol {{ margin:9px 4px 14px; }} li {{ margin:5px 0; }}
    strong {{ color:#172b46; }} code {{ padding:.13em .4em; border-radius:6px; color:#a12d40; background:#f8edf1; }}
    pre {{ overflow:auto; margin:16px 0; padding:18px; border-radius:13px; color:#e9f2ff; background:#17283f; }}
    pre code {{ padding:0; color:inherit; background:none; }}
    blockquote {{ margin:16px 0; padding:16px 19px; border:1px solid #cfe0ef; border-left:5px solid var(--green);
      border-radius:12px; color:#31475e; background:#edf7f3; }}
    hr {{ margin:38px 0; border:0; border-top:1px solid var(--line); }}
    .table-wrap {{ overflow:auto; margin:15px 0 22px; border:1px solid var(--line); border-radius:13px; background:#fff; }}
    table {{ width:100%; border-collapse:collapse; min-width:620px; }}
    th,td {{ padding:11px 13px; border-bottom:1px solid #e7edf4; text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; color:#173b5f; background:#edf4fb; font-size:13px; }}
    tr:last-child td {{ border-bottom:0; }} tbody tr:nth-child(even) {{ background:#fafcff; }}
    mark {{ border-radius:3px; padding:0 2px; background:#ffe69a; }}
    .empty {{ display:none; margin:20px 0; padding:14px; border-radius:10px; background:#fff3cd; }}
    @media(max-width:900px) {{ aside {{ display:none; }} main {{ margin-left:0; padding:92px 18px 50px; }}
      .brand {{ min-width:190px; }} .brand span {{ display:none; }} }}
    @media(max-width:620px) {{ header {{ padding:9px 12px; }} .actions {{ display:none; }} h2 {{ padding:18px; }} }}
    @media print {{ header,aside,.progress {{ display:none; }} main {{ margin:0; padding:0; }}
      h2 {{ break-before:page; color:var(--navy); background:#e9f1f8; box-shadow:none; }}
      h3,.table-wrap,blockquote {{ break-inside:avoid; box-shadow:none; }} }}
  </style>
</head>
<body>
  <div class="progress"><span></span></div>
  <header>
    <div class="brand"><strong>Telco Churn · Streamlit Guide</strong><span>System operation, graph reading and defence</span></div>
    <div class="search"><input id="search" type="search" placeholder="Search / 搜索：K-fold, threshold, VIF, Recall…"></div>
    <div class="actions"><button onclick="window.print()">Print / PDF</button></div>
  </header>
  <aside><div class="label">Contents / 目录</div>{''.join(nav)}</aside>
  <main><article id="guide">{body}<div class="empty" id="empty">No matching content / 没有找到相关内容</div></article></main>
  <script>
    const progress=document.querySelector('.progress span');
    const nav=[...document.querySelectorAll('aside a')];
    const sections=[...document.querySelectorAll('h2[id],h3[id]')];
    const search=document.getElementById('search');
    const guide=document.getElementById('guide');
    const source=guide.innerHTML;
    addEventListener('scroll',()=>{{
      const max=document.documentElement.scrollHeight-innerHeight;
      progress.style.width=(max?scrollY/max*100:0)+'%';
      let current=sections[0]?.id;
      for(const section of sections) if(section.getBoundingClientRect().top<120) current=section.id;
      nav.forEach(a=>a.classList.toggle('active',a.hash==='#'+current));
    }},{{passive:true}});
    search.addEventListener('input',()=>{{
      const q=search.value.trim();
      guide.innerHTML=source;
      if(!q){{ document.getElementById('empty').style.display='none'; return; }}
      const walker=document.createTreeWalker(guide,NodeFilter.SHOW_TEXT);
      const nodes=[]; while(walker.nextNode()) nodes.push(walker.currentNode);
      let hits=0; const escaped=q.replace(/[.*+?^${{}}()|[\]\\]/g,'\\$&'); const re=new RegExp(escaped,'ig');
      for(const node of nodes){{ if(['SCRIPT','STYLE','CODE'].includes(node.parentElement?.tagName)) continue;
        const value=node.nodeValue; if(!re.test(value)) continue; re.lastIndex=0; hits++;
        const span=document.createElement('span'); span.innerHTML=value.replace(re,m=>'<mark>'+m+'</mark>'); node.replaceWith(span); }}
      const empty=document.getElementById('empty'); empty.style.display=hits?'none':'block';
      document.querySelector('mark')?.scrollIntoView({{behavior:'smooth',block:'center'}});
    }});
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="STREAMLIT_USAGE_AND_UNDERSTANDING_GUIDE.md")
    parser.add_argument("--output", default="STREAMLIT_USAGE_AND_UNDERSTANDING_GUIDE.html")
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    body, toc = markdown_to_html(source.read_text(encoding="utf-8"))
    output.write_text(build_document(body, toc), encoding="utf-8")
    print(f"Built {output} from {source} ({len(toc)} navigation entries).")


if __name__ == "__main__":
    main()
