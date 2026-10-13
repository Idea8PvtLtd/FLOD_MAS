"""Build the handbook PDF from the Markdown chapters, the Graphviz diagrams and the charts.

    python docs/handbook/build.py            # everything
    python docs/handbook/build.py --html     # stop after the HTML (to look at it in a browser)

Needs: the ``markdown`` package, ``matplotlib``, Graphviz (``dot``) and Google Chrome or
Chromium (used without a window to print the page to PDF). None of them are needed by the
inspection app.

How a chapter is written
- One file per chapter in ``chapters/``, in file-name order. The first line is ``# Title``.
- A line ``<!-- part: Name -->`` before the title starts a new part of the book.
- A picture on a line of its own, ``![Caption](figures/name.svg)``, becomes a numbered
  figure. ``figures/name.svg`` is drawn from ``diagrams/name.dot`` when that file exists.
- Chapter, section and figure numbers and the contents list are made here; do not type them.
"""
import argparse
import datetime
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
TITLE = 'FLOD_MAS Handbook'
SUBTITLE = 'Architecture and pipelines of the glue-line inspection system'
OUTPUT = HERE / 'FLOD_MAS_Handbook.pdf'
PAGE = HERE / '_build' / 'handbook.html'
CHROME = ('google-chrome', 'chromium', 'chromium-browser', 'chrome')


def draw_diagrams():
    """Every ``diagrams/*.dot`` to ``figures/*.svg``."""
    for source in sorted((HERE / 'diagrams').glob('*.dot')):
        target = HERE / 'figures' / (source.stem + '.svg')
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            continue
        subprocess.run(['dot', '-Tsvg', str(source), '-o', str(target)], check=True)


def draw_charts():
    subprocess.run([sys.executable, str(HERE / 'charts.py')], check=True)


def git(*arguments):
    try:
        return subprocess.run(['git', '-C', str(HERE), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def chapters():
    """``(part or None, number, title, html, sections)`` for every chapter file."""
    found = []
    for number, path in enumerate(sorted((HERE / 'chapters').glob('*.md')), 1):
        text = path.read_text(encoding='utf-8')
        part = re.match(r'\s*<!--\s*part:\s*(.+?)\s*-->', text)
        if part:
            text = text[part.end():]
        converter = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list'])
        body = converter.convert(text)
        title = re.search(r'<h1>(.*?)</h1>', body, re.S)
        if not title:
            raise SystemExit(f'{path.name}: the first line must be "# Title"')
        body = body.replace(title.group(0), '', 1)
        sections = []

        def section(match):
            sections.append(match.group(1))
            index = len(sections)
            return (f'<h2 id="s{number}-{index}"><span class="number">{number}.{index}</span> '
                    f'{match.group(1)}</h2>')
        body = re.sub(r'<h2>(.*?)</h2>', section, body, flags=re.S)
        figures = [0]

        def figure(match):
            caption, source = match.group(1), match.group(2)
            if not (HERE / source).exists():
                raise SystemExit(f'{path.name}: picture not found: {source}')
            figures[0] += 1
            kind = 'photo' if source.lower().endswith(('.jpg', '.jpeg', '.png')) else 'drawing'
            return (f'<figure class="{kind}"><img src="../{source}" alt="">'
                    f'<figcaption><b>Figure {number}.{figures[0]}</b> {caption}</figcaption></figure>')
        body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', figure, body)
        # A short table stays on one page; a long one may break between rows.
        body = re.sub(r'<table>(?=(?:(?!</table>).)*</table>)',
                      lambda match: '<table class="keep">' if body[match.end():body.index('</table>', match.end())].count('<tr') <= 14 else '<table>',
                      body, flags=re.S)
        stray = re.search(r'<img alt=', body)
        if stray:
            raise SystemExit(f'{path.name}: a picture must stand on a line of its own')
        found.append((part.group(1) if part else None, number, title.group(1), body, sections))
    return found


def page(found):
    today = datetime.date.today().isoformat()
    branch, commit = git('rev-parse', '--abbrev-ref', 'HEAD'), git('rev-parse', '--short', 'HEAD')
    contents, body = [], []
    for part, number, title, text, sections in found:
        if part:
            contents.append(f'<li class="part">{html.escape(part)}</li>')
            body.append(f'<section class="part-page"><p>{html.escape(part.split(" - ")[0])}</p>'
                        f'<h1>{html.escape(part.split(" - ")[-1])}</h1></section>')
        contents.append(f'<li class="chapter"><a href="#c{number}"><span>{number}</span>{title}</a></li>')
        contents.extend(f'<li class="section"><a href="#s{number}-{index}">'
                        f'<span>{number}.{index}</span>{name}</a></li>'
                        for index, name in enumerate(sections, 1))
        body.append(f'<section class="chapter"><h1 id="c{number}"><span class="number">{number}</span>'
                    f'{title}</h1>\n{text}</section>')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<link rel="stylesheet" href="../style.css"></head><body>
<section class="cover">
  <p class="kind">Technical handbook</p>
  <h1>{TITLE}</h1>
  <p class="subtitle">{SUBTITLE}</p>
  <table>
    <tr><th>Built</th><td>{today}</td></tr>
    <tr><th>From</th><td>branch <code>{branch}</code>, commit <code>{commit}</code></td></tr>
    <tr><th>Covers</th><td>the inspection application and the glue-line research on that branch,
        and the application on <code>main</code> as read at the commit named in Part D</td></tr>
    <tr><th>Sources</th><td><code>docs/handbook/</code> (chapters, diagrams, build script)</td></tr>
  </table>
</section>
<section class="contents"><h1>Contents</h1><ul>{''.join(contents)}</ul></section>
{''.join(body)}
</body></html>'''


def print_pdf():
    chrome = next((shutil.which(name) for name in CHROME if shutil.which(name)), None)
    if chrome is None:
        raise SystemExit('Google Chrome or Chromium is needed to print the PDF')
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--virtual-time-budget=20000', f'--print-to-pdf={OUTPUT}', PAGE.as_uri()],
                   check=True, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--html', action='store_true', help='stop after writing the HTML page')
    arguments = parser.parse_args()
    (HERE / 'figures').mkdir(exist_ok=True)
    PAGE.parent.mkdir(exist_ok=True)
    draw_diagrams()
    draw_charts()
    PAGE.write_text(page(chapters()), encoding='utf-8')
    print('page:', PAGE)
    if not arguments.html:
        print_pdf()
        print('pdf: ', OUTPUT)


if __name__ == '__main__':
    main()
