# -*- coding: utf-8 -*-
"""把 report.md 转成与 knowledge-base 既有报告一致的暗色 HTML。"""
import re, html, os

BASE = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(BASE, 'report.md'), encoding='utf-8').read()

def inline(s):
    s = html.escape(s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return s

lines = md.split('\n')
out = []
i = 0
FIG = {
    '1-paradigms.html': ('图 1 · 适应下游任务的三种范式坐标：权重训练 / 上下文学习 / 语言程序优化', 600),
    '2-gepa-loop.html': ('图 2 · GEPA 主循环：选模块→跑 minibatch→收集 μ_f 反馈→反思更新→两级闸门', 620),
    '3-pareto.html': ('图 3 · Pareto 选择：实例级得分矩阵、非支配候选与按出现频率采样', 600),
    '4-genealogy.html': ('图 4 · 从 APE 到 GEPA：三代提示优化技术谱系（2022–2026）', 600),
    '5-ecosystem.html': ('图 5 · 2026 GEPA 跟进生态：改进 / 应用 / 诊断 / 替代四路线', 560),
}
def fig_block(key):
    cap, h = FIG[key]
    return f'<figure>\n  <iframe src="diagrams/{key}" height="{h}" loading="lazy"></iframe>\n  <figcaption>{cap}</figcaption>\n</figure>'

embedded = set()
while i < len(lines):
    ln = lines[i]
    s = ln.rstrip()
    if s.startswith('!['):
        i += 1; continue
    if s.startswith('>'):
        buf = []
        while i < len(lines) and lines[i].startswith('>'):
            buf.append(lines[i].lstrip('>').strip())
            i += 1
        txt = ' '.join(b for b in buf if b)
        m = re.findall(r'diagrams/([0-9a-z\-]+\.html)', txt)
        if m and m[0] not in embedded:
            embedded.add(m[0])
            if m[0] in FIG:
                out.append(fig_block(m[0]))
        elif m and m[0] in embedded:
            pass
        else:
            out.append('<blockquote>' + inline(txt) + '</blockquote>')
        continue
    if s.startswith('# '):
        i += 1; continue
    if s == '## 目录':
        i += 1
        while i < len(lines) and not lines[i].startswith('## '):
            i += 1
        continue
    if s.startswith('## '):
        title = s[3:].strip()
        sid = 'sec' + str(len([x for x in out if x.startswith('<h2')]) + 1)
        out.append(f'<h2 id="{sid}">{inline(title)}</h2>')
        i += 1; continue
    if s.startswith('### '):
        out.append('<h3>' + inline(s[4:].strip()) + '</h3>')
        i += 1; continue
    if s.startswith('```'):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith('```'):
            buf.append(lines[i]); i += 1
        i += 1
        out.append('<pre><code>' + html.escape('\n'.join(buf)) + '</code></pre>')
        continue
    if s.startswith('|') and i+1 < len(lines) and re.match(r'^\|[\s:\-\|]+\|$', lines[i+1].strip()):
        hdr = [c.strip() for c in s.strip('|').split('|')]
        i += 2
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            rows.append([c.strip() for c in lines[i].strip('|').split('|')])
            i += 1
        t = ['<table>', '<thead><tr>']
        for c in hdr: t.append('<th>' + inline(c) + '</th>')
        t.append('</tr></thead><tbody>')
        for r in rows:
            t.append('<tr>')
            for c in r: t.append('<td>' + inline(c) + '</td>')
            t.append('</tr>')
        t.append('</tbody></table>')
        out.append('\n'.join(t))
        continue
    if re.match(r'^\d+\. ', s) or s.startswith('- '):
        ordered = bool(re.match(r'^\d+\. ', s))
        items, cur = [], None
        while i < len(lines):
            l2 = lines[i].rstrip()
            m1 = re.match(r'^(\d+)\. (.*)', l2)
            m2 = re.match(r'^- (.*)', l2)
            if ordered and m1:
                cur = [m1.group(2)]; items.append(cur); i += 1
            elif (not ordered) and m2:
                cur = [m2.group(1)]; items.append(cur); i += 1
            elif l2.strip() == '':
                j = i+1
                if j < len(lines) and ((ordered and re.match(r'^\d+\. ', lines[j])) or ((not ordered) and lines[j].startswith('- '))):
                    i += 1; continue
                break
            else:
                if cur is not None: cur.append(l2.strip())
                i += 1
        tag = 'ol' if ordered else 'ul'
        block = [f'<{tag}>']
        for it in items:
            block.append('<li>' + inline(' '.join(x for x in it if x)) + '</li>')
        block.append(f'</{tag}>')
        out.append('\n'.join(block))
        continue
    if s == '---':
        out.append('<hr style="border:none;border-top:1px solid var(--border);margin:36px 0;">')
        i += 1; continue
    if s.strip():
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,3} |```|\||- |\d+\. |>|---|!\[)', lines[i].strip()):
            buf.append(lines[i].strip()); i += 1
        out.append('<p>' + inline(' '.join(buf)) + '</p>')
        continue
    i += 1

body = '\n\n'.join(out)

CSS = """  :root {
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --purple:#bc8cff;
    --amber:#d29922; --rose:#f85149; --cyan:#39c5cf;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,"PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif; line-height:1.85; }
  .container { max-width:980px; margin:0 auto; padding:40px 24px 80px; }
  header { border-bottom:1px solid var(--border); padding-bottom:24px; margin-bottom:28px; }
  h1 { font-size:27px; line-height:1.45; margin-bottom:12px; }
  .meta { color:var(--muted); font-size:13px; }
  .badge { display:inline-block; padding:1px 10px; border-radius:12px; font-size:12px; border:1px solid var(--border); color:var(--accent); margin:0 8px 8px 0; }
  h2 { font-size:22px; margin:48px 0 16px; padding-bottom:8px; border-bottom:1px solid var(--border); }
  h3 { font-size:17px; margin:28px 0 12px; color:var(--accent); }
  p { margin:10px 0; font-size:15px; }
  ul,ol { margin:10px 0 10px 24px; font-size:15px; }
  li { margin:4px 0; }
  code { background:#1f2937; padding:1px 6px; border-radius:4px; font-family:"JetBrains Mono",Menlo,monospace; font-size:13px; color:var(--cyan); }
  pre { background:#0a0e14; border:1px solid var(--border); border-radius:8px; padding:16px; overflow-x:auto; margin:14px 0; font-size:13px; line-height:1.6; }
  pre code { background:none; padding:0; color:#d1d9e0; }
  table { border-collapse:collapse; width:100%; margin:16px 0; font-size:13.5px; }
  th,td { border:1px solid var(--border); padding:8px 12px; text-align:left; vertical-align:top; }
  th { background:var(--panel); color:var(--accent); }
  tr:nth-child(even) td { background:rgba(255,255,255,0.02); }
  blockquote { border-left:3px solid var(--accent); padding:8px 18px; margin:16px 0; background:var(--panel); border-radius:0 8px 8px 0; color:var(--muted); font-size:14px; }
  figure { margin:28px 0; }
  figure iframe { width:100%; border:1px solid var(--border); border-radius:12px; background:#020617; }
  figcaption { text-align:center; color:var(--muted); font-size:13px; margin-top:8px; }
  a { color:var(--accent); text-decoration:none; word-break:break-all; }
  a:hover { text-decoration:underline; }
  .toc { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:20px 28px; margin:24px 0; }
  .toc h2 { border:none; margin:0 0 12px; font-size:16px; }
  .toc ol { margin-left:20px; }
  .toc a { color:var(--text); word-break:normal; }
  .footer { margin-top:60px; padding-top:20px; border-top:1px solid var(--border); color:var(--muted); font-size:12px; }
"""

secs = re.findall(r'<h2 id="(sec\d+)">(.*?)</h2>', body)
toc = ['<nav class="toc"><h2>目录</h2><ol>']
for sid, t in secs:
    toc.append(f'<li><a href="#{sid}">' + re.sub(r'<[^>]+>', '', t) + '</a></li>')
toc.append('</ol></nav>')

header = """<header>
  <h1>反思式提示进化：当提示成为可学习的程序——GEPA 与自动语言程序优化前沿技术报告</h1>
  <div><span class="badge">GEPA</span><span class="badge">反思式提示进化</span><span class="badge">语言程序</span><span class="badge">自动提示优化</span><span class="badge">DSPy 编译</span><span class="badge">TextGrad 文本梯度</span><span class="badge">MIPROv2</span><span class="badge">Pareto 选择</span><span class="badge">System-aware Merge</span><span class="badge">提示级过拟合</span><span class="badge">免梯度优化</span><span class="badge">RLVR/GRPO 对比</span></div>
  <div class="meta" style="margin-top:10px">
    <span>信源：34 篇 arXiv 论文（编号经 arxiv.org 摘要页逐一核实），量化数据均标注作者自述口径</span>
    &nbsp;·&nbsp;<span>报告日期：2026-10-28</span>
  </div>
</header>"""

final = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>反思式提示进化：当提示成为可学习的程序——GEPA 与自动语言程序优化</title>
<style>
{CSS}
</style>
</head><body><div class="container">
{header}
{''.join(toc)}
{body}
<div class="footer">反思式提示进化技术报告 · 2026-10-28 · 所有 arXiv 编号已经 arxiv.org 摘要页逐一核实 · 5 张暗色 SVG · 量化数据除注明外均为各论文作者自述口径</div>
</div></body></html>
"""

open(os.path.join(BASE, 'report.html'), 'w', encoding='utf-8').write(final)
print('report.html bytes:', len(final.encode('utf-8')))
print('h2 count:', len(secs))
print('figures:', body.count('<figure>'))
print('tables:', body.count('<table>'))
