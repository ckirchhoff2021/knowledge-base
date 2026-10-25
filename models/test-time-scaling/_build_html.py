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
    '1-timeline.html': ('图 1 · TTS 技术谱系时间线（2022→2026）：从 CoT 萌芽到 o1/R1 产品化，再到自适应与可证明缩放', 600),
    '2-two-modes.html': ('图 2 · 两大计算模式与搜索结构：顺序修订（深度）vs 并行采样（广度），及树图搜索与投机服务化', 600),
    '3-adaptive.html': ('图 3 · 自适应计算决策框架：触发 → 分配 → 监控 → 停机的闭环控制', 600),
    '4-latent.html': ('图 4 · 显式 CoT（语言空间）× 潜空间推理（连续向量）：思考的两种载体', 590),
    '5-fullstack.html': ('图 5 · TTS 全栈四层（训练→编排→验证→服务）与「推理经济」成本-采用反馈环', 630),
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
  <h1>Test-Time Scaling 前沿技术报告：当思考本身成为新的扩展轴——从 o1、DeepSeek-R1 到自适应计算、潜空间推理与测试时学习</h1>
  <div><span class="badge">Test-Time Scaling</span><span class="badge">推理模型</span><span class="badge">o1/R1</span><span class="badge">budget forcing</span><span class="badge">并行采样</span><span class="badge">MCTS</span><span class="badge">过程奖励 PRM</span><span class="badge">验证器</span><span class="badge">过度思考</span><span class="badge">自适应计算</span><span class="badge">早停</span><span class="badge">潜空间推理</span><span class="badge">Coconut</span><span class="badge">测试时学习</span><span class="badge">推理经济</span></div>
  <div class="meta" style="margin-top:10px">
    <span>信源：65 篇专题 arXiv 论文（编号经 arxiv.org 摘要页逐一核实）+ Anthropic/Google/Qwen 官方博客 + ARC Prize 一手测试报告 + Artificial Analysis 第三方聚合</span>
    &nbsp;·&nbsp;<span>报告日期：2026-10-25</span>
  </div>
</header>"""

final = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Test-Time Scaling 前沿技术报告：从 o1、R1 到自适应计算与潜空间推理</title>
<style>
{CSS}
</style>
</head><body><div class="container">
{header}
{''.join(toc)}
{body}
<div class="footer">Test-Time Scaling 前沿技术报告 · 2026-10-25 · 所有 arXiv 编号已经 arxiv.org 摘要页逐一核实 · 图表为手工暗色 SVG · 厂商自报数据标注「官方口径」，第三方聚合标注「第三方」</div>
</div></body></html>
"""

open(os.path.join(BASE, 'report.html'), 'w', encoding='utf-8').write(final)
print('report.html bytes:', len(final.encode('utf-8')))
print('h2 count:', len(secs))
print('figures:', body.count('<figure>'))
print('tables:', body.count('<table>'))
