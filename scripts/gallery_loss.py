"""Export recorded training loss into a self-contained gallery panel (stdlib only)."""
import argparse
import html
import json
import math
from pathlib import Path


def loss_panel(payload):
    rows = payload['records']
    ymax = max(r['loss_mean'] for r in rows) * 1.08
    xmin, xmax = rows[0]['step'], max(rows[0]['step'] + 1, rows[-1]['step'])
    def x(step):
        return 65 + 950 * (step - xmin) / (xmax - xmin)
    def y(value):
        return 255 - 220 * value / max(ymax, 1e-9)
    svg = ['<svg viewBox="0 0 1050 300" role="img" aria-label="PDD training loss" style="width:100%;max-height:350px">']
    for i in range(6):
        value = ymax * i / 5
        step = xmin + (xmax - xmin) * i / 5
        svg.append(f'<path d="M65 {y(value):.2f} H1015" stroke="#555"/><text x="5" y="{y(value)+4:.2f}" fill="#ddd" font-size="13">{value:.3f}</text><text x="{x(step):.2f}" y="280" fill="#ddd" font-size="13">{step:.0f}</text>')
    means = [sum(r['loss_mean'] for r in rows[max(0, i-9):i+1]) / min(i+1, 10) for i in range(len(rows))]
    for values, color in [( [r['loss_mean'] for r in rows], '#6bbcff'), (means, '#ffb35c')]:
        points = ' '.join(f"{x(r['step']):.2f},{y(v):.2f}" for r, v in zip(rows, values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    step = payload['checkpoint_step']
    if xmin <= step <= xmax:
        svg.append(f'<path d="M{x(step):.2f} 25 V255" stroke="white" stroke-dasharray="5 5"/><text x="{x(step)-5:.2f}" y="20" text-anchor="end" fill="white">Checkpoint {step}</text>')
    svg.append('</svg>')
    table = ''.join(f"<tr><td>{r['step']}</td><td>{r['loss_mean']:.6f}</td></tr>" for r in rows)
    return ('<!-- training-loss --><div style="background:#20252b;color:#eee;padding:18px;margin:20px 0;max-width:1160px">'
            f'<h2>PDD 训练 Loss · {xmin}–{rows[-1]["step"]} step</h2><p>评估 checkpoint：{step}；最新记录 loss：{rows[-1]["loss_mean"]:.6f}</p>'
            + ''.join(svg) + '<p>蓝线：每 step 跨 rank 平均 loss；橙线：10 step 滑动均值；虚线：评估 checkpoint。横轴为 optimizer step。</p>'
            + '<p>训练轨迹蒸馏 loss，不代表视频质量；当前日志快照。<a style="color:#8cd3ff" href="training_loss.html">交互曲线</a> · <a style="color:#8cd3ff" href="training_loss.json">下载数据</a></p>'
            + '<details><summary>展开全部 loss 数值</summary><div style="max-height:250px;overflow:auto"><table><thead><tr><th>Step</th><th>Loss mean</th></tr></thead><tbody>' + table + '</tbody></table></div></details></div><!-- /training-loss -->')


def attach(root, payload):
    for name in ('comparison.html', 'index.html'):
        path = Path(root) / name
        if not path.exists():
            continue
        page = path.read_text()
        marker, end = '<!-- training-loss -->', '<!-- /training-loss -->'
        if marker in page:
            begin = page.index(marker)
            finish = page.index(end, begin) + len(end)
            page = page[:begin] + loss_panel(payload) + page[finish:]
        else:
            begin = page.index('</h1>') + len('</h1>')
            page = page[:begin] + loss_panel(payload) + page[begin:]
        path.write_text(page)


def build(root):
    root = Path(root)
    cfg = json.loads((root / 'evaluation_config.json').read_text())
    checkpoint = Path(cfg['arguments']['student'])
    source = checkpoint.parent / 'metrics.jsonl'
    if not source.exists():
        raise FileNotFoundError(source)
    by_step = {}
    for line in source.read_text().splitlines():
        r = json.loads(line)
        if 'step' in r and 'loss_mean' in r and math.isfinite(r['loss_mean']):
            by_step[int(r['step'])] = {'step': int(r['step']), 'loss_mean': r['loss_mean']}
    rows = sorted(by_step.values(), key=lambda r: r['step'])
    if not rows:
        raise ValueError(f'No finite loss records: {source}')
    checkpoint_step = int(checkpoint.name.removeprefix('step_'))
    payload = {'source': str(source), 'checkpoint_step': checkpoint_step, 'records': rows}
    (root / 'training_loss.json').write_text(json.dumps(payload, indent=2) + '\n')
    data = json.dumps(payload).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDD training loss</title>
<style>body{font:15px system-ui;background:#181818;color:#eee;margin:12px}a{color:#8cd3ff}svg{width:100%;height:auto;max-height:330px}button{background:#333;color:#eee;padding:6px;margin:0 8px 8px 0;border:1px solid #666}p{margin:8px 0;overflow-wrap:anywhere}.muted{color:#bbb}</style>
<h2>PDD 训练 Loss</h2><p id="summary"></p><p><button id="scope">仅显示至评估 checkpoint</button><a href="training_loss.json" download>下载 loss 数据</a></p>
<svg id="chart" viewBox="0 0 1100 320" role="img" aria-label="Training loss by optimizer step"></svg><p id="hover">将鼠标移到曲线上查看具体 step 和 loss。</p>
<p>蓝色：每个 optimizer step 的 loss_mean（跨 rank 均值）；橙色：最近 10 个记录的滑动均值；虚线：评估 checkpoint。</p><p class="muted">这是训练轨迹蒸馏 loss，不是视频质量指标。数据为生成页面时的快照。</p><details><summary>日志来源</summary>SOURCE</details>
<script>const data=DATA;let limited=false;const svg=document.getElementById('chart');const ns='http://www.w3.org/2000/svg';
function el(tag,attrs,text){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.appendChild(n);return n}
function draw(){svg.replaceChildren();const rows=data.records.filter(r=>!limited||r.step<=data.checkpoint_step);if(!rows.length)return;const lo=rows[0].step,hi=Math.max(lo+1,rows.at(-1).step),max=Math.max(1e-9,...rows.map(r=>r.loss_mean))*1.08;const x=s=>70+(s-lo)/(hi-lo)*990,y=v=>270-v/max*245;
for(let i=0;i<=5;i++){const v=max*i/5;el('line',{x1:70,x2:1060,y1:y(v),y2:y(v),stroke:'#444'});el('text',{x:62,y:y(v)+4,fill:'#bbb','text-anchor':'end','font-size':12},v.toFixed(3));const s=lo+(hi-lo)*i/5;el('text',{x:x(s),y:292,fill:'#bbb','text-anchor':'middle','font-size':12},Math.round(s))}
el('text',{x:550,y:315,fill:'#bbb','text-anchor':'middle'},'Optimizer step');
const avg=rows.map((r,i)=>rows.slice(Math.max(0,i-9),i+1).reduce((s,q)=>s+q.loss_mean,0)/Math.min(i+1,10));
for(const [values,color] of [[rows.map(r=>r.loss_mean),'#6bbcff'],[avg,'#ffb35c']])el('polyline',{points:rows.map((r,i)=>`${x(r.step)},${y(values[i])}`).join(' '),fill:'none',stroke:color,'stroke-width':2});
if(data.checkpoint_step>=lo&&data.checkpoint_step<=hi){el('line',{x1:x(data.checkpoint_step),x2:x(data.checkpoint_step),y1:15,y2:270,stroke:'#eee','stroke-dasharray':'5 5'});el('text',{x:x(data.checkpoint_step)-5,y:13,fill:'#eee','text-anchor':'end'},`Checkpoint ${data.checkpoint_step}`)}
rows.forEach((r,i)=>{const n=el('circle',{cx:x(r.step),cy:y(r.loss_mean),r:5,fill:'transparent',tabindex:0});const title=document.createElementNS(ns,'title');title.textContent=`Step ${r.step}: ${r.loss_mean.toFixed(6)}`;n.appendChild(title);const show=()=>document.getElementById('hover').textContent=`Step ${r.step} · loss ${r.loss_mean.toFixed(6)} · moving mean ${avg[i].toFixed(6)}`;n.onmouseenter=show;n.onfocus=show});
const ck=data.records.find(r=>r.step===data.checkpoint_step);document.getElementById('summary').textContent=`记录 ${data.records.length} steps（${data.records[0].step}–${data.records.at(-1).step}）；评估 checkpoint ${data.checkpoint_step}${ck?'，loss '+ck.loss_mean.toFixed(6):''}。`;}
document.getElementById('scope').onclick=e=>{limited=!limited;e.target.textContent=limited?'显示全部训练记录':'仅显示至评估 checkpoint';draw()};draw();</script></html>'''
    (root / 'training_loss.html').write_text(page.replace('SOURCE', html.escape(str(source))).replace('DATA', data))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = build(args.directory)
    attach(args.directory, result)
    print(f"Exported {len(result['records'])} loss records; checkpoint {result['checkpoint_step']}")
