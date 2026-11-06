#!/usr/bin/env python3
"""T26: build the report's results tables from results/results.csv.

``tools/benchmark.py`` appends one row per (checkpoint, dataset). This turns
those rows into the five tables the project plan's Results sheet asks for, in
Markdown and LaTeX, so the report is generated rather than transcribed --
transcription is where sign errors and stale numbers get into papers.

Missing rows are printed as em-dashes rather than skipped, so a half-finished
table still shows what is outstanding.

A reminder the tool enforces rather than trusts you to remember: a gain of
about 0.01 S-alpha is a real improvement, while 0.002-0.003 is run-to-run
noise. Differences inside the noise band are flagged, not celebrated.

Usage:
    python tools/make_tables.py                       # all tables, Markdown
    python tools/make_tables.py --table rq1 --format latex
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOISE = 0.005          # S-alpha differences below this are not claimable
METRICS = [('sm', 'S-alpha', 4), ('wfm', 'F-w-beta', 4),
           ('em_mean', 'E-phi', 4), ('mae', 'MAE', 4),
           ('boundary_f', 'bound-F', 4)]

PUBLISHED = {
    'COD10K': dict(sm=0.844, wfm=0.748, em_mean=0.911, mae=0.029),
    'CAMO':   dict(sm=0.836, wfm=0.807, em_mean=0.897, mae=0.066),
    'NC4K':   dict(sm=0.862, wfm=0.824, em_mean=0.913, mae=0.042),
}

# Which rows belong in which table, matched against the --tag given to
# benchmark.py by the scripts in scripts/.
TABLES = {
    'baseline': dict(
        title='Table 1 - Baseline reproduction (COD10K)',
        match=lambda t: 'baseline' in t.lower(),
        datasets=['COD10K'], published=True),
    'part1': dict(
        title='Part 1 - Adaptive multi-band wavelet (COD10K)',
        match=lambda t: bool(re.search(r'\bK=\d', t)),
        datasets=['COD10K'], baseline_tag=r'K=2'),
    'final': dict(
        title='Table 5 - Final model, all test sets',
        match=lambda t: 'final' in t.lower(),
        datasets=['COD10K', 'NC4K', 'CAMO', 'CHAMELEON'], published=True),
}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fnum(row, key, nd=4):
    v = row.get(key, '')
    if v in ('', None):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if x != x else round(x, nd)        # NaN -> None


def cell(v, nd=4):
    return '--' if v is None else '{:.{n}f}'.format(v, n=nd)


def build_rows(rows, spec, dataset):
    """Rows for one table and one dataset, newest run per tag."""
    seen = {}
    for r in rows:
        if r['dataset'] != dataset:
            continue
        tag = r.get('tag', '')
        if not spec['match'](tag):
            continue
        seen[tag] = r                              # later rows win
    return [seen[t] for t in sorted(seen)]


def render(rows, spec, dataset, fmt):
    out = []
    pub = PUBLISHED.get(dataset) if spec.get('published') else None
    cols = ['Config', 'Protocol'] + [m[1] for m in METRICS]

    body = []
    if pub:
        body.append(['Published FEDER (paper)', '384px 159ep bs36'] +
                    [cell(pub.get(k)) for k, _, _ in METRICS])

    base_sm = None
    if spec.get('baseline_tag'):
        for r in rows:
            if re.search(spec['baseline_tag'], r.get('tag', '')):
                base_sm = fnum(r, 'sm')
                break

    notes = []
    for r in rows:
        vals = [fnum(r, k) for k, _, _ in METRICS]
        body.append([r.get('tag', ''), r.get('protocol', '')] +
                    [cell(v) for v in vals])
        sm = vals[0]
        if base_sm is not None and sm is not None and not re.search(
                spec['baseline_tag'], r.get('tag', '')):
            d = sm - base_sm
            flag = 'within noise' if abs(d) < NOISE else 'real'
            notes.append('  {}: d(S-alpha) = {:+.4f} vs K=2 -- {}'.format(
                r.get('tag', ''), d, flag))
        if pub and sm is not None and pub.get('sm'):
            notes.append('  {}: d(S-alpha) = {:+.4f} vs published'.format(
                r.get('tag', ''), sm - pub['sm']))

    if not rows:
        body.append(['(no runs yet)', '--'] + ['--'] * len(METRICS))

    if fmt == 'latex':
        out.append('\\begin{tabular}{ll' + 'r' * len(METRICS) + '}')
        out.append('\\toprule')
        out.append(' & '.join(c.replace('-', '--') for c in cols) + ' \\\\')
        out.append('\\midrule')
        for b in body:
            out.append(' & '.join(str(x) for x in b) + ' \\\\')
        out.append('\\bottomrule')
        out.append('\\end{tabular}')
    else:
        w = [max(len(str(r[i])) for r in [cols] + body) for i in range(len(cols))]
        out.append('| ' + ' | '.join(str(c).ljust(w[i]) for i, c in enumerate(cols)) + ' |')
        out.append('|' + '|'.join('-' * (x + 2) for x in w) + '|')
        for b in body:
            out.append('| ' + ' | '.join(str(x).ljust(w[i]) for i, x in enumerate(b)) + ' |')
    return out, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default='./results/results.csv')
    ap.add_argument('--table', default='all',
                    choices=['all'] + sorted(TABLES))
    ap.add_argument('--format', default='markdown', choices=['markdown', 'latex'])
    ap.add_argument('--out', default='', help='write to this file instead of stdout')
    a = ap.parse_args()

    rows = load(a.csv)
    if not rows:
        print('No results yet at {}.'.format(a.csv))
        print('Run tools/benchmark.py (or the scripts in scripts/) first.')
        return 1

    lines = []
    wanted = sorted(TABLES) if a.table == 'all' else [a.table]
    for name in wanted:
        spec = TABLES[name]
        lines.append('')
        lines.append('## ' + spec['title'])
        for ds in spec['datasets']:
            sub = build_rows(rows, spec, ds)
            if len(spec['datasets']) > 1:
                lines.append('')
                lines.append('### ' + ds)
            lines.append('')
            tbl, notes = render(sub, spec, ds, a.format)
            lines += tbl
            if notes:
                lines.append('')
                lines.append('<!-- differences')
                lines += notes
                lines.append('-->')
    lines.append('')
    lines.append('Report mean E-phi. A gain of ~0.01 S-alpha is a real improvement;')
    lines.append('0.002-0.003 is run-to-run noise and needs multiple seeds to claim.')

    text = '\n'.join(lines)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as f:
            f.write(text + '\n')
        print('wrote {}'.format(a.out))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
