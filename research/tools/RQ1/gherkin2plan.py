#!/usr/bin/env python3
"""BEWT Gherkin (.feature) -> Playwright planner 形式のテストプラン(Markdown) と検証用 JSON

usage: python gherkin2plan.py <BEWT_ROOT> <APP> <OUT.md> <SOURCE.json>
  区切り行は md に出さない．When を含まない区間の Then/And は perform
"""
import glob, json, os, re, subprocess, sys

KW_RE = re.compile(r'^(Given|When|And|Then)\s+(.*)$')
HEAD_RE = re.compile(r'^(Feature|Scenario)\s*:\s*(.*)$')
MARK_RE = re.compile(r'^the previous assertions? passed$')


def sort_key(path):
    name = os.path.basename(path)
    m = re.match(r'^(\d+)_', name)
    return (int(m.group(1)) if m else 10 ** 6, name)


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().replace('\r\n', '\n').split('\n')


def classify(path):
    title, title_line, rows = None, None, []
    for n, raw in enumerate(read_lines(path), 1):
        s = raw.strip()
        if not s:
            continue
        if s.startswith('#'):
            rows.append({'line': n, 'raw': s, 'keyword': None, 'text': s, 'role': 'comment'})
            continue
        m = HEAD_RE.match(s)
        if m:
            if m.group(1) == 'Scenario' or title is None:
                title, title_line = m.group(2).strip(), n
            rows.append({'line': n, 'raw': s, 'keyword': m.group(1), 'text': m.group(2).strip(), 'role': 'heading'})
            continue
        m = KW_RE.match(s)
        if not m:
            sys.exit(f'{path}:{n} キーワードの無い行: {s}')
        rows.append({'line': n, 'raw': s, 'keyword': m.group(1), 'text': m.group(2).strip(), 'role': None})
    if title is None:
        sys.exit(f'{path} に Feature: も Scenario: も無い')

    kws = [r for r in rows if r['role'] is None]
    block = set()
    for i, r in enumerate(kws):
        if r['keyword'] == 'Given' and MARK_RE.match(r['text']):
            j = i + 1
            while j < len(kws) and kws[j]['keyword'] != 'Given':
                j += 1
            if not any(kws[x]['keyword'] == 'When' for x in range(i + 1, j)):
                block.update(id(kws[x]) for x in range(i + 1, j))

    mode = None
    for r in kws:
        k = r['keyword']
        if k in ('Given', 'When'):
            mode = 'perform'
        elif k == 'Then':
            mode = 'expect'
        elif mode is None:
            sys.exit(f'{path}:{r["line"]} 先頭が And')
        if k == 'Given' and MARK_RE.match(r['text']):
            r['role'] = 'marker'
        elif id(r) in block:
            r['role'] = 'perform'
        else:
            r['role'] = mode
    return title, title_line, rows


def build_steps(path, rows):
    steps, last_marker = [], False
    for r in rows:
        if r['role'] == 'marker':
            last_marker = True
        elif r['role'] == 'perform':
            steps.append({'perform': r['text'], 'expect': []})
            r['step'] = len(steps)
            last_marker = False
        elif r['role'] == 'expect':
            if not steps or last_marker:
                sys.exit(f'{path}:{r["line"]} 手順の無い expect')
            steps[-1]['expect'].append(r['text'])
            r['step'], r['expect'] = len(steps), len(steps[-1]['expect'])
    return steps


def render(params):
    lines = [f'# {params["name"]}', '', '## Application Overview', '', params['overview'], '', '## Test Scenarios']
    for i, suite in enumerate(params['suites']):
        lines += ['', f'### {i + 1}. {suite["name"]}', '', f'**Seed:** `{suite["seedFile"]}`']
        for j, test in enumerate(suite['tests']):
            lines += ['', f'#### {i + 1}.{j + 1}. {test["name"]}', '', f'**File:** `{test["file"]}`', '', '**Steps:**']
            for k, st in enumerate(test['steps']):
                p = st.get('perform')
                lines.append(f'  {k + 1}. {p if p is not None else "-"}')
                lines += [f'    - expect: {e}' for e in st['expect']]
    lines.append('')
    return '\n'.join(lines)


def bewt_commit(root):
    try:
        return subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def convert(root, app, out_md):
    if not os.path.isdir(root) or app not in os.listdir(root):
        sys.exit(f'{root} に {app} というフォルダが無い（大文字小文字も区別する）')
    files = sorted(glob.glob(os.path.join(root, app, 'gherkin', '**', '*.feature'), recursive=True), key=sort_key)
    if not files:
        sys.exit(f'{root}/{app}/gherkin に .feature が無い')
    tests, src = [], []
    for f in files:
        base = os.path.basename(f)[:-len('.feature')]
        title, title_line, rows = classify(f)
        steps = build_steps(f, rows)
        tests.append({'name': title, 'file': f'tests/{app}/{base}.spec.ts', 'steps': steps})
        src.append({'id': f'{app}/{base}', 'source': os.path.relpath(f, root).replace(os.sep, '/'),
                    'title': title, 'title_line': title_line, 'lines': rows})
    params = {'name': f'BEWT {app}', 'fileName': os.path.basename(out_md), 'overview': '',
              'suites': [{'name': app, 'seedFile': '', 'tests': tests}]}
    meta = {'app': app, 'bewt_commit': bewt_commit(root), 'params': params, 'tests': src}
    return render(params), meta


def main(argv):
    if len(argv) != 4 or any(a in ('-h', '--help') for a in argv):
        print(__doc__)
        sys.exit(0 if any(a in ('-h', '--help') for a in argv) else 2)
    root, app, out_md, out_json = argv
    md, meta = convert(root, app, out_md)
    for p in (out_md, out_json):
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(out_md, 'w', encoding='utf-8', newline='\n') as f:
        f.write(md)
    with open(out_json, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f'{app}: {len(meta["tests"])} tests -> {out_md}, {out_json}')


if __name__ == '__main__':
    main(sys.argv[1:])
